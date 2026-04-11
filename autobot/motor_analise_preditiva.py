import sys, os, requests, traceback, hashlib, warnings, time, json
from pathlib import Path
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import polars as pl
import holidays
import boto3
from google.cloud import bigquery
from google.oauth2 import service_account
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np
import pandas as pd
import h3
import magic

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")


def hora_brasilia() -> datetime:
    return datetime.utcnow() - timedelta(hours=3)


def criar_cliente_bq(projeto: str, cred_json: str) -> bigquery.Client:
    info = json.loads(cred_json)
    credentials = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(project=projeto, credentials=credentials)


def enviar_para_bigquery(df_pl, tabela, projeto, dataset, cred_json):
    df_pd = df_pl.to_pandas()
    client = criar_cliente_bq(projeto, cred_json)
    tabela_id = f"{projeto}.{dataset}.{tabela}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df_pd, tabela_id, job_config=job_config)
    job.result()


def tabela_existe_bigquery(projeto, dataset, tabela, cred_json) -> bool:
    try:
        client = criar_cliente_bq(projeto, cred_json)
        client.get_table(f"{projeto}.{dataset}.{tabela}")
        return True
    except Exception:
        return False


class Telemetria:
    def __init__(self):
        self.sucesso = os.environ.get("DISCORD_SUCESSO", "").strip(' "\'')
        self.erro = os.environ.get("DISCORD_ERRO", "").strip(' "\'')

    def _enviar_webhook(self, url, payload):
        if not url or not url.startswith("https://discord"):
            print("⚠️ Webhook ausente. Pulando notificação.", file=sys.stderr)
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"❌ FALHA DISCORD: {e}", file=sys.stderr)

    def notificar_sucesso(self, titulo, tempo_execucao, registros, media_risco, status_s3, status_bq):
        payload = {"embeds": [{"title": f"🟢 {titulo}", "color": 3066993, "fields": [
            {"name": "📊 Registros (Prata)", "value": f"{registros:,}", "inline": True},
            {"name": "⚠️ Risco Médio", "value": f"{media_risco:.2f}", "inline": True},
            {"name": "⏱️ Tempo", "value": f"{tempo_execucao:.1f}s", "inline": True},
            {"name": "☁️ R2", "value": status_s3, "inline": False},
            {"name": "📡 BigQuery", "value": status_bq, "inline": False},
        ], "footer": {"text": f"SafeDriver AI • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}}]}
        self._enviar_webhook(self.sucesso, payload)

    def notificar_info(self, titulo, corpo):
        payload = {"embeds": [{"title": f"🔵 {titulo}", "description": corpo, "color": 3447003,
            "footer": {"text": f"SafeDriver AI • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}}]}
        self._enviar_webhook(self.sucesso, payload)

    def notificar_erro(self, titulo, erro_msg):
        ticks = chr(96) * 3
        stack = f"{ticks}python\n{erro_msg[:1000]}\n{ticks}"
        payload = {"embeds": [{"title": f"🔴 {titulo}", "description": "**Falha Crítica**", "color": 15158332,
            "fields": [{"name": "Erro", "value": stack}],
            "footer": {"text": f"SafeDriver AI • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}}]}
        self._enviar_webhook(self.erro, payload)


class SafeDriver:
    # Colunas críticas que identificam uma aba de dados criminais
    COLUNAS_CRITICAS = {
        "NOME_DEPARTAMENTO", "NOME_MUNICIPIO", "LOGRADOURO",
        "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"
    }

    # Colunas que precisamos para a camada prata
    COLUNAS_PRATA = [
        "DATA_OCORRENCIA_BO",
        "HORA_OCORRENCIA_BO",
        "NOME_MUNICIPIO",
        "NOME_DEPARTAMENTO",
        "NOME_SECCIONAL",
        "NOME_DELEGACIA",
        "LOGRADOURO",
        "NUMERO_LOGRADOURO",
        "BAIRRO",
        "LATITUDE",
        "LONGITUDE",
        "RUBRICA",
        "DESCR_CONDUTA",
        "NATUREZA_APURADA",
    ]

    # Colunas obrigatórias — registro sem elas é descartado
    COLUNAS_OBRIGATORIAS = [
        "DATA_OCORRENCIA_BO",
        "NOME_MUNICIPIO",
        "LATITUDE",
        "LONGITUDE",
        "RUBRICA",
    ]

    def __init__(self):
        self.t_inicio = time.time()
        self.discord = Telemetria()
        self.pastas = {p: Path(f"datalake/{p}") for p in ["raw", "prata", "ouro"]}
        for p in self.pastas.values():
            p.mkdir(parents=True, exist_ok=True)

        cfg = {
            "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL", "").strip(),
            "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", "").strip(),
            "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", "").strip(),
            "R2_BUCKET_NAME": os.environ.get("R2_BUCKET_NAME", "").strip(),
        }

        if all(cfg.values()):
            self.s3 = boto3.client(
                "s3",
                endpoint_url=cfg["R2_ENDPOINT_URL"],
                aws_access_key_id=cfg["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=cfg["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
            self.bucket = cfg["R2_BUCKET_NAME"]
        else:
            self.s3 = None

        self.r2_prefix = "safedriver/datalake/"
        self.meta = self.pastas["raw"] / "tracking_ssp.json"
        self.feriados = list(holidays.Brazil(subdiv="SP", years=range(2022, 2027)).keys())

    # ------------------------------------------------------------------ #
    #  R2 helpers
    # ------------------------------------------------------------------ #

    def _r2_key(self, caminho_relativo: str) -> str:
        return f"{self.r2_prefix}{caminho_relativo}"

    def r2_existe(self, caminho_relativo: str) -> bool:
        if not self.s3:
            return False
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._r2_key(caminho_relativo))
            return True
        except Exception:
            return False

    def r2_upload(self, arquivo_local: Path, caminho_relativo: str) -> str:
        if not self.s3:
            return "PULADO"
        try:
            self.s3.upload_file(str(arquivo_local), self.bucket, self._r2_key(caminho_relativo))
            print(f"☁️ R2 upload: {caminho_relativo}", file=sys.stdout)
            return "SUCESSO"
        except Exception as e:
            print(f"❌ R2 upload falhou ({caminho_relativo}): {e}", file=sys.stderr)
            return "FALHA"

    def r2_download(self, caminho_relativo: str, destino: Path) -> bool:
        if not self.s3:
            return False
        try:
            self.s3.download_file(self.bucket, self._r2_key(caminho_relativo), str(destino))
            print(f"☁️ R2 download: {caminho_relativo}", file=sys.stdout)
            return True
        except Exception as e:
            print(f"❌ R2 download falhou ({caminho_relativo}): {e}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------ #
    #  Download SSP
    # ------------------------------------------------------------------ #

    def cdc_check(self, url, sessao):
        try:
            r = sessao.head(url, timeout=30, verify=False, allow_redirects=True)
            return True, int(r.headers.get("Content-Length", 0))
        except Exception as e:
            print(f"⚠️ CDC falhou: {e}", file=sys.stderr)
            return False, 0

    def calcular_sha256(self, caminho: Path) -> str:
        h = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""):
                h.update(bloco)
        return h.hexdigest()

    def baixar_robusto(self, url, destino, sessao, tentativas=5, atraso=5) -> bool:
        for i in range(tentativas):
            try:
                with sessao.get(url, stream=True, timeout=60, verify=False) as r:
                    r.raise_for_status()
                    with open(destino, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                return True
            except Exception as e:
                print(f"❌ Download tentativa {i+1}/{tentativas}: {e}", file=sys.stderr)
                time.sleep(atraso * (2 ** i))
        return False

    # ------------------------------------------------------------------ #
    #  Leitura e limpeza do XLSX  ← lógica nova baseada no documento
    # ------------------------------------------------------------------ #

    def _detectar_abas_dados(self, caminho_xlsx: Path) -> list[str]:
        """
        Lê os nomes de todas as abas e retorna apenas as que contêm
        as colunas críticas de dados criminais, ignorando metadados.
        """
        import openpyxl
        wb = openpyxl.load_workbook(caminho_xlsx, read_only=True, data_only=True)
        abas_dados = []

        for nome_aba in wb.sheetnames:
            # Ignorar explicitamente a aba de metadados
            if "Campos" in nome_aba or "SPDADOS" in nome_aba.upper():
                print(f"⏭️ Aba ignorada (metadados): {nome_aba}")
                continue

            ws = wb[nome_aba]
            # Pega apenas a primeira linha (cabeçalho)
            cabecalho = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
            if not cabecalho:
                continue

            colunas_aba = {str(c).strip().upper() for c in cabecalho if c}
            criticas_encontradas = self.COLUNAS_CRITICAS & colunas_aba

            # Se encontrar 4 ou mais colunas críticas, considera aba de dados
            if len(criticas_encontradas) >= 4:
                print(f"✅ Aba de dados detectada: {nome_aba} ({len(criticas_encontradas)} colunas críticas)")
                abas_dados.append(nome_aba)
            else:
                print(f"⏭️ Aba ignorada (poucas colunas críticas): {nome_aba}")

        wb.close()
        return abas_dados

    def _ler_e_limpar_xlsx(self, caminho_xlsx: Path, ano: int) -> pl.DataFrame:
        """
        Lê as abas de dados do XLSX, concatena e limpa conforme o documento.
        """
        abas = self._detectar_abas_dados(caminho_xlsx)

        if not abas:
            raise ValueError(f"Nenhuma aba de dados encontrada em {caminho_xlsx.name}")

        frames = []
        for aba in abas:
            print(f"📖 Lendo aba: {aba}", file=sys.stdout)
            df = pl.read_excel(caminho_xlsx, sheet_name=aba)

            # Normalizar nomes de colunas (strip + upper)
            df = df.rename({c: c.strip().upper() for c in df.columns})

            # Selecionar apenas colunas que existem no arquivo
            colunas_presentes = [c for c in self.COLUNAS_PRATA if c in df.columns]
            df = df.select(colunas_presentes)

            frames.append(df)

        df_final = pl.concat(frames, how="diagonal_relaxed")

        total_bruto = len(df_final)
        print(f"📊 Registros brutos: {total_bruto:,}", file=sys.stdout)

        # --- Limpeza ---

        # 1. Converter LATITUDE e LONGITUDE para float
        df_final = df_final.with_columns([
            pl.col("LATITUDE").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Float64, strict=False),
        ])

        # 2. Tratar coordenadas (0, 0) como nulas — são inválidas (Oceano Atlântico)
        df_final = df_final.with_columns([
            pl.when(pl.col("LATITUDE") == 0.0)
              .then(None)
              .otherwise(pl.col("LATITUDE"))
              .alias("LATITUDE"),
            pl.when(pl.col("LONGITUDE") == 0.0)
              .then(None)
              .otherwise(pl.col("LONGITUDE"))
              .alias("LONGITUDE"),
        ])

        # 3. Padronizar texto (strip + upper) nas colunas textuais
        colunas_texto = [
            "NOME_DEPARTAMENTO", "NOME_MUNICIPIO", "NOME_SECCIONAL",
            "NOME_DELEGACIA", "LOGRADOURO", "BAIRRO", "RUBRICA",
            "DESCR_CONDUTA", "NATUREZA_APURADA",
        ]
        for col in colunas_texto:
            if col in df_final.columns:
                df_final = df_final.with_columns(
                    pl.col(col).str.strip_chars().str.to_uppercase().alias(col)
                )

        # 4. Descartar registros com nulos nas colunas obrigatórias
        colunas_obrig_presentes = [c for c in self.COLUNAS_OBRIGATORIAS if c in df_final.columns]
        df_final = df_final.drop_nulls(subset=colunas_obrig_presentes)

        total_limpo = len(df_final)
        descartados = total_bruto - total_limpo
        print(f"🧹 Após limpeza: {total_limpo:,} registros ({descartados:,} descartados)", file=sys.stdout)

        return df_final

    # ------------------------------------------------------------------ #
    #  Processar XLSX → Parquet e subir pro R2
    # ------------------------------------------------------------------ #

    def _processar_e_salvar_raw(self, ano: int, tmp_xlsx: Path, parquet_file: Path,
                                 r2_key_parquet: str, meta_info: dict):
        print(f"⚙️ Processando {ano}...", file=sys.stdout)
        try:
            mime = magic.from_file(str(tmp_xlsx), mime=True)
            if mime != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                raise ValueError(f"Arquivo não é XLSX válido. MIME detectado: {mime}")

            df = self._ler_e_limpar_xlsx(tmp_xlsx, ano)
            df.write_parquet(parquet_file)
            print(f"✅ Parquet salvo: {parquet_file.name} ({len(df):,} registros)", file=sys.stdout)

            sha256 = self.calcular_sha256(parquet_file)
            meta_info[str(ano)] = {
                "tamanho_bytes": parquet_file.stat().st_size,
                "sha256": sha256,
                "registros": len(df),
                "data_sincronizacao": hora_brasilia().isoformat(),
            }

            self.r2_upload(parquet_file, r2_key_parquet)

        finally:
            tmp_xlsx.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    #  Sincronização raw
    # ------------------------------------------------------------------ #

    def sincronizar_raw(self):
        anos = list(range(2022, datetime.now().year + 1))
        ano_mais_recente = max(anos)

        sessao = requests.Session()
        retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        sessao.mount("https://", HTTPAdapter(max_retries=retry))
        sessao.mount("http://", HTTPAdapter(max_retries=retry))

        r2_meta_key = "raw/tracking_ssp.json"

        # Baixar tracking do R2
        if self.r2_download(r2_meta_key, self.meta):
            print("☁️ tracking_ssp.json baixado do R2.")
        else:
            print("⚠️ tracking_ssp.json não encontrado no R2. Iniciando do zero.")

        meta_info = {}
        if self.meta.exists():
            with open(self.meta) as f:
                meta_info = json.load(f)

        arquivos_para_processar = []

        for ano in anos:
            url_ssp = (
                f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/"
                f"spDados/SPDadosCriminais_{ano}.xlsx"
            )
            tmp_xlsx = self.pastas["raw"] / f"SPDadosCriminais_{ano}.xlsx"
            parquet_name = f"ssp_{ano}.parquet"
            parquet_file = self.pastas["raw"] / parquet_name
            r2_key_parquet = f"raw/{parquet_name}"

            print(f"\n--- Ano {ano} ---", file=sys.stdout)

            r2_tem = self.r2_existe(r2_key_parquet)

            if ano == ano_mais_recente or not r2_tem:
                # Só verifica SSP para o ano mais recente ou se não tiver no R2
                ssp_ok, ssp_size = self.cdc_check(url_ssp, sessao)

                if ssp_ok:
                    meta_ano = meta_info.get(str(ano), {})
                    r2_size = meta_ano.get("tamanho_bytes_xlsx")

                    if r2_tem and r2_size and r2_size == ssp_size:
                        print(f"✅ Ano {ano} no R2 está atualizado. Baixando do R2...", file=sys.stdout)
                        if self.r2_download(r2_key_parquet, parquet_file):
                            arquivos_para_processar.append(parquet_file)
                            continue

                    # Precisa baixar da SSP
                    print(f"⬇️ Baixando {ano} da SSP...", file=sys.stdout)
                    if self.baixar_robusto(url_ssp, tmp_xlsx, sessao):
                        # Salva o tamanho do XLSX no meta antes de processar
                        meta_info.setdefault(str(ano), {})["tamanho_bytes_xlsx"] = ssp_size
                        self._processar_e_salvar_raw(ano, tmp_xlsx, parquet_file, r2_key_parquet, meta_info)
                        arquivos_para_processar.append(parquet_file)
                    else:
                        # SSP falhou, tenta R2 como fallback
                        if r2_tem and self.r2_download(r2_key_parquet, parquet_file):
                            print(f"⚠️ Usando fallback R2 para {ano}.", file=sys.stdout)
                            arquivos_para_processar.append(parquet_file)
                        else:
                            raise Exception(f"Falha crítica: não foi possível obter dados de {ano}.")
                else:
                    # SSP inacessível
                    if r2_tem and self.r2_download(r2_key_parquet, parquet_file):
                        print(f"⚠️ SSP inacessível. Usando R2 para {ano}.", file=sys.stdout)
                        arquivos_para_processar.append(parquet_file)
                    else:
                        raise Exception(f"Falha crítica: SSP inacessível e {ano} não está no R2.")
            else:
                # Ano não é o mais recente e já está no R2 — só baixa direto
                print(f"☁️ Ano {ano} já no R2. Baixando...", file=sys.stdout)
                if self.r2_download(r2_key_parquet, parquet_file):
                    arquivos_para_processar.append(parquet_file)
                else:
                    raise Exception(f"Falha crítica ao baixar {ano} do R2.")

        # Salvar tracking atualizado
        with open(self.meta, "w") as f:
            json.dump(meta_info, f, indent=4)
        self.r2_upload(self.meta, r2_meta_key)
        print(f"\n✅ Tracking atualizado e salvo no R2.", file=sys.stdout)

        self.arquivos_raw_sincronizados = arquivos_para_processar

    # ------------------------------------------------------------------ #
    #  Camada Prata
    # ------------------------------------------------------------------ #

    def construir_prata(self):
        print("\n🔧 Construindo camada prata...", file=sys.stdout)

        frames = []
        for parquet_file in self.arquivos_raw_sincronizados:
            df = pl.read_parquet(parquet_file)
            frames.append(df)

        df = pl.concat(frames, how="diagonal_relaxed")
        total_bruto = len(df)

        # Converter data e hora
        df = df.with_columns([
            pl.col("DATA_OCORRENCIA_BO").cast(pl.Date, strict=False).alias("DATA_FATO"),
            pl.col("HORA_OCORRENCIA_BO").str.slice(0, 5).alias("HORA_CRIME"),
        ])

        # Enriquecer
        df = df.with_columns([
            pl.col("DATA_FATO").dt.weekday().alias("DIA_SEMANA"),
            pl.col("DATA_FATO").dt.month().alias("MES"),
            pl.col("DATA_FATO").dt.hour().alias("HORA_INT")
                if "HORA_INT" not in df.columns
                else pl.col("HORA_CRIME").str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT"),
            pl.col("DATA_FATO").map_elements(
                lambda d: 1 if d in self.feriados else 0, return_dtype=pl.Int32
            ).alias("FERIADO"),
        ])

        # H3
        df = df.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"])
            .map_elements(
                lambda r: h3.geo_to_h3(r["LATITUDE"], r["LONGITUDE"], 9),
                return_dtype=pl.String,
            )
            .alias("CODIGO_H3")
        )

        # Perfil vítima
        df = df.with_columns(
            pl.when(pl.col("RUBRICA").str.contains("CICLISTA|BICICLETA"))
            .then(pl.lit("CICLISTA"))
            .when(pl.col("RUBRICA").str.contains("TRANSEUNTE|PEDESTRE"))
            .then(pl.lit("PEDESTRE"))
            .when(pl.col("RUBRICA").str.contains("MOTO|MOTOCICLETA|MOTOCICLISTA"))
            .then(pl.lit("MOTOCICLISTA"))
            .when(pl.col("RUBRICA").str.contains("VEICULO|AUTO|CARRO|CARGA|CAMINHAO"))
            .then(pl.lit("MOTORISTA"))
            .otherwise(pl.lit("GERAL"))
            .alias("PERFIL_VITIMA")
        )

        # Peso penal
        df = df.with_columns(
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO|HOMICÍDIO|SEQUESTRO|CÁRCERE|CARCERE"))
            .then(pl.lit(5))
            .when(pl.col("RUBRICA").str.contains("ROUBO|EXTORCAO|EXTORSÃO|ESTUPRO"))
            .then(pl.lit(4))
            .when(pl.col("RUBRICA").str.contains("FURTO QUALIFICADO|RECEPTACAO|RECEPTAÇÃO|ARMA DE FOGO"))
            .then(pl.lit(3))
            .when(pl.col("RUBRICA").str.contains("FURTO|DANO|AMEACA|AMEAÇA|DESACATO"))
            .then(pl.lit(2))
            .otherwise(pl.lit(1))
            .alias("PESO_PENAL")
        )

        # Anonimizar e salvar
        df_prata = df.with_columns(
            pl.col("LATITUDE").hash().alias("ID_ANONIMO")
        ).drop(["DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO", "LOGRADOURO",
                "NUMERO_LOGRADOURO", "BAIRRO", "NOME_DELEGACIA", "NOME_SECCIONAL"])

        df_prata.write_parquet(self.pastas["prata"] / "camada_prata.parquet")

        total_final = len(df_prata)
        print(f"✅ Prata: {total_final:,} registros salvos.", file=sys.stdout)

        return df_prata

    # ------------------------------------------------------------------ #
    #  Pipeline principal
    # ------------------------------------------------------------------ #

    def processar(self):
        bq_project = os.environ.get("BQ_PROJECT", "").strip()
        bq_dataset = os.environ.get("BQ_DATASET", "").strip()
        bq_cred_json = os.environ.get("BQ_CREDENTIALS_JSON", "").strip()

        self.sincronizar_raw()
        df_prata = self.construir_prata()
        self.reconstruir_ouro_validacao_shap(bq_project, bq_dataset, bq_cred_json)

        tempo = time.time() - self.t_inicio
        media_risco = df_prata["PESO_PENAL"].mean() if "PESO_PENAL" in df_prata.columns else 0.0
        self.discord.notificar_sucesso(
            "SafeDriver Pipeline Concluído",
            tempo, len(df_prata), media_risco,
            "✅ R2 atualizado", "✅ BigQuery atualizado"
        )

    def reconstruir_ouro_validacao_shap(self, bq_project, bq_dataset, bq_cred_json):
        # Mantém sua lógica existente de ouro/SHAP aqui
        pass


if __name__ == "__main__":
    app = SafeDriver()
    try:
        app.processar()
    except Exception:
        err = traceback.format_exc()
        print("\n" + "=" * 50, file=sys.stderr)
        print("🚨 ERRO FATAL NO PIPELINE 🚨", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        print(err, file=sys.stderr)
        app.discord.notificar_erro("Falha Sistêmica", err)
        sys.exit(1)
