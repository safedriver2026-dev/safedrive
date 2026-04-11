import sys, os, requests, traceback, hashlib, warnings, time, json
from pathlib import Path
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import polars as pl
import pandas as pd
import numpy as np
import h3
import holidays
import boto3
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from google.cloud import bigquery
from google.oauth2 import service_account
import magic # Importar biblioteca para detecção de tipo de arquivo

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

def hora_brasilia() -> datetime:
    return datetime.utcnow() - timedelta(hours=3)


def criar_cliente_bq(projeto: str, cred_json: str) -> bigquery.Client:
    info = json.loads(cred_json)
    credentials = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(project=projeto, credentials=credentials)


def enviar_para_bigquery(
    df_pl: pl.DataFrame, tabela: str, projeto: str, dataset: str, cred_json: str
):
    df_pd = df_pl.to_pandas()
    client = criar_cliente_bq(projeto, cred_json)
    tabela_id = f"{projeto}.{dataset}.{tabela}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df_pd, tabela_id, job_config=job_config)
    job.result()


def tabela_existe_bigquery(projeto: str, dataset: str, tabela: str, cred_json: str) -> bool:
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
            print("⚠️ AVISO: URL do Webhook ausente.", file=sys.stderr)
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"❌ FALHA CONEXÃO DISCORD: {e}", file=sys.stderr)

    def notificar_sucesso(
        self, titulo, tempo_execucao, registros, media_risco, status_s3, status_bq
    ):
        payload = {
            "embeds": [
                {
                    "title": f"🟢 {titulo}",
                    "description": (
                        "**Relatório Executivo SafeDriver**\n"
                        "O motor preditivo sincronizou os dados da SSP e atualizou o modelo com sucesso."
                    ),
                    "color": 3066993,
                    "fields": [
                        {
                            "name": "📊 Volumetria (Camada Prata)",
                            "value": f"{registros:,} ocorrências",
                            "inline": True,
                        },
                        {
                            "name": "⚠️ Risco Médio (volume)",
                            "value": f"{media_risco:.2f} pontos",
                            "inline": True,
                        },
                        {
                            "name": "⏱️ Tempo de Processamento",
                            "value": f"{tempo_execucao:.1f} segundos",
                            "inline": True,
                        },
                        {
                            "name": "☁️ Backup Cloudflare R2",
                            "value": status_s3,
                            "inline": False,
                        },
                        {
                            "name": "📡 Publicação BigQuery",
                            "value": status_bq,
                            "inline": False,
                        },
                    ],
                    "footer": {
                        "text": f"SafeDriver AI • Data/Hora: {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"
                    },
                }
            ]
        }
        self._enviar_webhook(self.sucesso, payload)

    def notificar_info(self, titulo, corpo):
        payload = {
            "embeds": [
                {
                    "title": f"🔵 {titulo}",
                    "description": corpo,
                    "color": 3447003,
                    "footer": {
                        "text": f"SafeDriver AI • Data/Hora: {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"
                    },
                }
            ]
        }
        self._enviar_webhook(self.sucesso, payload)

    def notificar_erro(self, titulo, erro_msg):
        ticks = chr(96) * 3
        stack = f"{ticks}python\n{erro_msg[:1000]}\n{ticks}"
        payload = {
            "embeds": [
                {
                    "title": f"🔴 {titulo}",
                    "description": "**Falha Crítica no Pipeline**",
                    "color": 15158332,
                    "fields": [
                        {"name": "Detalhes Técnicos", "value": stack, "inline": False}
                    ],
                    "footer": {
                        "text": f"SafeDriver AI Alerts • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"
                    },
                }
            ]
        }
        self._enviar_webhook(self.erro, payload)


class SafeDriver:
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

        self.feriados = list(
            holidays.Brazil(subdiv="SP", years=range(2022, 2027)).keys()
        )
        # O nome do arquivo meta.json no R2 é tracking_ssp.json
        self.meta = self.pastas["raw"] / "tracking_ssp.json" 

    def cdc_check(self, ano, url, sessao):
        try:
            r = sessao.head(url, timeout=30, verify=False, allow_redirects=True)
            size = int(r.headers.get("Content-Length", 0))
            return True, size
        except requests.exceptions.RequestException as e:
            print(f"⚠️ CDC: Falha ao verificar SSP para {ano}: {e}", file=sys.stderr)
            return False, 0
        except Exception as e:
            print(f"❌ CDC: Erro inesperado ao verificar SSP para {ano}: {e}", file=sys.stderr)
            return False, 0

    def calcular_sha256(self, caminho_arquivo):
        sha256_hash = hashlib.sha256()
        with open(caminho_arquivo, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def baixar_robusto(self, url, destino, sessao, max_tentativas=5, atraso_base=5):
        for tentativa in range(max_tentativas):
            try:
                with sessao.get(url, stream=True, timeout=30, verify=False) as r:
                    r.raise_for_status()
                    with open(destino, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                return True
            except requests.exceptions.RequestException as e:
                print(f"❌ Erro ao baixar {url} (tentativa {tentativa + 1}/{max_tentativas}): {e}", file=sys.stderr)
                time.sleep(atraso_base * (2**tentativa))
        return False

    def upload_para_r2(self, arquivo_local: Path, chave_r2: str):
        if not self.s3:
            print("⚠️ R2: Credenciais S3 não configuradas. Pulando upload.", file=sys.stderr)
            return "PULADO"
        try:
            self.s3.upload_file(str(arquivo_local), self.bucket, chave_r2)
            print(f"☁️ R2: Upload de {arquivo_local.name} para R2 como {chave_r2} concluído.", file=sys.stdout)
            return "SUCESSO"
        except Exception as e:
            print(f"❌ R2: Falha no upload de {arquivo_local.name} para R2: {e}", file=sys.stderr)
            return "FALHA"

    def download_do_r2(self, chave_r2: str, arquivo_local: Path):
        if not self.s3:
            return False
        try:
            self.s3.download_file(self.bucket, chave_r2, str(arquivo_local))
            print(f"☁️ R2: Download de {chave_r2} do R2 para {arquivo_local.name} concluído.", file=sys.stdout)
            return True
        except Exception as e:
            print(f"❌ R2: Falha no download de {chave_r2} do R2: {e}", file=sys.stderr)
            return False

    def r2_object_exists(self, chave_r2: str) -> bool:
        if not self.s3:
            return False
        try:
            self.s3.head_object(Bucket=self.bucket, Key=chave_r2)
            return True
        except Exception:
            return False

    def sincronizar_raw(self):
        anos = range(2022, datetime.now().year + 1)
        sessao = requests.Session()
        retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        sessao.mount("https://", adapter)
        sessao.mount("http://", adapter)

        # R2 Key para o arquivo de meta
        r2_meta_key = "raw/tracking_ssp.json"

        # Tenta baixar o meta.json do R2 primeiro
        if self.download_do_r2(r2_meta_key, self.meta):
            print("☁️ R2: tracking_ssp.json baixado do R2.", file=sys.stdout)
        else:
            print("⚠️ R2: tracking_ssp.json não encontrado no R2 ou falha no download. Criando novo ou usando local.", file=sys.stdout)

        # Carrega meta.json, ou inicializa se não existir
        if self.meta.exists():
            with open(self.meta, "r") as f:
                meta_info = json.load(f)
        else:
            meta_info = {}

        arquivos_para_processar = []

        for ano in anos:
            url_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            tmp_xlsx = self.pastas["raw"] / f"SPDadosCriminais_{ano}.xlsx"
            # CORREÇÃO AQUI: Usar o nome do arquivo como está no R2
            parquet_file_name = f"ssp_{ano}.parquet" 
            parquet_file = self.pastas["raw"] / parquet_file_name
            r2_key = f"raw/{parquet_file_name}" # E a chave do R2 também

            print(f"\n--- Sincronizando ano {ano} ---", file=sys.stdout)

            # 1. Verificar se o arquivo .parquet já existe no R2
            r2_tem_arquivo = self.r2_object_exists(r2_key)

            # 2. Lógica para o ano mais recente (ou se o arquivo não está no R2)
            if ano == max(anos) or not r2_tem_arquivo:
                print(f"🔄 Verificando SSP para o ano {ano} (último ano ou não no R2)...", file=sys.stdout)
                ssp_acessivel, ssp_size = self.cdc_check(ano, url_ssp, sessao)

                if ssp_acessivel:
                    meta_ano = meta_info.get(str(ano), {})

                    # Se o arquivo local (ou no R2) é diferente da SSP, ou não temos meta, ou não está no R2
                    if (
                        not meta_ano
                        or meta_ano.get("tamanho_bytes") != ssp_size
                        or not r2_tem_arquivo # Se não está no R2, consideramos que precisa ser baixado e processado
                    ):
                        print(f"⬇️ Baixando {ano} da SSP (arquivo novo/atualizado)...", file=sys.stdout)
                        if not self.baixar_robusto(url_ssp, tmp_xlsx, sessao):
                            print(f"❌ Falha ao baixar {ano} da SSP. Pulando este ano.", file=sys.stderr)
                            self.discord.notificar_erro(f"SafeDriver Sync {ano}", f"Falha ao baixar {ano} da SSP.")
                            continue

                        file_type = magic.from_file(str(tmp_xlsx), mime=True)
                        if file_type != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                            print(f"❌ Erro Crítico ao processar {ano}: Arquivo baixado não é um XLSX válido ({file_type}). Removendo e pulando.", file=sys.stderr)
                            tmp_xlsx.unlink(missing_ok=True)
                            self.discord.notificar_erro(f"SafeDriver Sync {ano}", f"Arquivo {ano} da SSP não é XLSX válido.")
                            continue

                        print(f"⚙️ Processando {ano}.xlsx para {parquet_file_name}...", file=sys.stdout)
                        try:
                            df_excel = pl.read_excel(str(tmp_xlsx), sheet_name=None)
                            aba_valida = None
                            for sheet_name, df_sheet in df_excel.items():
                                if len(df_sheet.columns) >= 5:
                                    aba_valida = df_sheet
                                    break

                            if aba_valida is None:
                                raise ValueError("Nenhuma aba válida encontrada com 5 ou mais colunas.")

                            aba_valida.write_parquet(parquet_file)
                            tmp_xlsx.unlink(missing_ok=True)

                            # Atualiza meta.json
                            meta_info[str(ano)] = {
                                "tamanho_bytes": ssp_size,
                                "sha256": self.calcular_sha256(str(parquet_file)),
                                "data_sincronizacao": hora_brasilia().isoformat(),
                            }

                            self.upload_para_r2(parquet_file, r2_key)
                            arquivos_para_processar.append(parquet_file)

                        except Exception as e:
                            print(f"❌ Erro ao processar {ano}.xlsx: {e}. Pulando este ano.", file=sys.stderr)
                            self.discord.notificar_erro(f"SafeDriver Sync {ano}", f"Erro ao processar XLSX: {e}")
                            tmp_xlsx.unlink(missing_ok=True)
                            continue
                    else:
                        print(f"✅ Arquivo {ano} na SSP inalterado e já no R2/local. Baixando do R2 para processamento local...", file=sys.stdout)
                        if self.download_do_r2(r2_key, parquet_file):
                            arquivos_para_processar.append(parquet_file)
                        else:
                            print(f"❌ F..file):
                    arquivos_para_processar.append(parquet_file)
                else:
                    print(f"❌ Falha ao baixar {ano}.parquet do R2 para processamento local. Pulando.", file=sys.stderr)
                    self.discord.notificar_erro(f"SafeDriver Sync {ano}", f"Falha ao baixar {ano}.parquet do R2 para local.")
                    continue

        # Salva o meta.json atualizado no R2
        with open(self.meta, "w") as f:
            json.dump(meta_info, f, indent=4)
        self.upload_para_r2(self.meta, r2_meta_key)
        print("☁️ R2: tracking_ssp.json atualizado e enviado para R2.", file=sys.stdout)

        self.arquivos_raw_sincronizados = arquivos_para_processar


    def publicar_bigquery_a_partir_de_arquivos(self, bq_project, bq_dataset, bq_cred_json):
        df_dashboard = pl.read_parquet(self.pastas["ouro"] / "dashboard_final.parquet")
        enviar_para_bigquery(df_dashboard, "sd_dashboard_final", bq_project, bq_dataset, bq_cred_json)
        print("📊 Publicado sd_dashboard_final no BigQuery.", file=sys.stdout)

        df_validacao = pl.read_parquet(self.pastas["ouro"] / "validacao_modelo.parquet")
        enviar_para_bigquery(df_validacao, "sd_validacao_modelo", bq_project, bq_dataset, bq_cred_json)
        print("✅ Publicado sd_validacao_modelo no BigQuery.", file=sys.stdout)

        df_shap = pl.read_parquet(self.pastas["ouro"] / "shap_audit.parquet")
        enviar_para_bigquery(df_shap, "sd_shap_audit", bq_project, bq_dataset, bq_cred_json)
        print("🔍 Publicado sd_shap_audit no BigQuery.", file=sys.stdout)

    def reconstruir_ouro_validacao_shap(self, bq_project, bq_dataset, bq_cred_json):
        print("✨ Construindo Camada Ouro e Modelos...", file=sys.stdout)

        df_prata = pl.read_parquet(self.pastas["prata"] / "camada_prata.parquet")

        crimes_por_h3_ano = (
            df_prata.group_by(["CODIGO_H3", "ANO"])
            .agg(
                pl.count().alias("CRIMES_POR_ANO"),
                pl.col("PESO_PENAL").sum().alias("CRIMES_POND_POR_ANO"),
                pl.col("LAT").mean().alias("LATITUDE_MEDIA"),
                pl.col("LON").mean().alias("LONGITUDE_MEDIA"),
                pl.col("LAT").std().alias("LAT_STD"),
                pl.col("LON").std().alias("LON_STD"),
                (pl.col("TIPO_CRIME") == "PATRIMONIO").sum().alias("PATRIMONIO_COUNT"),
                (pl.col("PERFIL_VITIMA") == "MOTORISTA").sum().alias("MOTORISTA_COUNT"),
                (pl.col("PERFIL_VITIMA") == "MOTOCICLISTA").sum().alias("MOTO_COUNT"),
                (pl.col("PERFIL_VITIMA") == "PEDESTRE").sum().alias("PEDESTRE_COUNT"),
                (pl.col("PERFIL_VITIMA") == "CICLISTA").sum().alias("CICLISTA_COUNT"),
                (pl.col("EH_FERIADO") == 1).sum().alias("FERIADO_COUNT"),
                (pl.col("SEMANA_PAGAMENTO") == 1).sum().alias("PAGAMENTO_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "MANHA").sum().alias("MANHA_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "TARDE").sum().alias("TARDE_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "NOITE").sum().alias("NOITE_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "MADRUGADA").sum().alias("MADRUGADA_COUNT"),
                (
                    (pl.col("PERIODO_DETALHADO") == "NOITE")
                    & (pl.col("TIPO_CRIME") == "PATRIMONIO")
                ).sum().alias("RISCO_NOITE_PATRIMONIO"),
                (
                    (pl.col("PERFIL_VITIMA") == "MOTOCICLISTA")
                    & (pl.col("PERIODO_DETALHADO") == "NOITE")
                ).sum().alias("RISCO_MOTO_NOITE"),
                (
                    (pl.col("PERFIL_VITIMA") == "MOTORISTA")
                    & (pl.col("SEMANA_PAGAMENTO") == 1)
                ).sum().alias("RISCO_MOTORISTA_PAGTO"),
            )
            .sort(["CODIGO_H3", "ANO"])
        )

        features_h3 = (
            crimes_por_h3_ano.group_by("CODIGO_H3")
            .agg(
                pl.col("CRIMES_POR_ANO").sum().alias("CRIMES_REAIS"),
                pl.col("CRIMES_POND_POR_ANO").sum().alias("CRIMES_POND_TOTAIS"),
                pl.col("LATITUDE_MEDIA").mean().alias("LATITUDE_MEDIA"),
                pl.col("LONGITUDE_MEDIA").mean().alias("LONGITUDE_MEDIA"),
                pl.col("LAT_STD").mean().alias("LAT_STD"),
                pl.col("LON_STD").mean().alias("LON_STD"),
                (pl.col("PATRIMONIO_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_PATRIMONIO"),
                (pl.col("MOTORISTA_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_MOTORISTA"),
                (pl.col("MOTO_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_MOTO"),
                (pl.col("PEDESTRE_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_PEDESTRE"),
                (pl.col("CICLISTA_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_CICLISTA"),
                (pl.col("FERIADO_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_FERIADO"),
                (pl.col("PAGAMENTO_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_PAGAMENTO"),
                (pl.col("MANHA_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_MANHA"),
                (pl.col("TARDE_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_TARDE"),
                (pl.col("NOITE_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_NOITE"),
                (pl.col("MADRUGADA_COUNT").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("PROP_MADRUGADA"),
                (pl.col("RISCO_NOITE_PATRIMONIO").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("RISCO_NOITE_PATRIMONIO"),
                (pl.col("RISCO_MOTO_NOITE").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("RISCO_MOTO_NOITE"),
                (pl.col("RISCO_MOTORISTA_PAGTO").sum() / pl.col("CRIMES_POR_ANO").sum()).alias("RISCO_MOTORISTA_PAGTO"),
                pl.col("CRIMES_POR_ANO").count().alias("ANOS_COM_CRIMES"),
            )
            .with_columns(
                (pl.col("CRIMES_REAIS") / pl.col("ANOS_COM_CRIMES")).alias("CRIMES_POR_ANO"),
                (pl.col("CRIMES_POND_TOTAIS") / pl.col("ANOS_COM_CRIMES")).alias("CRIMES_POND_POR_ANO"),
            )
            .drop("ANOS_COM_CRIMES")
            .with_columns(
                pl.col("CODIGO_H3")
                .map_elements(self.wkt, return_dtype=pl.String)
                .alias("GEOMETRIA_WKT")
            )
        )

        features_modelo = [
            "LATITUDE_MEDIA",
            "LONGITUDE_MEDIA",
            "PROP_PATRIMONIO",
            "PROP_MOTORISTA",
            "PROP_MOTO",
            "PROP_PEDESTRE",
            "PROP_CICLISTA",
            "PROP_FERIADO",
            "PROP_PAGAMENTO",
            "PROP_MANHA",
            "PROP_TARDE",
            "PROP_NOITE",
            "PROP_MADRUGADA",
            "RISCO_NOITE_PATRIMONIO",
            "RISCO_MOTO_NOITE",
            "RISCO_MOTORISTA_PAGTO",
            "LAT_STD",
            "LON_STD",
            "CRIMES_POR_ANO",
            "CRIMES_POND_POR_ANO",
        ]
        X = features_h3.select(features_modelo).to_pandas()
        y_volume = features_h3.select("CRIMES_REAIS").to_pandas().squeeze()
        y_penal = features_h3.select("CRIMES_POND_POR_ANO").to_pandas().squeeze()

        model_volume = VotingRegressor(
            estimators=[
                ("cat", CatBoostRegressor(random_state=42, verbose=0)),
                ("lgbm", LGBMRegressor(random_state=42)),
            ]
        )
        model_volume.fit(X, y_volume)
        features_h3 = features_h3.with_columns(
            pl.Series(
                "ESCORE_RISCO_VOLUME", model_volume.predict(X), dtype=pl.Float64
            ).alias("ESCORE_RISCO")
        )

        model_penal = VotingRegressor(
            estimators=[
                ("cat", CatBoostRegressor(random_state=42, verbose=0)),
                ("lgbm", LGBMRegressor(random_state=42)),
            ]
        )
        model_penal.fit(X, y_penal)
        features_h3 = features_h3.with_columns(
            pl.Series(
                "ESCORE_RISCO_PENAL", model_penal.predict(X), dtype=pl.Float64
            ).alias("ESCORE_RISCO_PENAL")
        )

        dashboard_final = features_h3.select(
            [
                "CODIGO_H3",
                "GEOMETRIA_WKT",
                "LATITUDE_MEDIA",
                "LONGITUDE_MEDIA",
                "CRIMES_REAIS",
                "ESCORE_RISCO",
                "ESCORE_RISCO_PENAL",
                "PROP_PATRIMONIO",
                "PROP_MOTORISTA",
                "PROP_MOTO",
                "PROP_PEDESTRE",
                "PROP_CICLISTA",
                "PROP_FERIADO",
                "PROP_PAGAMENTO",
                "PROP_MANHA",
                "PROP_TARDE",
                "PROP_NOITE",
                "PROP_MADRUGADA",
                "RISCO_NOITE_PATRIMONIO",
                "RISCO_MOTO_NOITE",
                "RISCO_MOTORISTA_PAGTO",
                "LAT_STD",
                "LON_STD",
                "CRIMES_POR_ANO",
                "CRIMES_POND_POR_ANO",
            ]
        )
        dashboard_final.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")

        validacao_modelo = features_h3.select(
            ["CODIGO_H3", "CRIMES_REAIS", "ESCORE_RISCO"]
        ).with_columns(
            (pl.col("CRIMES_REAIS") - pl.col("ESCORE_RISCO")).abs().alias("ERRO_ABS")
        )
        validacao_modelo.write_parquet(self.pastas["ouro"] / "validacao_modelo.parquet")

        explainer = shap.TreeExplainer(model_volume)
        shap_values = explainer.shap_values(X)
        shap_sum = np.abs(shap_values).mean(axis=0)
        shap_importance = pl.DataFrame(
            {
                "VARIAVEL": X.columns,
                "GRAU_IMPORTANCIA": shap_sum,
            }
        ).sort("GRAU_IMPORTANCIA", descending=True)
        shap_importance.write_parquet(self.pastas["ouro"] / "shap_audit.parquet")

        self.publicar_bigquery_a_partir_de_arquivos(bq_project, bq_dataset, bq_cred_json)

    def processar(self):
        print("Iniciando Verificação de Integridade (Self-Healing)...", file=sys.stdout)
        self.sincronizar_raw()

        bq_project = os.environ.get("BQ_PROJECT_ID")
        bq_dataset = os.environ.get("BQ_DATASET_ID")
        bq_cred_json = os.environ.get("BQ_SERVICE_ACCOUNT_JSON")

        arquivos_limpos = self.arquivos_raw_sincronizados
        if not arquivos_limpos:
            msg = "Nenhum arquivo RAW válido encontrado após sincronização."
            self.discord.notificar_erro("SafeDriver Sync", msg)
            raise Exception(msg)

        print("⚙️ Construindo Camada Prata...", file=sys.stdout)
        lf = pl.concat([pl.scan_parquet(f) for f in arquivos_limpos], how="diagonal")

        prata = (
            lf.with_columns(
                [
                    pl.col("D")
                    .str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
                    .alias("DATA_FATO"),
                    pl.col("LAT").str.replace(",", ".").cast(pl.Float32, strict=False),
                    pl.col("LON").str.replace(",", ".").cast(pl.Float32, strict=False),
                    pl.col("H")
                    .str.slice(0, 2)
                    .cast(pl.Int8, strict=False)
                    .alias("HORA_CRIME"),
                ]
            )
            .filter(pl.col("LAT").is_between(-25.5, -19.5))
            .with_columns(
                [
                    pl.col("DATA_FATO").dt.year().alias("ANO"),
                    pl.col("DATA_FATO").dt.month().alias("MES"),
                    pl.when(
                        pl.col("N").str.to_uppercase().str.contains("ROUBO|FURTO")
                    )
                    .then(pl.lit("PATRIMONIO"))
                    .otherwise(pl.lit("PESSOA"))
                    .alias("TIPO_CRIME"),
                    pl.when(pl.col("HORA_CRIME").is_between(5, 11))
                    .then(pl.lit("MANHA"))
                    .when(pl.col("HORA_CRIME").is_between(12, 17))
                    .then(pl.lit("TARDE"))
                    .when(pl.col("HORA_CRIME").is_between(18, 23))
                    .then(pl.lit("NOITE"))
                    .otherwise(pl.lit("MADRUGADA"))
                    .alias("PERIODO_DETALHADO"),
                    pl.col("DATA_FATO")
                    .dt.date()
                    .is_in(self.feriados)
                    .cast(pl.Int8)
                    .alias("EH_FERIADO"),
                    (
                        (pl.col("DATA_FATO").dt.day().is_between(28, 31))
                        | (pl.col("DATA_FATO").dt.day().is_between(1, 7))
                    )
                    .cast(pl.Int8)
                    .alias("SEMANA_PAGAMENTO"),
                    pl.when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("CICLISTA|BICICLETA")
                    )
                    .then(pl.lit("CICLISTA"))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("TRANSEUNTE|PEDESTRE")
                    )
                    .then(pl.lit("PEDESTRE"))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("MOTO|MOTOCICLETA|MOTOCICLISTA")
                    )
                    .then(pl.lit("MOTOCICLISTA"))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("VEICULO|AUTO|CARRO|CARGA|CAMINHAO")
                    )
                    .then(pl.lit("MOTORISTA"))
                    .otherwise(pl.lit("GERAL"))
                    .alias("PERFIL_VITIMA"),
                    pl.when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("LATROCINIO|HOMICIDIO|HOMICÍDIO|SEQUESTRO|CÁRCERE|CARCERE")
                    )
                    .then(pl.lit(5))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("ROUBO|EXTORCAO|EXTORSÃO|ESTUPRO")
                    )
                    .then(pl.lit(4))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("FURTO QUALIFICADO|RECEPTACAO|RECEPTAÇÃO|ARMA DE FOGO")
                    )
                    .then(pl.lit(3))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("FURTO|DANO|AMEACA|AMEAÇA|DESACATO")
                    )
                    .then(pl.lit(2))
                    .otherwise(pl.lit(1))
                    .alias("PESO_PENAL"),
                ]
            )
            .with_columns(
                pl.struct(["LAT", "LON"])
                .map_elements(
                    lambda coords: h3.geo_to_h3(coords["LAT"], coords["LON"], 9),
                    return_dtype=pl.String,
                )
                .alias("CODIGO_H3")
            )
            .collect(streaming=True)
        )

        prata.with_columns(
            pl.col("LAT").hash().alias("ID_ANONIMO")
        ).drop(["DATA_FATO", "H", "HORA_CRIME", "N"]).write_parquet(
            self.pastas["prata"] / "camada_prata.parquet"
        )

        self.reconstruir_ouro_validacao_shap(bq_project, bq_dataset, bq_cred_json)


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
