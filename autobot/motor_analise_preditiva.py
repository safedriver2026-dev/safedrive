import sys, os, requests, traceback, hashlib, gc, warnings, re, time, json
from pathlib import Path
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3, polars as pl, pandas as pd, numpy as np
import h3, holidays, boto3, shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from google.cloud import bigquery
from google.oauth2 import service_account

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")


def hora_brasilia() -> datetime:
    """Retorna datetime no fuso horário de Brasília (UTC-3), baseado em UTC do servidor."""
    return datetime.utcnow() - timedelta(hours=3)


def enviar_para_bigquery(df_pl: pl.DataFrame, tabela: str, projeto: str, dataset: str, cred_json: str):
    """
    Envia um DataFrame Polars para uma tabela do BigQuery.
    Sobrescreve a tabela (WRITE_TRUNCATE) em cada execução.
    """
    df_pd = df_pl.to_pandas()
    info = json.loads(cred_json)
    credentials = service_account.Credentials.from_service_account_info(info)
    client = bigquery.Client(project=projeto, credentials=credentials)
    tabela_id = f"{projeto}.{dataset}.{tabela}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df_pd, tabela_id, job_config=job_config)
    job.result()


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

    def notificar_sucesso(self, titulo, tempo_execucao, registros, media_risco, status_s3, status_bq):
        payload = {
            "embeds": [{
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
                        "name": "⚠️ Risco Médio",
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
            }]
        }
        self._enviar_webhook(self.sucesso, payload)

    def notificar_info(self, titulo, corpo):
        payload = {
            "embeds": [{
                "title": f"🔵 {titulo}",
                "description": corpo,
                "color": 3447003,
                "footer": {
                    "text": f"SafeDriver AI • Data/Hora: {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"
                },
            }]
        }
        self._enviar_webhook(self.sucesso, payload)

    def notificar_erro(self, titulo, erro_msg):
        ticks = chr(96) * 3
        stack = f"{ticks}python\n{erro_msg[:1000]}\n{ticks}"
        payload = {
            "embeds": [{
                "title": f"🔴 {titulo}",
                "description": "**Falha Crítica no Pipeline**",
                "color": 15158332,
                "fields": [{"name": "Detalhes Técnicos", "value": stack, "inline": False}],
                "footer": {
                    "text": f"SafeDriver AI Alerts • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"
                },
            }]
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
            k: os.environ.get(k, "").strip()
            for k in ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
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

        self.feriados = list(holidays.Brazil(subdiv="SP", years=range(2022, 2027)).keys())
        self.meta = self.pastas["raw"] / "meta.json"

    def cdc_check(self, ano, url, sessao):
        try:
            r = sessao.head(url, timeout=30, verify=False)
            size = int(r.headers.get("Content-Length", 0))
            if self.meta.exists():
                with open(self.meta, "r") as f:
                    dados_meta = json.load(f).get(str(ano))
                    if isinstance(dados_meta, dict):
                        if dados_meta.get("tamanho_bytes") == size:
                            return False, size
                    elif dados_meta == size:
                        return False, size
            return True, size
        except:
            return True, 0

    def calcular_sha256(self, caminho_arquivo):
        sha256_hash = hashlib.sha256()
        try:
            with open(caminho_arquivo, "rb") as f:
                for byte_block in iter(lambda: f.read(4096 * 1024), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return None

    def baixar_robusto(self, url, caminho_tmp, sessao):
        max_tentativas = 10
        for tentativa in range(max_tentativas):
            try:
                tamanho_atual = os.path.getsize(caminho_tmp) if os.path.exists(caminho_tmp) else 0
                headers = {}
                if tamanho_atual > 0:
                    headers["Range"] = f"bytes={tamanho_atual}-"

                with sessao.get(url, headers=headers, stream=True, timeout=(30, 120), verify=False) as r:
                    if r.status_code == 416:
                        return True
                    if r.status_code not in (200, 206):
                        time.sleep(5)
                        continue

                    modo = "ab" if r.status_code == 206 else "wb"

                    with open(caminho_tmp, modo) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                return True
            except Exception:
                print(
                    f"⚠️ Conexão caiu (Tentativa {tentativa+1}/{max_tentativas}). "
                    "Tentando retomar o download...",
                    file=sys.stdout,
                )
                time.sleep(5)
        return False

    def wkt(self, h3_id):
        try:
            b = h3.h3_to_geo_boundary(h3_id, geo_json=True)
            pts = ", ".join([f"{ln} {lt}" for ln, lt in b])
            return f"POLYGON(({pts}, {b[0][0]} {b[0][1]}))"
        except:
            return None

    def processar(self):
        print("Iniciando Verificação de Integridade (Self-Healing)...", file=sys.stdout)

        # Self-healing da Bronze
        for f in self.pastas["raw"].glob("*.parquet"):
            try:
                if "D" not in pl.scan_parquet(f).columns:
                    f.unlink()
            except:
                try:
                    f.unlink()
                except:
                    pass

        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
        s.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "*/*",
            }
        )

        novo = False

        # ETL BRONZE (download SSP + cache)
        for ano in range(2022, datetime.now().year + 1):
            url = (
                "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/"
                f"SPDadosCriminais_{ano}.xlsx"
            )
            path = self.pastas["raw"] / f"ssp_{ano}.parquet"
            tmp = self.pastas["raw"] / f"tmp_{ano}.xlsx"

            run, sz = self.cdc_check(ano, url, s)
            if not run and path.exists():
                continue

            try:
                print(f"📥 Baixando SSP {ano} (Resiliente)...", file=sys.stdout)

                if os.path.exists(tmp):
                    os.remove(tmp)

                sucesso_download = self.baixar_robusto(url, str(tmp), s)

                if sucesso_download:
                    novo = True
                    print(f"✅ Download {ano} concluído! Inspecionando abas...", file=sys.stdout)
                    import fastexcel

                    ex = fastexcel.read_excel(str(tmp))
                    abas = []

                    mapping = {
                        "LAT": ["LATITUDE", "LAT", "Y"],
                        "LON": ["LONGITUDE", "LON", "X"],
                        "D": [
                            "DATA_OCORRENCIA",
                            "DATAOCORRENCIA",
                            "DATA_FATO",
                            "DATA_DO_FATO",
                            "DATA_REF",
                        ],
                        "H": [
                            "HORA_OCORRENCIA",
                            "HORAOCORRENCIA",
                            "HORA_FATO",
                            "HORA_DO_FATO",
                            "HORA_REF",
                        ],
                        "N": ["NATUREZA_APURADA", "RUBRICA"],
                    }

                    for n in ex.sheet_names:
                        df = pl.read_excel(str(tmp), sheet_name=n, engine="calamine")

                        if len(df.columns) > 5:
                            df.columns = [
                                str(c)
                                .upper()
                                .strip()
                                .replace("\n", "")
                                .replace("\r", "")
                                .replace(" ", "_")
                                for c in df.columns
                            ]

                            f_cols = {}
                            for target, aliases in mapping.items():
                                for alias in aliases:
                                    encontrou = False
                                    for col in df.columns:
                                        if alias in col:
                                            if target in ["D", "H"] and (
                                                "REGISTRO" in col
                                                or "COMUNICACAO" in col
                                                or "ELABORACAO" in col
                                            ):
                                                continue

                                            f_cols[col] = target
                                            encontrou = True
                                            break
                                    if encontrou:
                                        break

                            if len(f_cols) == 5:
                                df_clean = (
                                    df.select(list(f_cols.keys()))
                                    .rename(f_cols)
                                    .with_columns(pl.all().cast(pl.String))
                                )
                                abas.append(df_clean)

                    if abas:
                        pl.concat(abas, how="diagonal").write_parquet(path)

                        assinatura_sha256 = self.calcular_sha256(str(tmp))
                        print(f"🔒 Assinatura SHA-256 ({ano}): {assinatura_sha256}", file=sys.stdout)

                        m_data = json.load(open(self.meta)) if self.meta.exists() else {}
                        m_data[str(ano)] = {"tamanho_bytes": sz, "sha256": assinatura_sha256}
                        with open(self.meta, "w") as f:
                            json.dump(m_data, f)
                    else:
                        raise Exception("Nenhum dado criminal estruturado encontrado.")

                    if os.path.exists(tmp):
                        os.remove(tmp)
                else:
                    raise Exception("Falha ao baixar arquivo apos multiplas tentativas.")

            except Exception as e:
                print(f"❌ Erro Crítico ao processar {ano}: {e}", file=sys.stderr)
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise e

        arquivos_limpos = list(self.pastas["raw"].glob("*.parquet"))
        if not arquivos_limpos:
            msg = "Portal SSP inoperante e sem cache local integro."
            self.discord.notificar_erro("SafeDriver Sync", msg)
            raise Exception(msg)

        # DECISÃO: preciso reconstruir Prata/Ouro?
        ouro_path = self.pastas["ouro"] / "dashboard_final.parquet"
        valid_path = self.pastas["ouro"] / "validacao_modelo.parquet"
        precisa_reconstruir = True

        if ouro_path.exists() and valid_path.exists():
            try:
                df_ouro = pl.read_parquet(ouro_path)
                colunas_ok = all(
                    c in df_ouro.columns
                    for c in ["ESCORE_RISCO", "GEOMETRIA_WKT", "CODIGO_H3", "PERFIL_VITIMA"]
                )
                if colunas_ok and not novo:
                    precisa_reconstruir = False
            except Exception:
                precisa_reconstruir = True

        if not precisa_reconstruir:
            print("Nenhuma atualização pendente. Camada Ouro já aderente às novas regras.")
            self.discord.notificar_info(
                "SafeDriver Info", "Sistema sincronizado com a SSP e regras de negócio."
            )
            return

        # CAMADA PRATA
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
                    pl.when(pl.col("N").str.to_uppercase().str.contains("ROUBO|FURTO"))
                    .then(pl.lit("PATRIMONIO"))
                    .otherwise(pl.lit("PESSOA"))
                    .alias("TIPO_CRIME"),
                    pl.when(pl.col("HORA_CRIME").is_between(6, 18))
                    .then(pl.lit("DIA"))
                    .otherwise(pl.lit("NOITE"))
                    .alias("PERIODO_DIA"),
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
                        .str.contains("VEICULO|AUTO|CARGA")
                    )
                    .then(pl.lit("MOTORISTA"))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("TRANSEUNTE|PEDESTRE")
                    )
                    .then(pl.lit("PEDESTRE"))
                    .when(
                        pl.col("N").str.to_uppercase().str.contains("MOTO")
                    )
                    .then(pl.lit("MOTOCICLISTA"))
                    .when(
                        pl.col("N")
                        .str.to_uppercase()
                        .str.contains("BICICLETA")
                    )
                    .then(pl.lit("CICLISTA"))
                    .otherwise(pl.lit("GERAL"))
                    .alias("PERFIL_VITIMA"),
                ]
            )
            .collect()
        )

        prata.with_columns(
            pl.col("LAT").hash().alias("ID_ANONIMO")
        ).drop(["DATA_FATO", "H", "HORA_CRIME", "N"]).write_parquet(
            self.pastas["prata"] / "camada_prata.parquet"
        )

        # CAMADA OURO (modelo)
        print("🧠 Treinando IA Preditiva...", file=sys.stdout)
        c = prata.select(["LAT", "LON"]).unique().to_pandas()
        c["CODIGO_H3"] = c.apply(
            lambda r: h3.latlng_to_cell(r["LAT"], r["LON"], 8), axis=1
        )

        fato = (
            prata.join(pl.from_pandas(c), on=["LAT", "LON"])
            .group_by(
                [
                    "CODIGO_H3",
                    "PERIODO_DIA",
                    "TIPO_CRIME",
                    "PERFIL_VITIMA",
                    "EH_FERIADO",
                    "SEMANA_PAGAMENTO",
                ]
            )
            .agg(
                [
                    pl.len().alias("TOTAL_OCORRENCIAS"),
                    pl.col("LAT").mean().alias("LATITUDE_MEDIA"),
                    pl.col("LON").mean().alias("LONGITUDE_MEDIA"),
                ]
            )
        )

        X = fato.select(
            ["LATITUDE_MEDIA", "LONGITUDE_MEDIA", "EH_FERIADO", "SEMANA_PAGAMENTO"]
        ).to_pandas()
        y = np.log1p(fato.select("TOTAL_OCORRENCIAS").to_numpy().ravel())

        ens = VotingRegressor(
            [
                ("c", CatBoostRegressor(iterations=100, silent=True)),
                ("l", LGBMRegressor(n_estimators=100, verbose=-1)),
            ]
        ).fit(X, y)
        preds = ens.predict(X)
        risco_avg = np.mean(np.expm1(preds))

        fato_ouro = fato.with_columns(
            [
                pl.Series("ESCORE_RISCO", np.round(np.expm1(preds), 2)),
                pl.col("CODIGO_H3")
                .map_elements(self.wkt, return_dtype=pl.String)
                .alias("GEOMETRIA_WKT"),
            ]
        )

        fato_ouro.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")

        # CAMADA DE VALIDAÇÃO (real x previsto por H3)
        prata_h3 = prata.join(pl.from_pandas(c), on=["LAT", "LON"])
        crimes_reais = (
            prata_h3.group_by("CODIGO_H3")
            .agg(pl.len().alias("CRIMES_REAIS"))
        )
        escore_medio = (
            fato_ouro.group_by("CODIGO_H3")
            .agg(pl.col("ESCORE_RISCO").mean().alias("ESCORE_RISCO"))
        )
        validacao = crimes_reais.join(escore_medio, on="CODIGO_H3", how="inner")
        validacao.write_parquet(self.pastas["ouro"] / "validacao_modelo.parquet")

        # INTERPRETABILIDADE (SHAP)
        sd = pd.DataFrame(
            shap.TreeExplainer(ens.estimators_[0]).shap_values(X),
            columns=X.columns,
        ).abs().mean().to_frame("GRAU_IMPORTANCIA").reset_index()
        sd.columns = ["VARIAVEL", "GRAU_IMPORTANCIA"]
        pl.from_pandas(sd).write_parquet(self.pastas["ouro"] / "shap_audit.parquet")

        # PUBLICAÇÃO NO BIGQUERY (opcional, se envs estiverem definidos)
        bq_project = os.environ.get("BQ_PROJECT_ID")
        bq_dataset = os.environ.get("BQ_DATASET_ID")
        bq_cred_json = os.environ.get("BQ_SERVICE_ACCOUNT_JSON")

        status_bq = "🔌 BigQuery desativado (variáveis de ambiente não configuradas)."
        if bq_project and bq_dataset and bq_cred_json:
            try:
                enviar_para_bigquery(
                    fato_ouro,
                    tabela="sd_dashboard_final",
                    projeto=bq_project,
                    dataset=bq_dataset,
                    cred_json=bq_cred_json,
                )
                enviar_para_bigquery(
                    validacao,
                    tabela="sd_validacao_modelo",
                    projeto=bq_project,
                    dataset=bq_dataset,
                    cred_json=bq_cred_json,
                )
                enviar_para_bigquery(
                    pl.from_pandas(sd),
                    tabela="sd_shap_audit",
                    projeto=bq_project,
                    dataset=bq_dataset,
                    cred_json=bq_cred_json,
                )
                status_bq = "✅ Tabelas sd_dashboard_final, sd_validacao_modelo e sd_shap_audit atualizadas."
                print("✅ Tabelas publicadas no BigQuery com sucesso.", file=sys.stdout)
            except Exception as e:
                status_bq = f"⚠️ Falha ao publicar no BigQuery: {e}"
                print(status_bq, file=sys.stderr)
                # Alerta informativo no Discord (não é erro fatal do pipeline)
                self.discord.notificar_info(
                    "SafeDriver BigQuery",
                    f"O pipeline executou, mas houve falha ao atualizar o BigQuery:\n`{e}`",
                )

        # BACKUP R2
        status_cloud = "❌ Desconectado"
        if self.s3:
            try:
                for f in self.pastas["ouro"].glob("*.parquet"):
                    self.s3.upload_file(str(f), self.bucket, f"ouro/{f.name}")
                status_cloud = "✅ Upload Realizado"
            except:
                status_cloud = "⚠️ Falha no Backup R2"

        tempo_total = time.time() - self.t_inicio
        self.discord.notificar_sucesso(
            "Execução Concluída",
            tempo_total,
            prata.height,
            risco_avg,
            status_cloud,
            status_bq,
        )


if __name__ == "__main__":
    app = SafeDriver()
    try:
        app.processar()
    except Exception as e:
        err = traceback.format_exc()
        print("\n" + "=" * 50, file=sys.stderr)
        print("🚨 ERRO FATAL NO PIPELINE 🚨", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        print(err, file=sys.stderr)
        app.discord.notificar_erro("Falha Sistêmica", err)
        sys.exit(1)
