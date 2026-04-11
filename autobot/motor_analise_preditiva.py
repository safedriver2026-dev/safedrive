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
        self.meta = self.pastas["raw"] / "meta.json"

    def cdc_check(self, ano, url, sessao, path_parquet: Path):
        try:
            r = sessao.head(url, timeout=30, verify=False, allow_redirects=True)
            size = int(r.headers.get("Content-Length", 0))

            if size <= 0 and path_parquet.exists():
                return False, size

            if self.meta.exists():
                with open(self.meta, "r") as f:
                    m = json.load(f).get(str(ano))

                if (
                    m
                    and m.get("tamanho_bytes") == size
                    and m.get("sha256") == self.calcular_sha256(str(path_parquet))
                ):
                    print(f"✅ CDC: Arquivo {ano} inalterado.", file=sys.stdout)
                    return False, size
            print(f"🔄 CDC: Arquivo {ano} alterado ou novo. Baixando...", file=sys.stdout)
            return True, size
        except requests.exceptions.RequestException as e:
            print(f"⚠️ CDC: Falha ao verificar {ano} (conexão ou 404): {e}", file=sys.stderr)
            if path_parquet.exists():
                print(f"✅ CDC: Usando cache local para {ano}.", file=sys.stdout)
                return False, 0
            print(f"❌ CDC: Sem cache local para {ano}. Requer download.", file=sys.stderr)
            return True, 0
        except Exception as e:
            print(f"❌ CDC: Erro inesperado ao verificar {ano}: {e}", file=sys.stderr)
            return True, 0

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

    def sincronizar_raw(self):
        sessao = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        sessao.mount("https://", HTTPAdapter(max_retries=retries))

        anos = range(2022, hora_brasilia().year + 1)
        for ano in anos:
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            path = self.pastas["raw"] / f"SPDadosCriminais_{ano}.parquet"
            tmp = self.pastas["raw"] / f"SPDadosCriminais_{ano}.xlsx"

            baixar, size = self.cdc_check(ano, url, sessao, path)

            if not baixar:
                continue

            try:
                print(f"⬇️ Baixando dados de {ano}...", file=sys.stdout)
                if self.baixar_robusto(url, tmp, sessao):
                    print(f"✅ Download de {ano} concluído. Processando...", file=sys.stdout)
                    abas = []
                    excel_file = pl.read_excel(tmp, sheet_id=0, engine="calamine")

                    # Mapeamento flexível de colunas
                    col_map = {
                        "ANO_BO": "ANO_BO", "ANO BO": "ANO_BO",
                        "NUM_BO": "NUM_BO", "NUM BO": "NUM_BO",
                        "DELEGACIA": "DELEGACIA",
                        "DATA_FATO": "D", "DATA FATO": "D",
                        "HORA_FATO": "H", "HORA FATO": "H",
                        "LATITUDE": "LAT",
                        "LONGITUDE": "LON",
                        "NATUREZA_CRIME": "N", "NATUREZA CRIME": "N",
                    }

                    # Tenta encontrar as colunas em cada aba
                    for sheet_name in excel_file.sheet_names:
                        df = pl.read_excel(tmp, sheet_name=sheet_name, engine="calamine")
                        f_cols = {}
                        for original_col, target_col in col_map.items():
                            if original_col in df.columns:
                                f_cols[original_col] = target_col

                        # Garante que as colunas essenciais para o processamento estejam presentes
                        if all(col in f_cols.values() for col in ["D", "LAT", "LON", "H", "N"]):
                            df_clean = (
                                df.select(list(f_cols.keys()))
                                .rename(f_cols)
                                .with_columns(pl.all().cast(pl.String))
                            )
                            abas.append(df_clean)

                    if abas:
                        pl.concat(abas, how="diagonal").write_parquet(path)

                        assinatura_sha256 = self.calcular_sha256(str(tmp))
                        print(
                            f"🔒 Assinatura SHA-256 ({ano}): {assinatura_sha256}",
                            file=sys.stdout,
                        )

                        m_data = json.load(open(self.meta)) if self.meta.exists() else {}
                        m_data[str(ano)] = {
                            "tamanho_bytes": size, # Usar o size obtido do HEAD request
                            "sha256": assinatura_sha256,
                        }
                        with open(self.meta, "w") as f:
                            json.dump(m_data, f)
                    else:
                        print(f"⚠️ Nenhum dado criminal estruturado encontrado na planilha {ano}. Pulando.", file=sys.stderr)
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        continue

                    if os.path.exists(tmp):
                        os.remove(tmp)
                else:
                    raise Exception(f"Falha ao baixar arquivo {ano} apos multiplas tentativas.")

            except Exception as e:
                print(f"❌ Erro Crítico ao processar {ano}: {e}", file=sys.stderr)
                if os.path.exists(tmp):
                    os.remove(tmp)
                self.discord.notificar_info(
                    f"SafeDriver Download SSP {ano}",
                    f"Falha ao processar o ano {ano}:\n`{e}`",
                )
                continue

    def wkt(self, h3_id: str) -> str:
        if not isinstance(h3_id, str):
            return None
        try:
            boundary = h3.h3_to_geo_boundary(h3_id, geo_json=True)
            return f"POLYGON (({', '.join([f'{lon} {lat}' for lon, lat in boundary])}))"
        except Exception:
            return None

    def publicar_bigquery_a_partir_de_arquivos(self, bq_project, bq_dataset, bq_cred_json):
        status_bq = "✅ Sucesso"
        try:
            for pasta in ["ouro"]:
                for arquivo in self.pastas[pasta].glob("*.parquet"):
                    df = pl.read_parquet(arquivo)
                    tabela_nome = arquivo.stem
                    enviar_para_bigquery(df, tabela_nome, bq_project, bq_dataset, bq_cred_json)
                    print(f"📡 Publicado {tabela_nome} no BigQuery.", file=sys.stdout)
        except Exception as e:
            status_bq = f"❌ Falha: {e}"
            print(f"❌ Erro ao publicar no BigQuery: {e}", file=sys.stderr)
        return status_bq

    def reconstruir_ouro_validacao_shap(self, bq_project, bq_dataset, bq_cred_json):
        print("✨ Construindo Camada Ouro...", file=sys.stdout)
        prata = pl.read_parquet(self.pastas["prata"] / "camada_prata.parquet")

        features_h3 = (
            prata.group_by("CODIGO_H3")
            .agg(
                pl.col("LAT").mean().alias("LATITUDE_MEDIA"),
                pl.col("LON").mean().alias("LONGITUDE_MEDIA"),
                pl.col("LAT").std().fill_null(0).alias("LAT_STD"),
                pl.col("LON").std().fill_null(0).alias("LON_STD"),
                pl.col("ID_ANONIMO").count().alias("CRIMES_REAIS"),
                (pl.col("TIPO_CRIME") == "PATRIMONIO").sum().alias("PATRIMONIO_COUNT"),
                (pl.col("PERFIL_VITIMA") == "MOTORISTA").sum().alias("MOTORISTA_COUNT"),
                (pl.col("PERFIL_VITIMA") == "MOTOCICLISTA").sum().alias("MOTO_COUNT"),
                (pl.col("PERFIL_VITIMA") == "PEDESTRE").sum().alias("PEDESTRE_COUNT"), # Adicionado
                (pl.col("PERFIL_VITIMA") == "CICLISTA").sum().alias("CICLISTA_COUNT"), # Adicionado
                (pl.col("EH_FERIADO") == 1).sum().alias("FERIADO_COUNT"),
                (pl.col("SEMANA_PAGAMENTO") == 1).sum().alias("PAGAMENTO_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "MANHA").sum().alias("MANHA_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "TARDE").sum().alias("TARDE_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "NOITE").sum().alias("NOITE_COUNT"),
                (pl.col("PERIODO_DETALHADO") == "MADRUGADA").sum().alias("MADRUGADA_COUNT"),
                pl.col("PESO_PENAL").sum().alias("PESO_PENAL_TOTAL"),
                pl.col("ANO").unique().count().alias("ANOS_COM_CRIMES"),
            )
            .with_columns(
                (pl.col("PATRIMONIO_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_PATRIMONIO"),
                (pl.col("MOTORISTA_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_MOTORISTA"),
                (pl.col("MOTO_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_MOTO"),
                (pl.col("PEDESTRE_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_PEDESTRE"), # Adicionado
                (pl.col("CICLISTA_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_CICLISTA"), # Adicionado
                (pl.col("FERIADO_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_FERIADO"),
                (pl.col("PAGAMENTO_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_PAGAMENTO"),
                (pl.col("MANHA_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_MANHA"),
                (pl.col("TARDE_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_TARDE"),
                (pl.col("NOITE_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_NOITE"),
                (pl.col("MADRUGADA_COUNT") / pl.col("CRIMES_REAIS")).alias("PROP_MADRUGADA"),
                (pl.col("CRIMES_REAIS") / pl.col("ANOS_COM_CRIMES")).alias("CRIMES_POR_ANO"),
                (pl.col("PESO_PENAL_TOTAL") / pl.col("ANOS_COM_CRIMES")).alias("CRIMES_POND_POR_ANO"),
            )
            .with_columns(
                (pl.col("PROP_NOITE") * pl.col("PROP_PATRIMONIO")).alias("RISCO_NOITE_PATRIMONIO"),
                (pl.col("PROP_MOTO") * pl.col("PROP_NOITE")).alias("RISCO_MOTO_NOITE"),
                (pl.col("PROP_MOTORISTA") * pl.col("PROP_PAGAMENTO")).alias("RISCO_MOTORISTA_PAGTO"),
            )
            .drop(
                [
                    "PATRIMONIO_COUNT",
                    "MOTORISTA_COUNT",
                    "MOTO_COUNT",
                    "PEDESTRE_COUNT", # Adicionado
                    "CICLISTA_COUNT", # Adicionado
                    "FERIADO_COUNT",
                    "PAGAMENTO_COUNT",
                    "MANHA_COUNT",
                    "TARDE_COUNT",
                    "NOITE_COUNT",
                    "MADRUGADA_COUNT",
                    "PESO_PENAL_TOTAL",
                    "ANOS_COM_CRIMES",
                ]
            )
            .with_columns(
                pl.col("CODIGO_H3")
                .map_elements(self.wkt, return_dtype=pl.String)
                .alias("GEOMETRIA_WKT")
            )
        )

        features_modelo = [
            "LATITUDE_MEDIA",
            "LONGITUDE_MEDIA",
            "LAT_STD",
            "LON_STD",
            "CRIMES_POR_ANO",
            "PROP_PATRIMONIO",
            "PROP_MOTORISTA",
            "PROP_MOTO",
            "PROP_PEDESTRE", # Adicionado
            "PROP_CICLISTA", # Adicionado
            "PROP_FERIADO",
            "PROP_PAGAMENTO",
            "PROP_MANHA",
            "PROP_TARDE",
            "PROP_NOITE",
            "PROP_MADRUGADA",
            "RISCO_NOITE_PATRIMONIO",
            "RISCO_MOTO_NOITE",
            "RISCO_MOTORISTA_PAGTO",
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
                "PROP_PEDESTRE", # Adicionado
                "PROP_CICLISTA", # Adicionado
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
        self.sincronizar_raw()

        bq_project = os.environ.get("BQ_PROJECT_ID")
        bq_dataset = os.environ.get("BQ_DATASET_ID")
        bq_cred_json = os.environ.get("BQ_SERVICE_ACCOUNT_JSON")

        arquivos_limpos = list(self.pastas["raw"].glob("*.parquet"))
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
