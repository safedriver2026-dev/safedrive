import sys, os, requests, traceback, hashlib, gc, warnings, re, time
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import polars as pl
import pandas as pd
import numpy as np
import h3, holidays, boto3
from catboost import CatBoostRegressor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

class NotificadorDiscord:
    def __init__(self):
        self.url_sucesso = os.environ.get("DISCORD_SUCESSO", "").strip()
        self.url_erro = os.environ.get("DISCORD_ERRO", "").strip()

    def enviar(self, webhook_url, titulo, mensagem, cor):
        if not webhook_url: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "footer": {"text": f"SafeDriver AI • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except: pass

    def relatar_sucesso(self, total_linhas, tempo_segundos, anos_proc):
        msg = (
            "**Status:** 🟢 Operação Concluída\n"
            f"**Anos Processados:** {anos_proc}\n"
            f"**Volume de Dados:** {total_linhas:,} registros limpos\n"
            f"**Tempo de Processamento:** {tempo_segundos:.1f} segundos\n"
            "**Destino:** Data Lake atualizado (Cloudflare R2)."
        )
        self.enviar(self.url_sucesso, "📊 Relatório Executivo - Pipeline", msg, 3066993)

    def relatar_erro(self, erro_msg):
        # Formatado de forma blindada contra quebras de linha acidentais
        detalhes_erro = str(erro_msg)[:3800]
        msg = f"**Status:** 🔴 Falha Crítica\n**Detalhes:**\n```python\n{detalhes_erro}\n```"
        self.enviar(self.url_erro, "⚠️ Alerta Operacional", msg, 15158332)

class SafeDriverMotor:
    def __init__(self):
        self.raiz = Path(".")
        self.hoje = datetime.now()
        self.discord = NotificadorDiscord()
        
        def clean(key): return os.environ.get(key, "").strip()
        self.r2_cfg = {
            "url": clean("R2_ENDPOINT_URL"),
            "key": clean("R2_ACCESS_KEY_ID"),
            "secret": clean("R2_SECRET_ACCESS_KEY"),
            "bucket": clean("R2_BUCKET_NAME")
        }
        
        self.pastas = {p: self.raiz / "datalake" / p for p in ["raw", "prata", "ouro"]}
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)

        self.s3 = boto3.client('s3', endpoint_url=self.r2_cfg["url"], 
                               aws_access_key_id=self.r2_cfg["key"], 
                               aws_secret_access_key=self.r2_cfg["secret"], region_name="auto")
        
        self.feriados = list(holidays.Brazil(subdiv='SP', years=range(2022, 2027)).keys())

    def limpar_esquema(self, df):
        cols_originais = [c.upper().strip() for c in df.columns]
        df.columns = cols_originais
        
        mapeamento = {
            'LAT': ['LATITUDE', 'LAT'], 'LON': ['LONGITUDE', 'LON'],
            'DATA_REF': ['DATAOCORRENCIA', 'DATA_OCORRENCIA_BO', 'DATA_OCORRENCIA', 'DATA_REF'],
            'HORA_REF': ['HORAOCORRENCIA', 'HORA_OCORRENCIA_BO', 'HORA_REF'],
            'NATUREZA_RAW': ['RUBRICA', 'NATUREZA_APURADA', 'NATUREZA']
        }
        
        colunas_finais = {}
        for destino, opcoes in mapeamento.items():
            for opt in opcoes:
                if opt in df.columns:
                    colunas_finais[opt] = destino
                    break 
        
        return df.rename(colunas_finais).select([c for c in colunas_finais.values()])

    def gerar_wkt_h3(self, h3_index):
        """Gera o polígono para o mapa do Looker Studio"""
        try:
            limites = h3.h3_to_geo_boundary(h3_index, geo_json=True)
            coords = ", ".join([f"{lon} {lat}" for lon, lat in limites])
            coords += f", {limites[0][0]} {limites[0][1]}"
            return f"POLYGON(({coords}))"
        except:
            return None

    def executar(self):
        inicio_timer = time.time()
        print("[SISTEMA] Processamento iniciado.", flush=True)
        
        s = requests.Session()
        s.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=3)))
        s.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'})

        anos_processados = []
        for ano in range(2022, self.hoje.year + 1):
            arq_raw = self.pastas["raw"] / f"ssp_{ano}.parquet"
            if arq_raw.exists() and ano < self.hoje.year: 
                anos_processados.append(ano)
                continue

            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            try:
                r = s.get(url, timeout=300, verify=False)
                if r.status_code == 200:
                    temp = self.pastas["raw"] / "temp.xlsx"
                    with open(temp, "wb") as f: f.write(r.content)
                    
                    import fastexcel
                    excel = fastexcel.read_excel(str(temp))
                    abas_ano = []
                    for aba in excel.sheet_names:
                        df_tmp = pl.read_excel(str(temp), sheet_name=aba, engine="calamine")
                        if len(df_tmp.columns) > 5:
                            df_tmp = self.limpar_esquema(df_tmp)
                            if "LAT" in df_tmp.columns and "DATA_REF" in df_tmp.columns:
                                abas_ano.append(df_tmp.with_columns(pl.all().cast(pl.String)))
                    
                    if abas_ano:
                        pl.concat(abas_ano, how="diagonal").write_parquet(arq_raw)
                        anos_processados.append(ano)
                        print(f" -> Ano {ano} OK.", flush=True)
                    if os.path.exists(temp): os.remove(temp)
            except Exception as e: 
                print(f"Erro download {ano}: {e}")

        arquivos = [str(f) for f in self.pastas["raw"].glob("*.parquet")]
        if not arquivos:
            raise ValueError("Nenhum arquivo RAW encontrado para processar.")

        print("Processando inteligência e filtros...", flush=True)
        lfs = [pl.scan_parquet(f) for f in arquivos]
        lf = pl.concat(lfs, how="diagonal")
        
        df_prata = lf.with_columns([
            pl.col("DATA_REF").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DT"),
            pl.col("LAT").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("LON").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("HORA_REF").str.slice(0, 2).cast(pl.Int8, strict=False).alias("H_INT")
        ]).filter(
            (pl.col("LAT").is_between(-25.5, -19.5)) & 
            (pl.col("LON").is_between(-53.5, -44.0)) &
            (pl.col("DT").is_not_null())
        ).with_columns([
            pl.when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("HOMICIDIO|LESAO|AMEACA|ESTUPRO|LATROCINIO"))
              .then(pl.lit("CONTRA_A_PESSOA"))
              .when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("ROUBO|FURTO|ESTELIONATO|DANO"))
              .then(pl.lit("AO_PATRIMONIO"))
              .otherwise(pl.lit("OUTROS")).alias("NATUREZA_CRIME"),
              
            pl.when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("VEICULO|CARGA|AUTO|MOTO|ONIBUS"))
              .then(pl.lit("MOTORISTA"))
              .when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("BICICLETA|BIKE"))
              .then(pl.lit("CICLISTA"))
              .otherwise(pl.lit("PEDESTRE")).alias("PERFIL"),

            pl.when(pl.col("H_INT").is_between(6, 11)).then(pl.lit("MANHA"))
              .when(pl.col("H_INT").is_between(12, 17)).then(pl.lit("TARDE"))
              .when(pl.col("H_INT").is_between(18, 23)).then(pl.lit("NOITE"))
              .otherwise(pl.lit("MADRUGADA")).alias("TURNO"),

            pl.col("DT").dt.date().is_in(self.feriados).cast(pl.Int8).alias("IS_FERIADO"),
            ((pl.col("DT").dt.day().is_between(28, 31)) | (pl.col("DT").dt.day().is_between(1, 7))).cast(pl.Int8).alias("IS_PAGAMENTO")
        ]).collect()

        linhas_totais = df_prata.height

        # ANONIMIZAÇÃO E SEGURANÇA LGPD
        df_prata = df_prata.with_columns(pl.col("LAT").hash(seed=100).alias("ID_ANONIMO"))
        cols_limpeza = [c for c in df_prata.columns if ("NUM" in c.upper() and "ANON" not in c.upper())]
        cols_limpeza += ["DATA_REF", "HORA_REF", "H_INT", "NATUREZA_RAW"]
        
        df_prata = df_prata.drop([c for c in cols_limpeza if c in df_prata.columns])
        df_prata.write_parquet(self.pastas["prata"] / "camada_prata.parquet")

        print("Gerando IA e sincronizando...", flush=True)
        coords = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        df_final = df_prata.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")

        fato = df_final.group_by(["H3", "PERFIL", "TURNO", "NATUREZA_CRIME", "IS_FERIADO", "IS_PAGAMENTO"]).agg([
            pl.len().alias("INCIDENTES"), pl.col("LAT").mean().alias("LAT_M"), pl.col("LON").mean().alias("LON_M")
        ])

        X = fato.select(["LAT_M", "LON_M", "IS_FERIADO", "IS_PAGAMENTO"]).to_pandas()
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        modelo = CatBoostRegressor(iterations=100, silent=True).fit(X, y)
        fato = fato.with_columns(pl.Series("RISCO_SCORE", np.round(np.expm1(modelo.predict(X)), 2)))

        # Aplica a geometria WKT para o Looker Studio
        fato = fato.with_columns(
            pl.col("H3").map_elements(self.gerar_wkt_h3, return_dtype=pl.String).alias("GEOMETRIA_WKT")
        )

        fato.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")
        
        for f in self.raiz.rglob("datalake/*/*"):
            if f.is_file():
                self.s3.upload_file(str(f), self.r2_cfg["bucket"], f.relative_to(self.raiz).as_posix())

        # Relatório de Sucesso pro Discord
        tempo_total = time.time() - inicio_timer
        self.discord.relatar_sucesso(linhas_totais, tempo_total, sorted(list(set(anos_processados))))

if __name__ == "__main__":
    motor = SafeDriverMotor()
    try:
        motor.executar()
    except Exception as e:
        erro_formatado = traceback.format_exc()
        motor.discord.relatar_erro(erro_formatado)
        sys.exit(1)
