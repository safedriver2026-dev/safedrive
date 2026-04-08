import sys
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback
import hashlib
import polars as pl
import pandas as pd
import numpy as np
import h3
import gc
import holidays
import warnings
import re
import unicodedata
from pathlib import Path
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

print("[INICIALIZACAO] Motor Autônomo SafeDriver ativado. Carregando módulos semânticos e preditivos...", flush=True)
warnings.filterwarnings("ignore")

class MotorSafeDriverCloudflare:
    def __init__(self):
        self.raiz = Path(".")
        
        # Credenciais do Cloudflare R2 injetadas via GitHub Secrets
        self.r2_endpoint = os.environ.get("R2_ENDPOINT_URL")
        self.r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
        self.r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.bucket_nome = os.environ.get("R2_BUCKET_NAME")
        
        if not all([self.r2_endpoint, self.r2_access_key, self.r2_secret_key, self.bucket_nome]):
            raise ValueError("[ERRO_FATAL] Credenciais do Cloudflare R2 ausentes no ambiente.")
            
        self.pastas = {
            "bruto": self.raiz / "datalake" / "raw",
            "processado": self.raiz / "datalake" / "prata",
            "refinado": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria",
            "relatorios": self.raiz / "datalake" / "relatorios"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        self.hoje = datetime.now()
        
        # Inicialização do Cliente S3 para Cloudflare R2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.r2_endpoint,
            aws_access_key_id=self.r2_access_key,
            aws_secret_access_key=self.r2_secret_key
        )
        
        self.assinaturas_seguranca = {}
        self.anomalias_detectadas = 0
        self.registros_brutos = 0
        self.registros_validados = 0
        self.registros_descartados = 0

        ano_atual = self.hoje.year
        calendario_feriados = holidays.Brazil(subdiv='SP', years=[ano_atual - 2, ano_atual - 1, ano_atual])
        self.datas_feriados = list(calendario_feriados.keys())

    def gerar_assinatura_criptografica(self, caminho):
        sha256_hash = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco_bytes in iter(lambda: f.read(4096), b""):
                sha256_hash.update(bloco_bytes)
        return sha256_hash.hexdigest()

    def validar_perimetro_geografico(self, df):
        return df.filter(
            (pl.col("LAT").is_between(-25.5, -19.5)) & 
            (pl.col("LON").is_between(-53.5, -44.0))
        )

    def criar_sessao_cacada(self):
        sessao = requests.Session()
        estrategia_retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        sessao.mount("https://", HTTPAdapter(max_retries=estrategia_retry))
        sessao.headers.update({'User-Agent': 'Mozilla/5.0'})
        return sessao

    @staticmethod
    def achatar_texto(texto):
        t_sem_acento = ''.join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^A-Z0-9]', '', t_sem_acento.upper())

    def normalizador_semantico(self, df, ano):
        df = df.rename({c: str(c).upper().strip() for c in df.columns})
        dicionario_sinonimos = {
            'NUMEROBOLETIM': 'NUM_BO', 'NUMEROBO': 'NUM_BO', 'BOLETIM': 'NUM_BO', 'NUMBO': 'NUM_BO',
            'DATAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATA': 'DATA_OCORRENCIA_BO',
            'NATUREZA': 'RUBRICA', 'DESCRICAOLOCAL': 'DESCR_TIPOLOCAL',
            'LAT': 'LATITUDE', 'LON': 'LONGITUDE', 'LATITUDE': 'LATITUDE', 'LONGITUDE': 'LONGITUDE'
        }
        mapa_renomeacao = {}
        for col_original in df.columns:
            col_achatada = self.achatar_texto(col_original)
            if col_achatada in dicionario_sinonimos:
                alvo = dicionario_sinonimos[col_achatada]
                if alvo not in df.columns and alvo not in mapa_renomeacao.values():
                    mapa_renomeacao[col_original] = alvo
        if mapa_renomeacao: df = df.rename(mapa_renomeacao)
        if "NUM_BO" not in df.columns:
            df = df.with_columns([pl.format("VIRT_BO_{}_{}", pl.lit(ano), pl.int_range(0, pl.len())).alias("NUM_BO")])
        return df

    def executar_extracao_dados(self):
        print("[COLETOR_DADOS] Iniciando sincronização multi-aba...", flush=True)
        anos_foco = [self.hoje.year - 2, self.hoje.year - 1, self.hoje.year]
        sessao = self.criar_sessao_cacada()
        
        for ano in anos_foco:
            arquivo_bruto = self.pastas["bruto"] / f"ssp_{ano}.parquet"
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            
            try:
                r = sessao.get(url, stream=True, timeout=180)
                if r.status_code == 200:
                    temp_xlsx = self.pastas["bruto"] / "temp.xlsx"
                    with open(temp_xlsx, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=2*1024*1024): f.write(chunk)
                    
                    import fastexcel
                    excel = fastexcel.read_excel(str(temp_xlsx))
                    df_abas_validas = []
                    indicadores = ['NUMBO', 'NUMEROBO', 'BOLETIM', 'LATITUDE', 'DATAOCORRENCIA']
                    
                    for aba in excel.sheet_names:
                        df_temp = pl.read_excel(str(temp_xlsx), sheet_name=aba, engine="calamine")
                        col_achatadas = [self.achatar_texto(c) for c in df_temp.columns]
                        if any(ind in col_achatadas for ind in indicadores) and len(df_temp.columns) > 5:
                            df_temp = self.normalizador_semantico(df_temp, ano).with_columns(pl.all().cast(pl.String))
                            df_abas_validas.append(df_temp)
                            print(f" -> Aba processada: {aba}", flush=True)
                            
                    if df_abas_validas:
                        df_novo = pl.concat(df_abas_validas, how="diagonal")
                        df_novo = df_novo.unique(subset=["NUM_BO"])
                        df_novo.write_parquet(arquivo_bruto)
                        self.assinaturas_seguranca[f"ssp_{ano}"] = self.gerar_signature_criptografica(arquivo_bruto)
                    os.remove(temp_xlsx)
                    gc.collect()
            except Exception as e: print(f"[ERRO_EXTRAÇÃO] Ano {ano}: {e}", flush=True)

    def executar_limpeza_dados(self):
        print("[REFINADOR_DADOS] Higienizando Data Lake...", flush=True)
        arquivos = list(self.pastas["bruto"].glob("*.parquet"))
        lf = pl.concat([pl.scan_parquet(str(f)) for f in arquivos], how="diagonal")
        
        self.registros_brutos = lf.select(pl.len()).collect().item()
        df_processado = (
            lf.with_columns([
                pl.col("DATA_OCORRENCIA_BO").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_DT"),
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
            ])
            .filter(pl.col("LAT").is_not_null() & pl.col("DATA_DT").is_not_null())
            .unique(subset=["NUM_BO"]).collect()
        )
        df_processado = self.validar_perimetro_geografico(df_processado)
        self.registros_validados = df_processado.height
        
        # LGPD: Destruição do identificador original
        df_processado = df_processado.with_columns(
            pl.format("SEC_KEY_{}", pl.col("NUM_BO").fill_null("NULL").hash(42)).alias("ID_ANONIMO")
        ).drop("NUM_BO")

        coords = df_processado.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        
        df_final = df_processado.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")
        df_final.write_parquet(self.pastas["processado"] / "camada_prata.parquet")
        return df_final

    def executar_predicao_inteligente(self, df):
        print("[NUCLEO_PREDITIVO] Calibrando modelos LightGBM + CatBoost...", flush=True)
        df_eng = df.with_columns([
            pl.col("DATA_DT").dt.month().alias("MES"),
            pl.col("DATA_DT").dt.hour().alias("HORA"),
            pl.col("DATA_DT").dt.weekday().alias("DIA_SEMANA"),
            pl.col("DATA_DT").dt.day().is_between(28, 31).cast(pl.Int8).alias("PAGAMENTO")
        ])

        fato = df_eng.group_by(["H3", "MES", "HORA", "DIA_SEMANA", "PAGAMENTO"]).agg([
            pl.col("LAT").mean().alias("LAT"), pl.col("LON").mean().alias("LON"), pl.len().alias("INCIDENTES")
        ])

        X = StandardScaler().fit_transform(fato.select(["LAT", "LON", "MES", "HORA", "PAGAMENTO"]).to_pandas())
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        
        m1 = CatBoostRegressor(iterations=100, silent=True).fit(X, y)
        m2 = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        pred = np.expm1((m1.predict(X) * 0.7) + (m2.predict(X) * 0.3))
        
        fato = fato.with_columns([pl.Series("PREVISAO_FINAL", np.round(pred, 2))])
        fato.write_parquet(self.pastas["refinado"] / "dashboard_final.parquet")
        
        # Sincronização Cloudflare R2
        print("[SINCRONIZACAO_REMOTA] Transmitindo para o Cloudflare R2...", flush=True)
        for f in self.raiz.rglob("datalake/*/*"):
            if f.is_file():
                caminho_blob = f.relative_to(self.raiz).as_posix()
                self.s3_client.upload_file(str(f), self.bucket_nome, caminho_blob)

if __name__ == "__main__":
    try:
        motor = MotorSafeDriverCloudflare()
        motor.executar_extracao_dados()
        df = motor.executar_limpeza_dados()
        motor.executar_predicao_inteligente(df)
    except Exception:
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
