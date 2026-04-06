import sys
import os
import json
import requests
import traceback
import polars as pl
import pandas as pd
import numpy as np
import h3
import gc
import time
import holidays
import warnings
import shap
import fastexcel
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

class MotorSafeDriverCloud:
    def __init__(self):
        self.raiz = Path(".")
        self.bucket_nome = os.environ.get("GCP_BUCKET_NAME")
        self.pastas = {
            "raw": self.raiz / "datalake" / "raw",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        self.hoje = datetime.now()
        self.webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        self.storage_client = storage.Client()
        self.feriados_br = holidays.Brazil(years=[self.hoje.year, self.hoje.year-1, self.hoje.year-2])

    def garantir_infraestrutura_bucket(self):
        try:
            self.storage_client.get_bucket(self.bucket_nome)
        except Exception:
            self.storage_client.create_bucket(self.bucket_nome, location="US-EAST1")

    # ==========================================
    # CAMADA RAW: Ingestão de Alta Performance (Polars + FastExcel)
    # ==========================================
    def processar_camada_raw(self):
        ano_inicio, ano_atual = 2022, self.hoje.year
        mapeamento = {
            'DATAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATA DO FATO': 'DATA_OCORRENCIA_BO',
            'NATUREZA': 'RUBRICA', 'NUMERO_BOLETIM': 'NUM_BO', 'NÚMERO DO BO': 'NUM_BO'
        }

        print("🟤 [LAYER RAW] Iniciando ingestão textual segura...")
        for ano in range(ano_inicio, ano_atual + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            xlsx_temp = self.pastas["raw"] / f"temp_{ano}.xlsx"
            parquet_raw = self.pastas["raw"] / f"ssp_bruto_{ano}.parquet"
            
            try:
                h = requests.head(url, verify=False, timeout=15)
                remoto = int(h.headers.get('Content-Length', 0))
                if parquet_raw.exists():
                    print(f"✅ {ano} já em cache.")
                    continue
            except: pass

            for t in range(3):
                try:
                    print(f"📥 Baixando {ano} (T{t+1})...")
                    r = requests.get(url, stream=True, verify=False, headers={'User-Agent': 'Mozilla/5.0'}, timeout=(60, 1800))
                    if r.status_code == 200:
                        with open(xlsx_temp, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=2*1024*1024): f.write(chunk)
                        break
                except: time.sleep(5)

            if not xlsx_temp.exists(): continue

            try:
                # FastExcel lê os metadados instantaneamente
                excel_reader = fastexcel.read_excel(str(xlsx_temp))
                abas_validas = [
                    aba for aba in excel_reader.sheet_names 
                    if "capa" not in aba.lower() and "campo" not in aba.lower() and "tabela" not in aba.lower()
                ]
                
                dfs_ano = []
                for aba in abas_validas:
                    # Polars lê nativamente via Calamine (Rust)
                    df_aba = pl.read_excel(str(xlsx_temp), sheet_name=aba, engine="calamine")
                    
                    # Padroniza nomes e força tudo para String
                    df_aba = df_aba.rename({c: str(c).upper().strip() for c in df_aba.columns})
                    for velho, novo in mapeamento.items():
                        if velho in df_aba.columns:
                            df_aba = df_aba.rename({velho: novo})
                            
                    df_aba = df_aba.with_columns(pl.all().cast(pl.String))
                    dfs_ano.append(df_aba)
                
                if dfs_ano:
                    df_ano_completo = pl.concat(dfs_ano, how="diagonal")
                    df_ano_completo.write_parquet(parquet_raw, compression='snappy')
                    print(f"💾 Raw salva: {parquet_raw.name} ({df_ano_completo.height} linhas)")
                
                os.remove(xlsx_temp)
                gc.collect()
            except Exception as e:
                print(f"❌ Erro ao processar o Excel de {ano}: {e}")

    # ==========================================
    # CAMADA PRATA: Lazy Evaluation e Geoprocessamento Inteligente
    # ==========================================
    def processar_camada_prata(self):
        print("⚪ [LAYER PRATA] Iniciando tipagem e tratamento (Lazy Evaluation)...")
        arquivos_raw = [str(p) for p in self.pastas["raw"].glob("ssp_bruto_*.parquet")]
        if not arquivos_raw: raise ValueError("Datalake Raw vazio.")
        
        # O LazyFrame não gasta RAM, apenas planeja a execução
        lazy_df = pl.scan_parquet(arquivos_raw)
        
        # Pipeline de Limpeza de Dados
        df_prata = (
            lazy_df
            .with_columns([
                # Remove nulos e converte datas
                pl.col("NUM_BO").replace(["nan", "NaN", "NULL", "None", ""], None),
                pl.col("DATA_OCORRENCIA_BO").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_DT"),
            ])
            .filter(pl.col("NUM_BO").is_not_null())
            .unique(subset=["NUM_BO"], keep="last")
        )

        if "LATITUDE" in df_prata.columns and "LONGITUDE" in df_prata.columns:
            df_prata = df_prata.with_columns([
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
            ]).filter(
                (pl.col("LAT").is_not_null()) & (pl.col("LON").is_not_null()) &
                (pl.col("LAT") != 0.0) & (pl.col("LON") != 0.0) &
                (pl.col("DATA_DT").is_not_null())
            )
        else:
            raise ValueError("Colunas espaciais não encontradas.")

        # Executa a limpeza estrutural e joga para a memória
        df_prata = df_prata.collect()
        
        print("📍 Calculando H3 Vetorizado (Alta Velocidade)...")
        # Extrai só locais únicos para não calcular o H3 milhões de vezes
        coords_unicas = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords_unicas['H3'] = coords_unicas.apply(lambda row: h3.latlng_to_cell(row['LAT'], row['LON'], 8), axis=1)
        coords_pl = pl.from_pandas(coords_unicas)
        
        # Junta a inteligência H3 na base inteira instantaneamente
        df_prata = df_prata.join(coords_pl, on=["LAT", "LON"], how="left")
        
        parquet_prata = self.pastas["prata"] / "camada_prata_limpa.parquet"
        df_prata.write_parquet(parquet_prata, compression='snappy')
        print(f"✨ Prata finalizada! {df_prata.height} registros aprovados.")
        return df_prata

    # ==========================================
    # CAMADA OURO & ML: Inteligência de Máquina
    # ==========================================
    def processar_camada_ouro_e_ml(self, df):
        print("🟡 [LAYER OURO] Iniciando Feature Engineering e Modelagem...")
        t_ini = time.time()
        v_bruto = df.height
        
        col_crime = 'NATUREZA_APURADA' if 'NATUREZA_APURADA' in df.columns else 'RUBRICA'
        
        # Feature Engineering com Polars
        df = df.with_columns([
            pl.when((self.hoje - pl.col("DATA_DT")).dt.total_days() <= 180).then(3.0).otherwise(1.0).alias("PESO"),
            pl.col(col_crime).str.contains("ROUBO").alias("IS_ROUBO"),
            pl.col(col_crime).str.contains("FURTO").alias("IS_FURTO"),
            pl.col("DATA_DT").dt.hour().alias("HORA")
        ])

        df = df.with_columns([
            pl.when(pl.col("IS_ROUBO")).then(20)
              .when(pl.col("IS_FURTO")).then(10)
              .otherwise(5).alias("RISCO_BASE")
        ]).with_columns((pl.col("RISCO_BASE") * pl.col("PESO")).alias("RISCO"))

        df = df.with_columns([
            pl.when(pl.col("HORA") <= 6).then(0)
              .when(pl.col("HORA") <= 12).then(1)
              .when(pl.col("HORA") <= 18).then(2)
              .otherwise(3).alias("TURNO")
        ])

        # Agrupamento OLAP
        fato_pl = df.group_by(["H3", "TURNO", "DATA_DT"]).agg([
            pl.col("RISCO").sum().alias("RISCO"),
            pl.col("LAT").mean().alias("LAT"),
            pl.col("LON").mean().alias("LON")
        ]).sort("DATA_DT")

        fato_pl = fato_pl.with_columns([
            pl.col("DATA_DT").dt.weekday().alias("DIA_SEM"),
            pl.col("DATA_DT").dt.month().alias("MES"),
            pl.col("DATA_DT").dt.day().is_in([5,6,7,20,21]).cast(pl.Int8).alias("IS_PGTO"),
            pl.col("DATA_DT").dt.date().is_in(self.feriados_br).cast(pl.Int8).alias("IS_FERIADO")
        ])

        # Entrega ao SciKit-Learn (Pandas)
        fato = fato_pl.to_pandas()
        
        X = fato[['LAT', 'LON', 'TURNO', 'DIA_SEM', 'MES', 'IS_PGTO', 'IS_FERIADO']]
        y = np.log1p(fato['RISCO'])
        X_s = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, shuffle=False)

        print("🧠 Iniciando Auto-Tuning do Motor...")
        melhor_r2, melhor_cat, melhor_lgb = -1, None, None
        for _ in range(10):
            d, lr = int(np.random.choice([4,6,8])), float(np.random.choice([0.01, 0.05, 0.1]))
            c = CatBoostRegressor(iterations=500, depth=d, learning_rate=lr, silent=True).fit(X_tr, y_tr)
            l = LGBMRegressor(n_estimators=500, max_depth=d, learning_rate=lr, verbose=-1).fit(X_tr, y_tr)
            r2 = r2_score(y_te, (c.predict(X_te)*0.7 + l.predict(X_te)*0.3))
            if r2 > melhor_r2: melhor_r2, melhor_cat, melhor_lgb = r2, c, l
            if melhor_r2 > 0.42: break

        print("🔍 Extraindo Inteligência SHAP...")
        explainer = shap.TreeExplainer(melhor_cat)
        shap_vals = explainer.shap_values(pd.DataFrame(X_te[:1000], columns=X.columns))
        importancias = pd.DataFrame({'VAR': X.columns, 'SHAP': np.abs(shap_vals).mean(axis=0)}).sort_values('SHAP', ascending=False)
        top_driver = importancias.iloc[0]['VAR']
        
        fato['PREVISAO_RISCO'] = np.round(np.expm1((melhor_cat.predict(X_s)*0.7) + (melhor_lgb.predict(X_s)*0.3)), 2)

        # Retorna para o Polars para gravação otimizada
        importancias_pl = pl.from_pandas(importancias)
        fato_final_pl = pl.from_pandas(fato[['H3', 'DATA_DT', 'TURNO', 'RISCO', 'PREVISAO_RISCO']])

        importancias_pl.write_parquet(self.pastas["ouro"] / "explicabilidade_shap.parquet", compression='snappy')
        fato_final_pl.write_parquet(self.pastas["ouro"] / "dashboard_risco_real.parquet", compression='snappy')

        r2_tr = r2_score(y_tr, (melhor_cat.predict(X_tr)*0.7 + melhor_lgb.predict(X_tr)*0.3))
        manifesto = {
            "auditoria_estatistica": {
                "r2_treino": float(r2_tr), "r2_teste": float(melhor_r2),
                "degradacao_overfitting": float(r2_tr - melhor_r2)
            },
            "linhas_processadas": int(v_bruto), "timestamp": self.hoje.isoformat()
        }
        with open(self.pastas["auditoria"] / "auditoria_pipeline.json", "w") as f: json.dump(manifesto, f, indent=4)
        
        self.garantir_infraestrutura_bucket()
        self.fazer_upload_diretorio(self.raiz / "datalake")
        self._notificar_sucesso(melhor_r2, r2_tr, top_driver)

    def fazer_upload_diretorio(self, local):
        bucket = self.storage_client.get_bucket(self.bucket_nome)
        for aqv in Path(local).rglob("*"):
            if aqv.is_file(): bucket.blob(str(aqv.relative_to(self.raiz))).upload_from_filename(str(aqv))

    def _notificar_sucesso(self, r2_te, r2_tr, driver):
        if not self.webhook_sucesso: return
        payload = { "embeds": [{
            "title": "⚡ SafeDriver: Pipeline Estado da Arte (Polars)", "color": 3066993, "fields": [
                {"name": "📊 R2 Teste", "value": f"{r2_te:.2%}", "inline": True},
                {"name": "📉 R2 Treino", "value": f"{r2_tr:.2%}", "inline": True},
                {"name": "🧠 Decisão Guiada Por (SHAP)", "value": f"**{driver}**", "inline": False}
            ]
        }]}
        requests.post(self.webhook_sucesso, json=payload)

if __name__ == "__main__":
    try:
        motor = MotorSafeDriverCloud()
        motor.processar_camada_raw()
        df_prata = motor.processar_camada_prata()
        motor.processar_camada_ouro_e_ml(df_prata)
    except Exception:
        if os.environ.get("DISCORD_ERRO"):
            requests.post(os.environ.get("DISCORD_ERRO"), json={"content": f"❌ Erro Crítico:\n
http://googleusercontent.com/immersive_entry_chip/0
