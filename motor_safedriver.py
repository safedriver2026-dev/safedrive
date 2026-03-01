import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, holidays, json
from datetime import datetime
from xgboost import XGBRegressor
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverAutonomous:
    def __init__(self):
        self.db = self._iniciar_firebase()
        
        self.naturezas = ['FURTO DE VEÍCULO', 'FURTO DE CARGA', 'ROUBO DE VEÍCULO', 'ROUBO DE CARGA', 'LATROCINIO', 'EXTORSÃO MEDIANTE A SEQUESTRO']

    def _iniciar_firebase(self):
        if not firebase_admin._apps:
            secret_json = os.environ.get('FIREBASE_JSON')
            if not secret_json: return None 
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        return firestore.client()

    def classificar_perfil(self, row):
        nat, sub = str(row['NATUREZA_APURADA']).upper(), str(row['DESCR_SUBTIPOLOCAL']).upper()
        if 'CICLOFAIXA' in sub or 'BICICLETA' in nat: return 'ciclista'
        if 'MOTO' in nat: return 'motociclista'
        if 'TRANSEUNTE' in sub or 'PEDESTRE' in nat: return 'pedestre'
        return 'motorista'

    def executar(self):
        print("-> Iniciando extração 2022-2026...")
        dfs = []
        for ano in range(2022, datetime.now().year + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            try:
                res = requests.get(url, timeout=120)
                df_ano = pd.read_excel(io.BytesIO(res.content), engine='openpyxl')
                df_ano.columns = [c.upper().strip() for c in df_ano.columns]
                dfs.append(df_ano[df_ano['NATUREZA_APURADA'].isin(self.naturezas)])
            except: pass

        df = pd.concat(dfs, ignore_index=True)
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        
    
        df['GRID_ID'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['CATEGORIA'] = df.apply(self.classificar_perfil, axis=1)
        df['DATA'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df['MES'], df['ANO'] = df['DATA'].dt.month, df['DATA'].dt.year

       
        grid = df.groupby(['GRID_ID', 'CATEGORIA', 'ANO', 'MES']).size().reset_index(name='CRIMES')
        grid = grid.sort_values(['GRID_ID', 'CATEGORIA', 'ANO', 'MES'])
        grid['HISTORICO_1'] = grid.groupby(['GRID_ID', 'CATEGORIA'])['CRIMES'].shift(1).fillna(0)
        
        model = XGBRegressor(n_estimators=200, learning_rate=0.1)
        model.fit(grid[['MES', 'HISTORICO_1']], grid['CRIMES'])
        grid['PREVISAO_IA'] = model.predict(grid[['MES', 'HISTORICO_1']]).clip(min=0).round(2)

       
        grid.to_parquet("safedriver_full_history.parquet", index=False)
        grid.to_csv("dados_publicos_safedriver.csv", index=False) 

       
        if self.db:
            u_ano, u_mes = grid['ANO'].max(), grid[grid['ANO'] == grid['ANO'].max()]['MES'].max()
            atuais = grid[(grid['ANO'] == u_ano) & (grid['MES'] == u_mes)]
            batch = self.db.batch()
            for i, row in atuais.iterrows():
                doc_id = f"{row['GRID_ID']}_{row['CATEGORIA']}"
                batch.set(self.db.collection('niveis_risco').document(doc_id), {
                    'risco': float(row['PREVISAO_IA']), 'categoria': row['CATEGORIA'],
                    'atualizado': firestore.SERVER_TIMESTAMP
                })
                if (i+1) % 400 == 0: batch.commit(); batch = self.db.batch()
            batch.commit()

if __name__ == "__main__":
    SafeDriverAutonomous().executar()
