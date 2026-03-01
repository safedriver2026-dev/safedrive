import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, holidays, json, time
from datetime import datetime
from xgboost import XGBRegressor
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverFinalEngine:
    def __init__(self):
        # Conecta no banco de dados (Firestore)
        self.db = self._iniciar_firebase()
        
      
        self.naturezas = ['FURTO DE VEÍCULO', 'FURTO DE CARGA', 'ROUBO DE VEÍCULO', 'ROUBO DE CARGA', 'LATROCINIO', 'EXTORSÃO MEDIANTE A SEQUESTRO']
        self.tipos_local = ['VIA PÚBLICA', 'RODOVIA/ESTRADA']

    def _iniciar_firebase(self):
        if not firebase_admin._apps:
            secret_json = os.environ.get('FIREBASE_JSON')
            if not secret_json:
                raise ValueError("Chave JSON não configurada")
            
       
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        return firestore.client()

    def classificar_perfil(self, row):
        # Lógica para separar os ícones igual no Google Maps
        nat = str(row['NATUREZA_APURADA']).upper()
        sub = str(row['DESCR_SUBTIPOLOCAL']).upper()
        
        if 'CICLOFAIXA' in sub or 'BICICLETA' in nat: return 'ciclista'
        if 'MOTO' in nat or 'MOTOCICLETA' in nat: return 'motociclista'
        if 'TRANSEUNTE' in sub or 'PEDESTRE' in nat: return 'pedestre'
        
       
        return 'motorista'

    def executar_pipeline(self):
        print(f"[{datetime.now()}] -> Começando a baixar tudo de 2022 até hoje...")
        dfs = []
        ano_atual = datetime.now().year
        
        # PEGAR OS DADOS
        for ano in range(2022, ano_atual + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            try:
                res = requests.get(url, timeout=120)
                if res.status_code == 200:
                    df_ano = pd.read_excel(io.BytesIO(res.content), engine='openpyxl')
                    df_ano.columns = [c.upper().strip() for c in df_ano.columns]
                    
                  
                    df_ano = df_ano[df_ano['NATUREZA_APURADA'].isin(self.naturezas)]
                    dfs.append(df_ano)
                    print(f"   ✓ Dados de {ano} baixados!")
            except:
                print(f"   x Ano {ano} ainda não está no site da SSP.")

        # Junta todos os anos em uma tabela só
        df_total = pd.concat(dfs, ignore_index=True)

        # LIMPEZA DE COORDENADAS
        df_total['LATITUDE'] = pd.to_numeric(df_total['LATITUDE'], errors='coerce')
        df_total['LONGITUDE'] = pd.to_numeric(df_total['LONGITUDE'], errors='coerce')
        df_total = df_total.dropna(subset=['LATITUDE', 'LONGITUDE'])
        
        # Cria o "Quadrado" no mapa
        df_total['GRID_ID'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df_total['LATITUDE'], df_total['LONGITUDE'])]
        df_total['CATEGORIA'] = df_total.apply(self.classificar_perfil, axis=1)
        
        # PREPARANDO A IA SÉRIE TEMPORAL
        df_total['DATA'] = pd.to_datetime(df_total['DATA_OCORRENCIA_BO'], errors='coerce')
        df_total['MES'], df_total['ANO'] = df_total['DATA'].dt.month, df_total['DATA'].dt.year
        
        # Agrupa para contar quantos crimes teve por lugar, mês e tipo de usuario
        grid = df_total.groupby(['GRID_ID', 'CATEGORIA', 'ANO', 'MES']).size().reset_index(name='CRIMES')
        grid = grid.sort_values(['GRID_ID', 'CATEGORIA', 'ANO', 'MES'])
        
        # "Faz a IA olhar o que aconteceu no mês passado para prever o próximo
        grid['HISTORICO_1'] = grid.groupby(['GRID_ID', 'CATEGORIA'])['CRIMES'].shift(1).fillna(0)
        grid['TENDENCIA'] = grid['CRIMES'] - grid['HISTORICO_1']

        # 4. TREINANDO O MODELO 
        print("-> Ensinando a IA a identificar o risco...")
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        features = ['MES', 'HISTORICO_1', 'TENDENCIA']
        model.fit(grid[features], grid['CRIMES'])
        
        # Cria o Score de Risco (de 0 a 10) para o App usar
        grid['SCORE_RISCO'] = model.predict(grid[features]).clip(min=0)
        grid['SCORE_RISCO'] = (grid['SCORE_RISCO'] / grid['SCORE_RISCO'].max() * 10).round(1)

        # SALVANDO PARA O POWER BI 
        # Gera o arquivo Parquet
        grid.to_parquet("safedriver_analytics_history.parquet", index=False, compression='snappy')

        # ENVIANDO PRO FIREBASE 
        # Pega só o último mês da lista para não encher o banco de dados do App com lixo
        u_ano, u_mes = grid['ANO'].max(), grid[grid['ANO'] == grid['ANO'].max()]['MES'].max()
        dados_atuais = grid[(grid['ANO'] == u_ano) & (grid['MES'] == u_mes)]

        print(f"-> Subindo {len(dados_atuais)} pontos de risco para a nuvem...")
        batch = self.db.batch()
        for i, row in dados_atuais.iterrows():
            # ID único que mistura Lugar + Perfil (ex: "7h2j_ciclista")
            doc_id = f"{row['GRID_ID']}_{row['CATEGORIA']}"
            doc_ref = self.db.collection('niveis_risco').document(doc_id)
            
            # Salva os dados para o App só desenhar a rota
            batch.set(doc_ref, {
                'grid_id': row['GRID_ID'],
                'categoria': row['CATEGORIA'],
                'risco': float(row['SCORE_RISCO']),
                'tendencia': "SUBINDO" if row['TENDENCIA'] > 0 else "CAINDO",
                'atualizado_em': firestore.SERVER_TIMESTAMP
            })
            
          
            if (i+1) % 400 == 0:
                batch.commit()
                batch = self.db.batch()
        
        batch.commit()
        print(" Tudo pronto! App atualizado e Base do Power BI gerada.")

if __name__ == "__main__":
    SafeDriverFinalEngine().executar_pipeline()
