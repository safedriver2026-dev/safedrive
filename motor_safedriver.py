import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
from xgboost import XGBRegressor
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverLegalEngine:
    """
    Motor do SafeDriver: Processamento de dados da Segurança Pública 
    
    """
    def __init__(self):
        self.db = self._iniciar_firebase()
        
        # Filtros de Natureza Criminal através do código penal
        self.naturezas_alvo = [
            'FURTO DE VEÍCULO', 'FURTO DE CARGA', 
            'EXTORSÃO MEDIANTE A SEQUESTRO', 'ROUBO DE VEÍCULO', 
            'ROUBO DE CARGA', 'LATROCINIO'
        ]

        
        self.tipos_permitidos = ['VIA PÚBLICA', 'RODOVIA/ESTRADA']
        
        self.subtipos_permitidos = [
            'VIA PÚBLICA', 'TRANSEUNTE', 'ACOSTAMENTO', 'ÁREA DE DESCANSO',
            'BALANÇA', 'CICLOFAIXA', 'DE FRENTE A RESIDÊNCIA DA VITÍMA',
            'FEIRA LIVRE', 'INTERIOR DE VEÍCULO DE CARGA', 
            'INTERIOR DE VEÍCULO DE PARTICULAR', 'POSTO DE AUXÍLIO',
            'POSTO DE FISCALIZAÇÃO', 'POSTO POLICIAL', 'PRAÇA',
            'PRAÇA DE PEDÁGIO', 'SEMÁFORO', 'TÚNEL/VIADUTO/PONTE',
            'VEÍCULO EM MOVIMENTO'
        ]

    def _iniciar_firebase(self):
        secret_json = os.environ.get('FIREBASE_JSON')
        if secret_json and not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        return firestore.client() if firebase_admin._apps else None

    def identificar_usuario_e_peso(self, row):
        """
        Cruza DESCR_TIPOLOCAL e DESCR_SUBTIPOLOCAL para identificar o perfil.
        Aplica pesos baseados na gravidade penal da ocorrência.
        """
        tipo = str(row['DESCR_TIPOLOCAL']).upper()
        subtipo = str(row['DESCR_SUBTIPOLOCAL']).upper()
        natureza = str(row['NATUREZA_APURADA']).upper()

        # Definição de Peso Legal 
        peso = 1.0 # Base: Furto
        if 'ROUBO' in natureza: peso = 2.5
        if 'EXTORSÃO' in natureza or 'LATROCINIO' in natureza: peso = 5.0

        # Lógica de Cruzamento para Identificação do Usuário
        # Perfil: Pedestre / Ciclista
        locais_pedestre = ['TRANSEUNTE', 'CICLOFAIXA', 'PRAÇA', 'FEIRA LIVRE']
        if tipo == 'VIA PÚBLICA' and any(x in subtipo for x in locais_pedestre):
            return pd.Series(['pedestre', peso])

        # Perfil: Motorista
        locais_motorista = [
            'VEÍCULO', 'RODOVIA', 'ACOSTAMENTO', 'SEMÁFORO', 
            'TÚNEL', 'BALANÇA', 'PEDÁGIO', 'POSTO'
        ]
        if tipo == 'RODOVIA/ESTRADA' or any(x in subtipo for x in locais_motorista):
            return pd.Series(['motorista', peso])

        # Caso Genérico em Via Pública 
        return pd.Series(['motorista', peso])

    def executar_motor(self):
        print("Iniciando processamento com cruzamento de locais...")
        
        # Carga de Dados 
        ano = datetime.now().year
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        try:
            res = requests.get(url, timeout=120)
            df = pd.read_excel(io.BytesIO(res.content))
        except:
            print("Erro ao acessar base de dados.")
            return

        # Aplicação de Filtros
        df.columns = [c.upper().strip() for c in df.columns]
        
        df = df[df['NATUREZA_APURADA'].isin(self.naturezas_alvo)]
        df = df[df['DESCR_TIPOLOCAL'].str.upper().isin(self.tipos_permitidos)]
        df = df[df['DESCR_SUBTIPOLOCAL'].str.upper().isin(self.subtipos_permitidos)]
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

        # Cruzamento de Dados e Classificação
        df[['PERFIL', 'PESO_LEGAL']] = df.apply(self.identificar_usuario_e_peso, axis=1)
        
        # Geoprocessamento 
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        # 4. Cálculo do Score de Risco 
        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({
            'PESO_LEGAL': 'sum',
            'NATUREZA_APURADA': 'count'
        }).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SOMA_PESOS', 'QTD_CRIMES']

        # Fórmula: (Severidade * 0.8) + Piso de Segurança 0.5
        grid['SCORE_RISCO'] = (grid['SOMA_PESOS'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

        # 5. Exportação para CSV
        grid['DATA_ATUALIZACAO'] = datetime.now().strftime("%d/%m/%Y %H:%M")
        grid.to_csv("dados_publicos_safedriver.csv", index=False)

        # 6. Sincronização Firestore
        if self.db:
            batch = self.db.batch()
            for i, row in grid.iterrows():
               
                doc_id = f"{row['GEOHASH']}_{row['PERFIL']}"
                doc_ref = self.db.collection('niveis_risco').document(doc_id)
                
                batch.set(doc_ref, {
                    'id': doc_id,
                    'score': float(row['SCORE_RISCO']),
                    'perfil': row['PERFIL'],
                    'geohash': row['GEOHASH'],
                    'timestamp': firestore.SERVER_TIMESTAMP
                })

                if (i + 1) % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            
            batch.commit()
            print("Sincronização finalizada.")

if __name__ == "__main__":
    SafeDriverLegalEngine().executar_motor()
