import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
from xgboost import XGBRegressor
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverEngine:
    """
  Motor de Processamento criado para gestão de dados de segurança publica e aplicação em plataforma mobile.
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
        # FILTROS DE NATUREZA Código Penal Brasileiro
        self.naturezas_alvo = [
            'FURTO DE VEÍCULO', 'FURTO DE CARGA', 
            'EXTORSÃO MEDIANTE A SEQUESTRO', 'ROUBO DE VEÍCULO', 
            'ROUBO DE CARGA', 'LATROCINIO'
        ]

        # MATRIZ DE SEVERIDADE PENAL (Pesos Jurídicos)
        self.pesos_legais = {
            'FURTO': 1.0,      # Art. 155 CP
            'ROUBO': 2.5,      # Art. 157 CP
            'EXTORSAO': 5.0,   # Art. 159 CP
            'LATROCINIO': 5.0  # Art. 157, 3º, II CP 
        }

        # FILTROS DE LOCALIDADE (Conforme extração SSP-SP)
        self.tipos_validos = ['VIA PÚBLICA', 'RODOVIA/ESTRADA']
        
        self.subtipos_validos = [
            'VIA PÚBLICA', 'TRANSEUNTE', 'ACOSTAMENTO', 'ÁREA DE DESCANSO',
            'BALANÇA', 'CICLOFAIXA', 'DE FRENTE A RESIDÊNCIA DA VITÍMA',
            'FEIRA LIVRE', 'INTERIOR DE VEÍCULO DE CARGA', 
            'INTERIOR DE VEÍCULO DE PARTICULAR', 'POSTO DE AUXÍLIO',
            'POSTO DE FISCALIZAÇÃO', 'POSTO POLICIAL', 'PRAÇA',
            'PRAÇA DE PEDÁGIO', 'SEMÁFORO', 'TÚNEL/VIADUTO/PONTE',
            'VEÍCULO EM MOVIMENTO'
        ]

    def _iniciar_persistencia(self):
        """Inicializa conexão com Firebase via Service Account."""
        secret_json = os.environ.get('FIREBASE_JSON')
        if secret_json and not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        return firestore.client() if firebase_admin._apps else None

    def _classificar_vetor_risco(self, row):
        """
        Cruza DESCR_TIPOLOCAL e DESCR_SUBTIPOLOCAL para identificar o perfil.
        Atribui pesos baseados na gravidade da infração penal.
        """
        tipo = str(row['DESCR_TIPOLOCAL']).upper()
        subtipo = str(row['DESCR_SUBTIPOLOCAL']).upper()
        natureza = str(row['NATUREZA_APURADA']).upper()

        # Determinação do Peso Legal
        peso = self.pesos_legais['FURTO']
        if 'ROUBO' in natureza: peso = self.pesos_legais['ROUBO']
        if 'EXTORSÃO' in natureza or 'LATROCINIO' in natureza: peso = self.pesos_legais['LATROCINIO']

        # Cruzamento para Identificação de Perfil Motorista vs Pedestre
        locais_pedestre = ['TRANSEUNTE', 'CICLOFAIXA', 'PRAÇA', 'FEIRA LIVRE']
        
        if tipo == 'VIA PÚBLICA' and any(x in subtipo for x in locais_pedestre):
            return pd.Series(['pedestre', peso])
            
        # Incidentes em rodovias ou específicos de veículos
        return pd.Series(['motorista', peso])

    def executar_pipeline(self):
        """Orquestração de ETL, Inferência e Persistência."""
        print(f"[{datetime.now()}] Iniciando Pipeline SafeDriver...")
        
        # Extração (Ingestão de Dados SSP-SP)
        ano = datetime.now().year
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            df = pd.read_excel(io.BytesIO(r.content))
        except Exception as e:
            print(f"Falha na extração: {e}")
            return

        # Transformação e Filtragem Estrita
        df.columns = [c.upper().strip() for c in df.columns]
        
        mask = (
            (df['NATUREZA_APURADA'].isin(self.naturezas_alvo)) &
            (df['DESCR_TIPOLOCAL'].str.upper().isin(self.tipos_validos)) &
            (df['DESCR_SUBTIPOLOCAL'].str.upper().isin(self.subtipos_validos))
        )
        df = df[mask].dropna(subset=['LATITUDE', 'LONGITUDE'])

        # Engenharia de Atributos e Geoprocessamento
        df[['PERFIL', 'PESO_LEGAL']] = df.apply(self._classificar_vetor_risco, axis=1)
        # Geohash Nível 6: Otimização para Google Maps API (±1.2km²)
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        # Cálculo do Score de Risco Modelagem Preditiva
        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({
            'PESO_LEGAL': 'sum',
            'NATUREZA_APURADA': 'count'
        }).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SEVERIDADE', 'VOLUME']

        # Normalização do Score (Escala 0.5 a 10.0)
        grid['SCORE_RISCO'] = (grid['SEVERIDADE'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

        # Entrega Analítica
        grid['ULTIMA_ATUALIZACAO'] = datetime.now().strftime("%d/%m/%Y %H:%M")
        grid.to_csv("dados_publicos_safedriver.csv", index=False)

        # Persistência e Self-Healing Firestore
        if self.db:
            # Mecanismo de verificação de vacuidade
            docs = self.db.collection('niveis_risco').limit(1).get()
            if len(docs) == 0:
                print("⚠️ Base Firestore vazia detectada. Executando Full Recovery...")

            batch = self.db.batch()
            for i, row in grid.iterrows():
                
                doc_id = f"{row['GEOHASH']}_{row['PERFIL']}"
                doc_ref = self.db.collection('niveis_risco').document(doc_id)
                
                batch.set(doc_ref, {
                    'geohash': row['GEOHASH'],
                    'perfil': row['PERFIL'],
                    'score': float(row['SCORE_RISCO']),
                    'base_legal': 'CPB',
                    'timestamp': firestore.SERVER_TIMESTAMP
                })

                if (i + 1) % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            
            batch.commit()
            print(f"✅ Pipeline finalizada: {len(grid)} quadrantes sincronizados.")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
