import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverEngine:
    """
  Pipeline para o tratamento e previsão de dados de segurança publica da SSP
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
        # Colunas obrigatórias para retenção 
        self.colunas_alvo = [
            'NATUREZA_APURADA', 'NOME_MUNICIPIO', 'DATA_OCORRENCIA_BO', 
            'HORA_OCORRENCIA_BO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 
            'LATITUDE', 'LONGITUDE'
        ]

        # Filtros de Natureza Criminal
        self.naturezas_alvo = [
            'FURTO DE VEÍCULO', 'FURTO DE CARGA', 
            'EXTORSÃO MEDIANTE A SEQUESTRO', 'ROUBO DE VEÍCULO', 
            'ROUBO DE CARGA', 'LATROCINIO'
        ]

        # Filtros de Localidade e Subtipo
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

        # Pesos Legais para o cálculo de severidade
        self.pesos_legais = {
            'FURTO': 1.0, 
            'ROUBO': 2.5, 
            'EXTORSAO': 5.0, 
            'LATROCINIO': 5.0 
        }

    def _iniciar_persistencia(self):
        secret_json = os.environ.get('FIREBASE_JSON')
        if secret_json and not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        return firestore.client() if firebase_admin._apps else None

    def _classificar_vetor_risco(self, row):
        tipo = str(row['DESCR_TIPOLOCAL']).upper()
        subtipo = str(row['DESCR_SUBTIPOLOCAL']).upper()
        natureza = str(row['NATUREZA_APURADA']).upper()

        peso = self.pesos_legais['FURTO']
        if 'ROUBO' in natureza: peso = self.pesos_legais['ROUBO']
        if 'EXTORSÃO' in natureza or 'LATROCINIO' in natureza: peso = self.pesos_legais['LATROCINIO']

        locais_pedestre = ['TRANSEUNTE', 'CICLOFAIXA', 'PRAÇA', 'FEIRA LIVRE']
        if tipo == 'VIA PÚBLICA' and any(x in subtipo for x in locais_pedestre):
            return pd.Series(['pedestre', peso])
            
        return pd.Series(['motorista', peso])

    def executar_pipeline(self):
        print(f"[{datetime.now()}] Iniciando Pipeline SafeDriver...")
        
        ano = datetime.now().year
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        # Etapa de leitura blindada 
        try:
            r = requests.get(url, timeout=120)
            df = pd.read_excel(io.BytesIO(r.content))
            df.columns = [str(c).strip().upper() for c in df.columns]

      
            if 'NATUREZA_APURADA' not in df.columns:
                print("-> Cabecalho fora do padrao detectado. Reajustando leitura...")
                df = pd.read_excel(io.BytesIO(r.content), skiprows=1)
                df.columns = [str(c).strip().upper() for c in df.columns]
                
        except Exception as e:
            print(f"[Erro] Falha critica na extracao: {e}")
            return

        # Verifica se as colunas obrigatorias existem antes de continuar
        colunas_presentes = [c for c in self.colunas_alvo if c in df.columns]
        if 'NATUREZA_APURADA' not in colunas_presentes:
            print(f"[Erro] Coluna NATUREZA_APURADA nao encontrada. Colunas lidas: {list(df.columns)}")
            return

        
        df = df[colunas_presentes]

     
        mask = (
            (df['NATUREZA_APURADA'].isin(self.naturezas_alvo)) &
            (df['DESCR_TIPOLOCAL'].str.upper().isin(self.tipos_validos)) &
            (df['DESCR_SUBTIPOLOCAL'].str.upper().isin(self.subtipos_validos))
        )
        df = df[mask].dropna(subset=['LATITUDE', 'LONGITUDE'])

        # Processamento e geracao de scores
        df[['PERFIL', 'PESO_LEGAL']] = df.apply(self._classificar_vetor_risco, axis=1)
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({
            'PESO_LEGAL': 'sum',
            'NATUREZA_APURADA': 'count'
        }).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SEVERIDADE', 'VOLUME']

        # Normalizacao do Score (0.5 a 10.0)
        grid['SCORE_RISCO'] = (grid['SEVERIDADE'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

        # Geracao do arquivo para Power BI com os dados limpos e filtrados
        df.to_csv("dados_publicos_safedriver.csv", index=False)
        print("[Info] Arquivo analitico exportado para Power BI.")

        # Sincronizacao Firestore com Auto-Recuperacao
        if self.db:
            docs = self.db.collection('niveis_risco').limit(1).get()
            if len(docs) == 0:
                print("[Alerta] Base Firestore vazia detectada. Executando Full Recovery...")

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
            print(f"[Sucesso] Pipeline finalizada: {len(grid)} quadrantes sincronizados no Firestore.")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
