import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import unicodedata

class SafeDriverEngine:
    """
    Pipeline preditivo com dados da SSP-SP.
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
   
        self.colunas_obrigatorias = [
            'NATUREZA_APURADA', 'NOME_MUNICIPIO', 'DATA_OCORRENCIA_BO', 
            'HORA_OCORRENCIA_BO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 
            'LATITUDE', 'LONGITUDE'
        ]

       
        self.naturezas_alvo = ['FURTO DE VEICULO', 'FURTO DE CARGA', 'EXTORSAO MEDIANTE A SEQUESTRO', 'ROUBO DE VEICULO', 'ROUBO DE CARGA', 'LATROCINIO']
        self.tipos_validos = ['VIA PUBLICA', 'RODOVIA/ESTRADA']
        self.subtipos_validos = ['VIA PUBLICA', 'TRANSEUNTE', 'ACOSTAMENTO', 'AREA DE DESCANSO', 'BALANCA', 'CICLOFAIXA', 'DE FRENTE A RESIDENCIA DA VITIMA', 'FEIRA LIVRE', 'INTERIOR DE VEICULO DE CARGA', 'INTERIOR DE VEICULO DE PARTICULAR', 'POSTO DE AUXILIO', 'POSTO DE FISCALIZACAO', 'POSTO POLICIAL', 'PRACA', 'PRACA DE PEDAGIO', 'SEMAFORO', 'TUNEL/VIADUTO/PONTE', 'VEICULO EM MOVIMENTO']
        
        self.pesos_legais = {'FURTO': 1.0, 'ROUBO': 2.5, 'EXTORSAO': 5.0, 'LATROCINIO': 5.0}

    def _limpar_texto(self, texto):
        if not isinstance(texto, str): return str(texto)
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

    def _notificar_discord(self, mensagem, tipo="sucesso"):
        webhook_url = os.environ.get('DISCORD_ERRO') if tipo == "erro" else os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        try:
            requests.post(webhook_url, json={"content": f"**[SafeDriver]** {mensagem}"}, timeout=10)
        except: pass

    def _iniciar_persistencia(self):
        secret_json = os.environ.get('FIREBASE_JSON')
        if not secret_json: return None
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(json.loads(secret_json))
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except: return None

    def executar_pipeline(self):
        print(f"[{datetime.now()}] Iniciando Pipeline...")
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{datetime.now().year}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
    
            df_full = pd.read_excel(io.BytesIO(r.content), header=None)
            
            header_row = 0
            for i, row in df_full.head(20).iterrows(): # Varre as 20 primeiras linhas
                vals = [self._limpar_texto(str(v)) for v in row.values]
                if 'NATUREZA_APURADA' in vals or 'DATA_OCORRENCIA_BO' in vals:
                    header_row = i
                    break
            
        
            df = pd.read_excel(io.BytesIO(r.content), skiprows=header_row)
            df.columns = [self._limpar_texto(c) for c in df.columns]
            
    
            if 'DESCR_TIPOLOCAL' not in df.columns and 'DESCR_SUBTIPOLOCAL' in df.columns:
                df['DESCR_TIPOLOCAL'] = df['DESCR_SUBTIPOLOCAL']

        except Exception as e:
            self._notificar_discord(f"Falha na leitura: {e}", "erro")
            return

  
        if 'NATUREZA_APURADA' not in df.columns:
            self._notificar_discord("Coluna NATUREZA_APURADA nao encontrada no arquivo.", "erro")
            return

        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df['NAT_LIMPA'] = df['NATUREZA_APURADA'].apply(self._limpar_texto)
        df['SUB_LIMPO'] = df['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto)
        df['TIPO_LIMPO'] = df['DESCR_TIPOLOCAL'].apply(self._limpar_texto) if 'DESCR_TIPOLOCAL' in df.columns else df['SUB_LIMPO']

        mask = (df['NAT_LIMPA'].isin(self.naturezas_alvo)) & (df['SUB_LIMPO'].isin(self.subtipos_validos))
        df = df[mask]

        if df.empty:
            self._notificar_discord("Nenhum crime encontrado para os filtros aplicados.", "erro")
            return

        def calcular(row):
            nat, sub = row['NAT_LIMPA'], row['SUB_LIMPO']
            peso = self.pesos_legais['FURTO']
            if 'ROUBO' in nat: peso = self.pesos_legais['ROUBO']
            if 'EXTORSAO' in nat or 'LATROCINIO' in nat: peso = self.pesos_legais['LATROCINIO']
            
            locais_pedestre = ['TRANSEUNTE', 'CICLOFAIXA', 'PRACA', 'FEIRA LIVRE']
            perfil = 'pedestre' if any(x in sub for x in locais_pedestre) else 'motorista'
            return pd.Series([perfil, peso])

        df[['PERFIL', 'PESO_LEGAL']] = df.apply(calcular, axis=1)
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({'PESO_LEGAL': 'sum'}).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SEVERIDADE']
        grid['SCORE_RISCO'] = (grid['SEVERIDADE'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

      
        df.to_csv("dados_publicos_safedriver.csv", index=False)
        if self.db:
            batch = self.db.batch()
            for i, row in grid.iterrows():
                doc_ref = self.db.collection('niveis_risco').document(f"{row['GEOHASH']}_{row['PERFIL']}")
                batch.set(doc_ref, {
                    'geohash': str(row['GEOHASH']), 'perfil': str(row['PERFIL']),
                    'score': float(row['SCORE_RISCO']), 'timestamp': firestore.SERVER_TIMESTAMP
                })
                if (i + 1) % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            batch.commit()
            self._notificar_discord(f"Sucesso: {len(grid)} pontos atualizados.", "sucesso")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
