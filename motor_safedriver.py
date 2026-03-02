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
    Engine SafeDriver - Versao de Alta Resiliencia.
    Varredura completa de cabecalho para evitar erros de leitura da SSP-SP.
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
      
        self.naturezas_alvo = ['FURTO DE VEICULO', 'FURTO DE CARGA', 'EXTORSAO MEDIANTE A SEQUESTRO', 'ROUBO DE VEICULO', 'ROUBO DE CARGA', 'LATROCINIO']
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
         
            df = pd.read_excel(io.BytesIO(r.content))
            
          
            df.columns = [self._limpar_texto(str(c)) for c in df.columns]

         
            if 'NATUREZA_APURADA' not in df.columns:
                print("Tentando localizar cabecalho na linha 1...")
                new_header = df.iloc[0] 
                df = df[1:] 
                df.columns = [self._limpar_texto(str(c)) for c in new_header]

        except Exception as e:
            self._notificar_discord(f"Falha na leitura do Excel: {e}", "erro")
            return

   
        if 'NATUREZA_APURADA' not in df.columns:
        
            colunas_lidas = ", ".join(list(df.columns)[:5])
            self._notificar_discord(f"ERRO: NATUREZA_APURADA nao encontrada. Primeiras colunas lidas: {colunas_lidas}", "erro")
            return

   
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df['NAT_LIMPA'] = df['NATUREZA_APURADA'].apply(self._limpar_texto)
        df['SUB_LIMPO'] = df['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto) if 'DESCR_SUBTIPOLOCAL' in df.columns else ""

        mask = (df['NAT_LIMPA'].isin(self.naturezas_alvo)) & (df['SUB_LIMPO'].isin(self.subtipos_validos))
        df = df[mask]

        if df.empty:
            self._notificar_discord("Nenhum dado encontrado apos os filtros.", "erro")
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
                doc_id = f"{row['GEOHASH']}_{row['PERFIL']}"
                batch.set(self.db.collection('niveis_risco').document(doc_id), {
                    'geohash': str(row['GEOHASH']), 'perfil': str(row['PERFIL']),
                    'score': float(row['SCORE_RISCO']), 'timestamp': firestore.SERVER_TIMESTAMP
                })
                if (i + 1) % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            batch.commit()
            self._notificar_discord(f"Sucesso: {len(grid)} pontos atualizados no Firestore.", "sucesso")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
