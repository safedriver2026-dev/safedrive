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
  Pipeline de extração, limpeza, transformação e previsão através de dados da SSP
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
        # Colunas obrigatorias conforme escopo
        self.colunas_alvo = [
            'NATUREZA_APURADA', 'NOME_MUNICIPIO', 'DATA_OCORRENCIA_BO', 
            'HORA_OCORRENCIA_BO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 
            'LATITUDE', 'LONGITUDE'
        ]

        # Filtros de Natureza (Normalizados)
        self.naturezas_alvo = [
            'FURTO DE VEICULO', 'FURTO DE CARGA', 
            'EXTORSAO MEDIANTE A SEQUESTRO', 'ROUBO DE VEICULO', 
            'ROUBO DE CARGA', 'LATROCINIO'
        ]

        # Filtros de Localidade
        self.tipos_validos = ['VIA PUBLICA', 'RODOVIA/ESTRADA']
        self.subtipos_validos = [
            'VIA PUBLICA', 'TRANSEUNTE', 'ACOSTAMENTO', 'AREA DE DESCANSO',
            'BALANCA', 'CICLOFAIXA', 'DE FRENTE A RESIDENCIA DA VITIMA',
            'FEIRA LIVRE', 'INTERIOR DE VEICULO DE CARGA', 
            'INTERIOR DE VEICULO DE PARTICULAR', 'POSTO DE AUXILIO',
            'POSTO DE FISCALIZACAO', 'POSTO POLICIAL', 'PRACA',
            'PRACA DE PEDAGIO', 'SEMAFORO', 'TUNEL/VIADUTO/PONTE',
            'VEICULO EM MOVIMENTO'
        ]

        self.pesos_legais = {'FURTO': 1.0, 'ROUBO': 2.5, 'EXTORSAO': 5.0, 'LATROCINIO': 5.0}

    def _limpar_texto(self, texto):
        """Padroniza strings removendo acentos e espacos."""
        if not isinstance(texto, str): return str(texto)
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

    def _notificar_discord(self, mensagem, tipo="sucesso"):
        webhook_url = os.environ.get('DISCORD_ERRO') if tipo == "erro" else os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        try:
            prefixo = "[SafeDriver Falha]" if tipo == "erro" else "[SafeDriver Sucesso]"
            requests.post(webhook_url, json={"content": f"**{prefixo}**\n{mensagem}"}, timeout=10)
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
        ano = datetime.now().year
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            # Carrega o Excel sem definir cabecalho inicialmente para busca dinamica
            df_raw = pd.read_excel(io.BytesIO(r.content), header=None)
            
            # Busca em qual linha esta a coluna NATUREZA_APURADA
            header_row = 0
            for i, row in df_raw.head(10).iterrows(): # Busca nas primeiras 10 linhas
                row_cleaned = [self._limpar_texto(str(val)) for val in row]
                if 'NATUREZA_APURADA' in row_cleaned:
                    header_row = i
                    break
            
            # Recarrega o dataframe com o cabecalho correto encontrado
            df = pd.read_excel(io.BytesIO(r.content), skiprows=header_row)
            df.columns = [self._limpar_texto(c) for c in df.columns]
                
        except Exception as e:
            self._notificar_discord(f"Erro na leitura dinamica: {e}", "erro")
            return

        # Valida colunas alvo
        if 'NATUREZA_APURADA' not in df.columns:
            self._notificar_discord("Coluna NATUREZA_APURADA nao localizada apos busca dinamica.", "erro")
            return

        # Seleciona apenas o que importa e limpa nulos de coordenadas
        df = df[[c for c in self.colunas_alvo if c in df.columns]].dropna(subset=['LATITUDE', 'LONGITUDE'])

        # Aplicacao de filtros de negocio
        df['NAT_LIMPA'] = df['NATUREZA_APURADA'].apply(self._limpar_texto)
        df['TIPO_LIMPO'] = df['DESCR_TIPOLOCAL'].apply(self._limpar_texto)
        df['SUB_LIMPO'] = df['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto)

        mask = (
            (df['NAT_LIMPA'].isin(self.naturezas_alvo)) &
            (df['TIPO_LIMPO'].isin(self.tipos_validos)) &
            (df['SUB_LIMPO'].isin(self.subtipos_validos))
        )
        df = df[mask]

        if df.empty:
            self._notificar_discord("Filtros aplicados resultaram em zero registros.", "erro")
            return

        # Classificacao de Perfil e Score
        def classificar(row):
            natureza = row['NAT_LIMPA']
            tipo = row['TIPO_LIMPO']
            subtipo = row['SUB_LIMPO']
            peso = self.pesos_legais['FURTO']
            if 'ROUBO' in natureza: peso = self.pesos_legais['ROUBO']
            if 'EXTORSAO' in natureza or 'LATROCINIO' in natureza: peso = self.pesos_legais['LATROCINIO']
            
            locais_pedestre = ['TRANSEUNTE', 'CICLOFAIXA', 'PRACA', 'FEIRA LIVRE']
            perfil = 'pedestre' if tipo == 'VIA PUBLICA' and any(x in subtipo for x in locais_pedestre) else 'motorista'
            return pd.Series([perfil, peso])

        df[['PERFIL', 'PESO_LEGAL']] = df.apply(classificar, axis=1)
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        # Agregacao para o Firestore
        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({'PESO_LEGAL': 'sum'}).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SEVERIDADE']
        grid['SCORE_RISCO'] = (grid['SEVERIDADE'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

        # Exporta CSV
        df.drop(columns=['NAT_LIMPA', 'TIPO_LIMPO', 'SUB_LIMPO']).to_csv("dados_publicos_safedriver.csv", index=False)

        # Firestore
        if self.db:
            batch = self.db.batch()
            for i, row in grid.iterrows():
                doc_id = f"{row['GEOHASH']}_{row['PERFIL']}"
                doc_ref = self.db.collection('niveis_risco').document(doc_id)
                batch.set(doc_ref, {
                    'geohash': str(row['GEOHASH']),
                    'perfil': str(row['PERFIL']),
                    'score': float(row['SCORE_RISCO']),
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                if (i + 1) % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            batch.commit()
            self._notificar_discord(f"Sucesso: {len(grid)} quadrantes atualizados no Firestore.", "sucesso")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
