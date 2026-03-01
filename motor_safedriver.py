import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverEngine:
    """
Pipeline de ETL e previsão para dados da segurança publica SSP
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
        self.colunas_alvo = [
            'NATUREZA_APURADA', 'NOME_MUNICIPIO', 'DATA_OCORRENCIA_BO', 
            'HORA_OCORRENCIA_BO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 
            'LATITUDE', 'LONGITUDE'
        ]

        # Naturezas criminais definidas no escopo do projeto
        self.naturezas_alvo = [
            'FURTO DE VEICULO', 'FURTO DE CARGA', 
            'EXTORSAO MEDIANTE A SEQUESTRO', 'ROUBO DE VEICULO', 
            'ROUBO DE CARGA', 'LATROCINIO'
        ]

        # Filtros espaciais
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

     
        self.pesos_legais = {
            'FURTO': 1.0, 
            'ROUBO': 2.5, 
            'EXTORSAO': 5.0, 
            'LATROCINIO': 5.0 
        }

    def _iniciar_persistencia(self):
        print("[Log] Inicializando conexao com o Firebase...")
        secret_json = os.environ.get('FIREBASE_JSON')
        
        if not secret_json:
            print("[Erro] Variavel de ambiente FIREBASE_JSON nao localizada. Persistencia em nuvem abortada.")
            return None
            
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(json.loads(secret_json))
                firebase_admin.initialize_app(cred)
            print("[Sucesso] Conexao com Firestore estabelecida com exito.")
            return firestore.client()
        except Exception as e:
            print(f"[Erro] Falha ao autenticar credenciais do Firebase: {e}")
            return None

    def _classificar_vetor_risco(self, row):
        tipo = str(row['DESCR_TIPOLOCAL']).upper()
        subtipo = str(row['DESCR_SUBTIPOLOCAL']).upper()
        natureza = str(row['NATUREZA_APURADA']).upper()

        peso = self.pesos_legais['FURTO']
        if 'ROUBO' in natureza: peso = self.pesos_legais['ROUBO']
        if 'EXTORSAO' in natureza or 'LATROCINIO' in natureza: peso = self.pesos_legais['LATROCINIO']

        locais_pedestre = ['TRANSEUNTE', 'CICLOFAIXA', 'PRACA', 'FEIRA LIVRE']
        if tipo == 'VIA PUBLICA' and any(x in subtipo for x in locais_pedestre):
            return pd.Series(['pedestre', peso])
            
        return pd.Series(['motorista', peso])

    def executar_pipeline(self):
        print(f"[{datetime.now()}] Iniciando Pipeline Operacional SafeDriver...")
        
        ano = datetime.now().year
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            df = pd.read_excel(io.BytesIO(r.content))
            
            # Padronizacao inicial de cabecalhos
            df.columns = [str(c).strip().upper().replace('Í', 'I').replace('Â', 'A').replace('É', 'E').replace('Ç', 'C') for c in df.columns]

            if 'NATUREZA_APURADA' not in df.columns:
                print("[Alerta] Formato atipico detectado no cabecalho. Aplicando correcao de leitura...")
                df = pd.read_excel(io.BytesIO(r.content), skiprows=1)
                df.columns = [str(c).strip().upper().replace('Í', 'I').replace('Â', 'A').replace('É', 'E').replace('Ç', 'C') for c in df.columns]
                
        except Exception as e:
            print(f"[Erro] Interrupcao na extracao de dados da SSP-SP: {e}")
            return

        colunas_presentes = [c for c in self.colunas_alvo if c in df.columns]
        if 'NATUREZA_APURADA' not in colunas_presentes:
            print(f"[Erro] Estrutura de dados incompativel. Colunas detectadas: {list(df.columns)}")
            return

        df = df[colunas_presentes]

        # Tratamento de caracteres especiais 
        df['NATUREZA_APURADA'] = df['NATUREZA_APURADA'].str.upper().str.replace('Í', 'I').str.replace('Ã', 'A')
        df['DESCR_TIPOLOCAL'] = df['DESCR_TIPOLOCAL'].str.upper().str.replace('Ú', 'U')
        df['DESCR_SUBTIPOLOCAL'] = df['DESCR_SUBTIPOLOCAL'].str.upper().str.replace('Ê', 'E').str.replace('Á', 'A').str.replace('Í', 'I').str.replace('Ç', 'C')

        # Aplicacao de filtros de dominio
        mask = (
            (df['NATUREZA_APURADA'].isin([n.replace('Í', 'I').replace('Ã', 'A') for n in self.naturezas_alvo])) &
            (df['DESCR_TIPOLOCAL'].isin([t.replace('Ú', 'U') for t in self.tipos_validos])) &
            (df['DESCR_SUBTIPOLOCAL'].isin([s.replace('Ê', 'E').replace('Á', 'A').replace('Í', 'I').replace('Ç', 'C') for s in self.subtipos_validos]))
        )
        
        df = df[mask].dropna(subset=['LATITUDE', 'LONGITUDE'])

        if df.empty:
            print("[Alerta] Nenhum registro atendeu aos criterios de filtro estipulados.")
            return

        # Computacao do risco e indexacao geoespacial
        df[['PERFIL', 'PESO_LEGAL']] = df.apply(self._classificar_vetor_risco, axis=1)
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({
            'PESO_LEGAL': 'sum',
            'NATUREZA_APURADA': 'count'
        }).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SEVERIDADE', 'VOLUME']

        # Normalizacao e definicao do piso de risco
        grid['SCORE_RISCO'] = (grid['SEVERIDADE'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

        # Exportacao para camada analitica
        df.to_csv("dados_publicos_safedriver.csv", index=False)
        print("[Sucesso] Arquivo de dados CSV exportado para consumo no Power BI.")

        # Integracao operacional com Firebase
        if self.db is not None:
            print(f"[Log] Iniciando gravacao em lote no Firestore. Total de areas processadas: {len(grid)}")
            
            try:
                batch = self.db.batch()
                contador = 0
                
                for i, row in grid.iterrows():
                    doc_id = f"{row['GEOHASH']}_{row['PERFIL']}"
                    doc_ref = self.db.collection('niveis_risco').document(doc_id)
                    
                    esquema_app = {
                        'geohash': str(row['GEOHASH']),
                        'perfil': str(row['PERFIL']),
                        'score': float(row['SCORE_RISCO']),
                        'base_legal': 'CPB',
                        'timestamp': firestore.SERVER_TIMESTAMP
                    }
                    
                    batch.set(doc_ref, esquema_app)
                    contador += 1

                    if contador == 400:
                        batch.commit()
                        batch = self.db.batch()
                        contador = 0
                
                if contador > 0:
                    batch.commit()
                
                print("[Sucesso] Sincronizacao de dados com o Firestore concluida perfeitamente.")
            except Exception as e:
                print(f"[Erro] Falha critica durante escrita no banco de dados: {e}")
        else:
            print("[Aviso] A rotina de gravacao no Firestore foi suprimida devido a falhas de autenticacao.")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
