import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

class SafeDriverEngine:
    """
    Engine de Processamento Preditivo para Seguranca Publica com Observabilidade via Discord.
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
        self.colunas_alvo = [
            'NATUREZA_APURADA', 'NOME_MUNICIPIO', 'DATA_OCORRENCIA_BO', 
            'HORA_OCORRENCIA_BO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 
            'LATITUDE', 'LONGITUDE'
        ]

        self.naturezas_alvo = [
            'FURTO DE VEICULO', 'FURTO DE CARGA', 
            'EXTORSAO MEDIANTE A SEQUESTRO', 'ROUBO DE VEICULO', 
            'ROUBO DE CARGA', 'LATROCINIO'
        ]

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

    def _notificar_discord(self, mensagem, tipo="sucesso"):
        """Envia um relatorio operacional para o canal do Discord baseado no tipo de evento."""
        if tipo == "erro":
            webhook_url = os.environ.get('DISCORD_ERRO')
            prefixo = "[SafeDriver Falha Critica]"
        else:
            webhook_url = os.environ.get('DISCORD_SUCESSO')
            prefixo = "[SafeDriver Operacao Concluida]"

        if not webhook_url:
            print(f"[Aviso] Variavel do Discord ({tipo}) nao configurada. Notificacao ignorada.")
            return
            
        try:
            payload = {"content": f"**{prefixo}**\n{mensagem}"}
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            print(f"[Erro] Falha ao enviar alerta para o Discord ({tipo}): {e}")

    def _iniciar_persistencia(self):
        print("[Log] Inicializando conexao com o Firebase...")
        secret_json = os.environ.get('FIREBASE_JSON')
        
        if not secret_json:
            msg_erro = "Variavel FIREBASE_JSON nao localizada. O upload para o banco foi abortado."
            print(f"[Erro Critico] {msg_erro}")
            self._notificar_discord(f"[FALHA DE AMBIENTE] {msg_erro}", tipo="erro")
            return None
            
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(json.loads(secret_json))
                firebase_admin.initialize_app(cred)
            print("[Sucesso] Conexao com Firestore estabelecida com exito.")
            return firestore.client()
        except Exception as e:
            msg_erro = f"Falha ao decodificar chave do Firebase. Verifique a formatacao do JSON no GitHub. Detalhes: {e}"
            print(f"[Erro Critico] {msg_erro}")
            self._notificar_discord(f"[FALHA DE CREDENCIAL] {msg_erro}", tipo="erro")
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
            df.columns = [str(c).strip().upper().replace('Í', 'I').replace('Â', 'A').replace('É', 'E').replace('Ç', 'C') for c in df.columns]

            if 'NATUREZA_APURADA' not in df.columns:
                df = pd.read_excel(io.BytesIO(r.content), skiprows=1)
                df.columns = [str(c).strip().upper().replace('Í', 'I').replace('Â', 'A').replace('É', 'E').replace('Ç', 'C') for c in df.columns]
                
        except Exception as e:
            self._notificar_discord(f"[FALHA DE EXTRACAO] Nao foi possivel baixar a planilha da SSP: {e}", tipo="erro")
            return

        colunas_presentes = [c for c in self.colunas_alvo if c in df.columns]
        if 'NATUREZA_APURADA' not in colunas_presentes:
            self._notificar_discord("[FALHA DE ESTRUTURA] Colunas essenciais estao ausentes na planilha da SSP.", tipo="erro")
            return

        df = df[colunas_presentes]

        df['NATUREZA_APURADA'] = df['NATUREZA_APURADA'].str.upper().str.replace('Í', 'I').str.replace('Ã', 'A')
        df['DESCR_TIPOLOCAL'] = df['DESCR_TIPOLOCAL'].str.upper().str.replace('Ú', 'U')
        df['DESCR_SUBTIPOLOCAL'] = df['DESCR_SUBTIPOLOCAL'].str.upper().str.replace('Ê', 'E').str.replace('Á', 'A').str.replace('Í', 'I').str.replace('Ç', 'C')

        mask = (
            (df['NATUREZA_APURADA'].isin([n.replace('Í', 'I').replace('Ã', 'A') for n in self.naturezas_alvo])) &
            (df['DESCR_TIPOLOCAL'].isin([t.replace('Ú', 'U') for t in self.tipos_validos])) &
            (df['DESCR_SUBTIPOLOCAL'].isin([s.replace('Ê', 'E').replace('Á', 'A').replace('Í', 'I').replace('Ç', 'C') for s in self.subtipos_validos]))
        )
        
        df = df[mask].dropna(subset=['LATITUDE', 'LONGITUDE'])

        if df.empty:
            msg = "A planilha foi processada, mas nenhum registro atendeu aos criterios dos filtros. O banco de dados nao recebeu atualizacoes neste ciclo."
            print(f"[Alerta] {msg}")
            self._notificar_discord(f"[ALERTA DE NEGOCIO] {msg}", tipo="erro")
            return

        df[['PERFIL', 'PESO_LEGAL']] = df.apply(self._classificar_vetor_risco, axis=1)
        df['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]

        grid = df.groupby(['GEOHASH', 'PERFIL']).agg({
            'PESO_LEGAL': 'sum',
            'NATUREZA_APURADA': 'count'
        }).reset_index()
        grid.columns = ['GEOHASH', 'PERFIL', 'SEVERIDADE', 'VOLUME']

        grid['SCORE_RISCO'] = (grid['SEVERIDADE'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

        df.to_csv("dados_publicos_safedriver.csv", index=False)
        print("[Sucesso] Arquivo analitico CSV exportado.")

        if self.db is not None:
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
                
                msg_sucesso = f"ETL finalizado sem erros operacionais. {len(grid)} zonas de risco foram atualizadas no Firestore e o arquivo CSV de analise foi gerado."
                print(msg_sucesso)
                self._notificar_discord(msg_sucesso, tipo="sucesso")
            except Exception as e:
                msg_erro = f"Falha de transacao ao gravar dados no Firestore: {e}"
                print(f"[Erro Critico] {msg_erro}")
                self._notificar_discord(f"[ERRO DE GRAVACAO] {msg_erro}", tipo="erro")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
