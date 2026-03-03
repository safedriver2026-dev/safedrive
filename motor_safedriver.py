import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import unicodedata

class SafeDriverEngine:
    def __init__(self):
        self.db = self._iniciar_persistencia()
        self.locais_foco = [self._limpar_texto(s) for s in [
            'Via Pública', 'Transeunte', 'Acostamento', 'Ciclofaixa', 'Feira Livre', 
            'Ponto de Ônibus', 'Terminal/Estação', 'Semáforo', 'Praça', 
            'Túnel/Viaduto/Ponte', 'Veículo em Movimento', 'Rodovia/Estrada',
            'Interior de Transporte Coletivo', 'Metroviário e Ferroviário Metropolitano'
        ]]
        self.pesos = {'FURTO': 1.0, 'ROUBO': 2.5, 'CRITICO': 5.0}

    def _limpar_texto(self, texto):
        if not isinstance(texto, str): return str(texto)
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

    def _definir_periodo(self, hora):
        try:
            h = int(str(hora).split(':')[0])
            if 0 <= h < 6: return 'Madrugada'
            if 6 <= h < 12: return 'Manhã'
            if 12 <= h < 18: return 'Tarde'
            return 'Noite'
        except: return 'Indefinido'

    def _iniciar_persistencia(self):
        secret_json = os.environ.get('FIREBASE_JSON')
        if not secret_json: return None
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        return firestore.client()

    def _limpar_base_obsoleta(self):
        if not self.db: return 0
        colecao_ref = self.db.collection('niveis_risco')
        docs = colecao_ref.limit(500).stream()
        deletados = 0
        while True:
            batch = self.db.batch()
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            if count == 0: break
            batch.commit()
            deletados += count
            docs = colecao_ref.limit(500).stream()
        return deletados

    def executar_pipeline(self):
        momento_inicio = datetime.now()
        data_extenso = momento_inicio.strftime('%d/%m/%Y às %H:%M:%S')
        
        # INGESTÃO DE DADOS
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{momento_inicio.year}.xlsx"
        try:
            r = requests.get(url, timeout=120)
            df_bruto = pd.read_excel(io.BytesIO(r.content), skiprows=1)
            df_bruto.columns = [self._limpar_texto(str(c)) for c in df_bruto.columns]
            total_bruto = len(df_bruto)
        except Exception as e:
            return

        # SANEAMENTO E FILTRAGEM 
        df_bruto['LATITUDE'] = pd.to_numeric(df_bruto['LATITUDE'], errors='coerce')
        df_bruto['LONGITUDE'] = pd.to_numeric(df_bruto['LONGITUDE'], errors='coerce')
        
        df_geo = df_bruto.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df_geo = df_geo[df_geo['LATITUDE'] != 0].copy()

        def classificar_natureza(nat):
            n = self._limpar_texto(nat)
            if any(x in n for x in ['LATROCINIO', 'SEQUESTRO']): return 'ALERTA', self.pesos['CRITICO']
            if 'ROUBO' in n: return 'QUALIFICADO', self.pesos['ROUBO']
            if 'FURTO' in n: return 'QUALIFICADO', self.pesos['FURTO']
            return None, 0

        res = df_geo['NATUREZA_APURADA'].apply(classificar_natureza)
        df_geo['CATEGORIA'], df_geo['PESO_BASE'] = zip(*res)
        df_geo['PERIODO'] = df_geo['HORA_OCORRENCIA_BO'].apply(self._definir_periodo)
        
        df_qualificado = df_geo[df_geo['CATEGORIA'].notna()].copy()
        total_uteis = len(df_qualificado)
        eficiencia = (total_uteis / total_bruto) * 100 if total_bruto > 0 else 0

      
        if total_uteis == 0:
            self._notificar_discord({
                "📊 Status": "Sincronização Concluída",
                "📝 Nota": "Não foram detectados novos registros qualificados na SSP.",
                "📅 Data": data_extenso
            }, "neutro")
            return

        # TRANSFORMAÇÃO E GEOPROCESSAMENTO
        df_qualificado['perfil'] = df_qualificado['CATEGORIA'].apply(lambda x: ['Pedestre', 'Ciclista', 'Motorista'] if x == 'ALERTA' else ['Pedestre', 'Ciclista'])
        df_final = df_qualificado.explode('perfil')
        df_final['geohash'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df_final['LATITUDE'], df_final['LONGITUDE'])]

        # FIRESTORE
        if self.db:
            removidos = self._limpar_base_obsoleta()
            
       
            grid = df_final.groupby(['geohash', 'perfil', 'PERIODO']).size().reset_index(name='frequencia')
            grid['score'] = (grid['frequencia'] * 2.1).clip(0.5, 10.0).round(2)

            batch = self.db.batch()
            for i, row in grid.iterrows():
                doc_id = f"{row['geohash']}_{row['perfil']}_{row['PERIODO']}"
                batch.set(self.db.collection('niveis_risco').document(doc_id), {
                    'geohash': row['geohash'], 'perfil': row['perfil'],
                    'periodo': row['PERIODO'], 'score': float(row['score']),
                    'last_sync': firestore.SERVER_TIMESTAMP
                })
                if (i + 1) % 400 == 0: 
                    batch.commit()
                    batch = self.db.batch()
            batch.commit()

         
            df_bi = df_final[[
                'DATA_OCORRENCIA_BO', 'PERIODO', 'perfil', 'geohash', 
                'LATITUDE', 'LONGITUDE', 'NATUREZA_APURADA', 'PESO_BASE'
            ]].copy()
            df_bi['DATA_ATUALIZACAO'] = momento_inicio
            df_bi.to_csv("safedriver_bi_analytics.csv", index=False, encoding='utf-8-sig')

            # 6. RELATÓRIO TÉCNICO-EXECUTIVO
            tempo_execucao = (datetime.now() - momento_inicio).seconds
            logs = {
                "📥 Ingestão SSP": f"{total_bruto:,} registros brutos",
                "💎 Aproveitamento": f"{eficiencia:.1f}% de dados qualificados",
                "🧹 Limpeza": f"{removidos:,} registros obsoletos removidos do Firestore",
                "☁️ Sincronização ": f"{len(grid):,} células atualizadas no Firestore",
                "📈 Dados Estatisticos": "Base CSV atualizada",
                "⏱️ Latência": f"{tempo_execucao}s",
                "📅 Sincronizado em": data_extenso
            }
            self._notificar_discord(logs, "sucesso")

    def _notificar_discord(self, campos, status="sucesso"):
        webhook_url = os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        cor = 0x2ecc71 if status == "sucesso" else 0x3498db
        embed = {
            "username": "SafeDriver Cloud Engine",
            "embeds": [{
                "title": f"🚀 Pipeline de Dados: {'Sucesso' if status == 'sucesso' else 'Info'}",
                "description": "Relatório técnico de sincronização da malha de risco e base analítica.",
                "color": cor,
                "fields": [{"name": k, "value": v, "inline": True if "Data" not in k else False} for k, v in campos.items()],
                "footer": {"text": "Infraestrutura SafeDriver • Monitoramento de Processos"}
            }]
        }
        requests.post(webhook_url, json=embed)

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
