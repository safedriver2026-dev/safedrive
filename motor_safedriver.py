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
    Safedrive - ETL e Processamento de Geohash
    """
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

    def _higienizar_ambiente_cloud(self):
        """ Remove registros obsoletos para garantir a integridade da malha atual """
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
        cronometro_inicio = datetime.now()
        data_referencia = cronometro_inicio.strftime('%d/%m/%Y às %H:%M')
        
        # INGESTÃO E MAPEAMENTO DE CABEÇALHO
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{cronometro_inicio.year}.xlsx"
        try:
            requisicao = requests.get(url, timeout=120)
            excel = pd.ExcelFile(io.BytesIO(requisicao.content))
            df_bruto = pd.DataFrame()
            for aba in excel.sheet_names:
                df_temp = excel.parse(aba, header=None)
                indice_cabecalho = 0
                for i, linha in df_temp.head(25).iterrows():
                    if 'LATITUDE' in [str(celula).upper() for celula in linha.values]:
                        indice_cabecalho = i
                        break
                df_corrigido = excel.parse(aba, skiprows=indice_cabecalho)
                df_corrigido.columns = [self._limpar_texto(str(c)) for c in df_corrigido.columns]
                df_bruto = pd.concat([df_bruto, df_corrigido], ignore_index=True)
            total_registros_brutos = len(df_bruto)
        except Exception as e:
            self._enviar_relatorio_consolidado({"Erro de Processamento": str(e)}, "erro")
            return

        # SANEAMENTO E QUALIFICAÇÃO
        df_bruto['LATITUDE'] = pd.to_numeric(df_bruto['LATITUDE'], errors='coerce')
        df_bruto['LONGITUDE'] = pd.to_numeric(df_bruto['LONGITUDE'], errors='coerce')
        df_geo = df_bruto.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        df_geo = df_geo[df_geo['LATITUDE'] != 0]

        def filtrar_natureza(nat):
            n = self._limpar_texto(nat)
            if any(x in n for x in ['LATROCINIO', 'SEQUESTRO']): return 'ALERTA'
            if 'ROUBO' in n or 'FURTO' in n: return 'QUALIFICADO'
            return None

        df_geo['QUALIFICACAO'] = df_geo['NATUREZA_APURADA'].apply(filtrar_natureza)
        df_filtrado = df_geo[df_geo['QUALIFICACAO'].notna()].copy()
        total_qualificados = len(df_filtrado)
        aproveitamento_final = (total_qualificados / total_registros_brutos) * 100 if total_registros_brutos > 0 else 0

        # VERIFICAÇÃO DE ATUALIZAÇÕES DISPONÍVEIS
        if total_qualificados == 0:
            self._enviar_relatorio_consolidado({
                "Sincronização": "Finalizada",
                "Observação": "Não foram detectados novos registros qualificados. Malha atual mantida.",
                "Data": data_referencia
            }, "neutro")
            return

        # PROCESSAMENTO DE PERFIS E INDEXAÇÃO GEOESPACIAL
        df_filtrado['PERIODO'] = df_filtrado['HORA_OCORRENCIA_BO'].apply(self._definir_periodo)
        df_filtrado['perfil'] = df_filtrado['QUALIFICACAO'].apply(lambda x: ['Pedestre', 'Ciclista', 'Motorista'] if x == 'ALERTA' else ['Pedestre', 'Ciclista'])
        df_final = df_filtrado.explode('perfil')
        df_final['geohash'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df_final['LATITUDE'], df_final['LONGITUDE'])]

        # CONSOLIDAÇÃO NO FIRESTORE E EXPORTAÇÃO CSV
        if self.db:
            docs_higienizados = self._higienizar_ambiente_cloud()
            
            # Agrupamento de Scores para o App
            resumo_grid = df_final.groupby(['geohash', 'perfil', 'PERIODO']).size().reset_index(name='frequencia')
            resumo_grid['score'] = (resumo_grid['frequencia'] * 2.3).clip(0.5, 10.0).round(2)

            batch = self.db.batch()
            for i, row in resumo_grid.iterrows():
                doc_id = f"{row['geohash']}_{row['perfil']}_{row['PERIODO']}"
                batch.set(self.db.collection('niveis_risco').document(doc_id), {
                    'geohash': row['geohash'], 'perfil': row['perfil'],
                    'periodo': row['PERIODO'], 'score': float(row['score']),
                    'atualizado_em': firestore.SERVER_TIMESTAMP
                })
                if (i + 1) % 400 == 0: batch.commit(); batch = self.db.batch()
            batch.commit()

           
            df_final['TIMESTAMP_ATUALIZACAO'] = cronometro_inicio
            df_final.to_csv("analise_consolidada_safedriver.csv", index=False, encoding='utf-8-sig')

            # RELATÓRIO EXECUTIVO AUTOBOT
            distribuicao = df_final['perfil'].value_counts(normalize=True) * 100
            logs = {
                "📂 Ingestão SSP": f"{total_registros_brutos:,} registros",
                "💎 Aproveitamento": f"{total_qualificados:,} úteis ({aproveitamento_final:.1f}%)",
                "🧹 Saneamento Cloud": f"{docs_higienizados:,} removidos",
                "🚶 Pedestres": f"{distribuicao.get('Pedestre', 0):.1f}%",
                "🚲 Ciclistas": f"{distribuicao.get('Ciclista', 0):.1f}%",
                "🚗 Motoristas": f"{distribuicao.get('Motorista', 0):.1f}%",
                "🚀 Status Cloud": "Firestore & BI Atualizados",
                "⏱️ Tempo de Resposta": f"{(datetime.now() - cronometro_inicio).seconds}s",
                "📅 Concluído em": data_exec
            }
            self._enviar_relatorio_consolidado(logs, "sucesso")

    def _enviar_relatorio_consolidado(self, campos, status="sucesso"):
        webhook_url = os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        cores = {"sucesso": 0x27ae60, "neutro": 0x3498db, "erro": 0xe74c3c}
        embed = {
            "username": "SafeDriver Autobot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2082/2082805.png",
            "embeds": [{
                "title": f"🛡️ Relatório de Operações – {'Sucesso' if status == 'sucesso' else 'Informativo'}",
                "description": "Atualização da malha de risco e base analítica de mobilidade efetuada.",
                "color": cores.get(status),
                "fields": [{"name": k, "value": v, "inline": True if "%" in v or "s" in v else False} for k, v in campos.items()],
                "footer": {"text": "Autobot Infrastructure • Monitoramento Consolidado"}
            }]
        }
        requests.post(webhook_url, json=embed)

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()import pandas as pd
