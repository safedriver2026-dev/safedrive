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
PIPELINE DE EXTRAÇÃO, LIMPEZA, TRANSFORMAÇÃO E PREVISÃO DE DADOS DA SSP
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        self.naturezas_alvo = ['FURTO DE VEICULO', 'FURTO DE CARGA', 'EXTORSAO MEDIANTE A SEQUESTRO', 'ROUBO DE VEICULO', 'ROUBO DE CARGA', 'LATROCINIO']
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
        if not isinstance(texto, str): return str(texto)
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

    def _notificar_discord(self, titulo, campos, tipo="sucesso"):
        """Envia um Embed profissional para o Discord."""
        webhook_url = os.environ.get('DISCORD_ERRO') if tipo == "erro" else os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return

  
        cor = 0x2ecc71 if tipo == "sucesso" else 0xe74c3c
        emoji_titulo = "✅" if tipo == "sucesso" else "🚨"

        embed = {
            "username": "SafeDriver Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/1183/1183210.png",
            "embeds": [{
                "title": f"{emoji_titulo} {titulo}",
                "color": cor,
                "fields": [{"name": k, "value": str(v), "inline": True} for k, v in campos.items()],
                "footer": {"text": f"Executado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"},
                "thumbnail": {"url": "https://cdn-icons-png.flaticon.com/512/2092/2092663.png"}
            }]
        }

        try:
            requests.post(webhook_url, json=embed, timeout=10)
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
        inicio_exec = datetime.now()
        print(f"[{inicio_exec}] Iniciando Pipeline...")
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{inicio_exec.year}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            excel_file = pd.ExcelFile(io.BytesIO(r.content))
            df_final = pd.DataFrame()

            # Varredura em todas as abas
            for sheet_name in excel_file.sheet_names:
                df_raw = excel_file.parse(sheet_name, header=None)
                found_row = -1
                for i in range(min(len(df_raw), 50)):
                    linha_limpa = [self._limpar_texto(str(val)) for val in df_raw.iloc[i].values]
                    if 'NATUREZA_APURADA' in linha_limpa:
                        found_row = i
                        break
                
                if found_row != -1:
                    df_sheet = excel_file.parse(sheet_name, skiprows=found_row)
                    df_sheet.columns = [self._limpar_texto(str(c)) for c in df_sheet.columns]
                    df_final = pd.concat([df_final, df_sheet], ignore_index=True)

            total_bruto = len(df_final)

        except Exception as e:
            self._notificar_discord("Falha Critica na Extracao", {"Erro": str(e), "Status": "Abortado"}, "erro")
            return

        if df_final.empty:
            self._notificar_discord("Base Vazia", {"Status": "Sem dados novos na SSP", "Acao": "Nenhuma alteracao feita"}, "sucesso")
            return

       
        df_final['LATITUDE'] = pd.to_numeric(df_final['LATITUDE'], errors='coerce')
        df_final['LONGITUDE'] = pd.to_numeric(df_final['LONGITUDE'], errors='coerce')
        df_final = df_final.dropna(subset=['LATITUDE', 'LONGITUDE'])

        df_final['NAT_LIMPA'] = df_final['NATUREZA_APURADA'].apply(self._limpar_texto)
        df_final['SUB_LIMPO'] = df_final['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto) if 'DESCR_SUBTIPOLOCAL' in df_final.columns else ""

        mask = (df_final['NAT_LIMPA'].isin(self.naturezas_alvo)) & (df_final['SUB_LIMPO'].isin(self.subtipos_validos))
        df_filtrado = df_final[mask].copy()

        total_util = len(df_filtrado)
        taxa_saude = (total_util / total_bruto * 100) if total_bruto > 0 else 0

        if not df_filtrado.empty:
          
            def calcular(row):
                nat, sub = row['NAT_LIMPA'], row['SUB_LIMPO']
                peso = self.pesos_legais['FURTO']
                if 'ROUBO' in nat: peso = self.pesos_legais['ROUBO']
                if 'EXTORSAO' in nat or 'LATROCINIO' in nat: peso = self.pesos_legais['LATROCINIO']
                perfil = 'pedestre' if any(x in sub for x in ['TRANSEUNTE', 'PRACA', 'CICLOFAIXA']) else 'motorista'
                return pd.Series([perfil, peso])

            df_filtrado[['PERFIL', 'PESO_LEGAL']] = df_filtrado.apply(calcular, axis=1)
            df_filtrado['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df_filtrado['LATITUDE'], df_filtrado['LONGITUDE'])]

            grid = df_filtrado.groupby(['GEOHASH', 'PERFIL']).agg({'PESO_LEGAL': 'sum'}).reset_index()
            grid['SCORE_RISCO'] = (grid['PESO_LEGAL'] * 0.8 + 0.5).clip(0.5, 10.0).round(2)

            # Firebase
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

            # Relatorio Final
            tempo = (datetime.now() - inicio_exec).seconds
            campos_relatorio = {
                "🏥 Saude": "Excelente" if taxa_saude > 2 else "Baixa Retencao",
                "📊 Brutos": f"{total_bruto} linhas",
                "🔍 Filtrados": f"{total_util} crimes",
                "🚀 Previsoes": f"{len(grid)} zonas",
                "📈 Eficiencia": f"{taxa_saude:.1f}%",
                "⏱️ Tempo": f"{tempo}s"
            }
            self._notificar_discord("Relatorio SafeDriver Concluido", campos_relatorio, "sucesso")
        else:
            self._notificar_discord("Alerta de Processamento", {"Aviso": "Filtros resultaram em 0 crimes", "Bruto": total_bruto}, "erro")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
