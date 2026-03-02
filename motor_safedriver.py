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
  PIPLINE DE TRATAMENTO E PREVISÕES SSP 
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

    def _identificar_grupo(self, nat):
        """Categoriza o crime por palavras-chave (Robusto contra erros da SSP)."""
        n = self._limpar_texto(nat)
        if 'LATROCINIO' in n or 'SEQUESTRO' in n: return 'ALERTA', self.pesos['CRITICO']
        if 'ROUBO' in n and 'VEICULO' in n: return 'VEICULAR', self.pesos['ROUBO']
        if 'ROUBO' in n and 'CARGA' in n: return 'VEICULAR', self.pesos['ROUBO']
        if 'ROUBO' in n: return 'RUA', self.pesos['ROUBO']
        if 'FURTO' in n and 'VEICULO' in n: return 'VEICULAR', self.pesos['FURTO']
        if 'FURTO' in n and 'CARGA' in n: return 'VEICULAR', self.pesos['FURTO']
        if 'FURTO' in n: return 'RUA', self.pesos['FURTO']
        return None, 0

    def _notificar_discord(self, titulo, campos, tipo="sucesso"):
        webhook_url = os.environ.get('DISCORD_ERRO') if tipo == "erro" else os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        cor = 0x3498db if tipo == "sucesso" else 0xe74c3c 
        
        embed = {
            "username": "SafeDriver Intelligence",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/1042/1042339.png",
            "embeds": [{
                "title": f"🛡️ {titulo}",
                "description": "Atualizamos o mapa de segurança para os seus trajetos.",
                "color": cor,
                "fields": [{"name": k, "value": str(v), "inline": True} for k, v in campos.items()],
                "footer": {"text": "SafeDriver • Movimente-se com segurança"},
                "thumbnail": {"url": "https://cdn-icons-png.flaticon.com/512/854/854878.png"}
            }]
        }
        try: requests.post(webhook_url, json=embed, timeout=10)
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
        inicio_timer = datetime.now()
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{inicio_timer.year}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            excel = pd.ExcelFile(io.BytesIO(r.content))
            df_total = pd.DataFrame()

            # Varredura global de abas e cabeçalhos
            for sheet in excel.sheet_names:
                df_raw = excel.parse(sheet, header=None)
                h_idx = -1
                for i in range(min(len(df_raw), 50)):
                    line = [self._limpar_texto(str(v)) for v in df_raw.iloc[i].values]
                    if 'NATUREZA_APURADA' in line: h_idx = i; break
                
                if h_idx != -1:
                    df_sheet = excel.parse(sheet, skiprows=h_idx)
                    df_sheet.columns = [self._limpar_texto(str(c)) for c in df_sheet.columns]
                    df_total = pd.concat([df_total, sheet_df], ignore_index=True)

            total_bruto = len(df_total)
        except Exception as e:
            self._notificar_discord("Ops! Algo deu errado", {"Erro": str(e)}, "erro")
            return

  
        df_total['LATITUDE'] = pd.to_numeric(df_total['LATITUDE'], errors='coerce')
        df_total['LONGITUDE'] = pd.to_numeric(df_total['LONGITUDE'], errors='coerce')
        df_clean = df_total[(df_total['LATITUDE'] != 0)].dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        
        res = df_clean['NATUREZA_APURADA'].apply(self._identificar_grupo)
        df_clean['GRUPO'] = [x[0] for x in res]
        df_clean['PESO'] = [x[1] for x in res]
        df_clean['SUB_LIMPO'] = df_clean['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto)
        
        df_filtrado = df_clean[(df_clean['GRUPO'].notna()) & (df_clean['SUB_LIMPO'].isin(self.locais_foco))].copy()
        limpeza_pct = ((total_bruto - len(df_filtrado)) / total_bruto * 100) if total_bruto > 0 else 0

        if not df_filtrado.empty:
        
            def mapear_modais(row):
                g = row['GRUPO']
     
                if g == 'ALERTA': return ['pe', 'onibus', 'bicicleta', 'carro', 'moto']
            
                if g == 'RUA': return ['pe', 'onibus', 'bicicleta']
             
                return ['carro', 'moto']

            df_filtrado['PERFIL_LIST'] = df_filtrado.apply(mapear_modais, axis=1)
            df_final = df_filtrado.explode('PERFIL_LIST').rename(columns={'PERFIL_LIST': 'perfil'})
            
          
            df_final['geohash'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df_final['LATITUDE'], df_final['LONGITUDE'])]

            grid = df_final.groupby(['geohash', 'perfil']).agg({'PESO': 'sum'}).reset_index()
            grid['score'] = (grid['PESO'] * 1.5 + 0.5).clip(0.5, 10.0).round(2)

 
            if self.db:
                batch = self.db.batch()
                for i, row in grid.iterrows():
                    doc_id = f"{row['geohash']}_{row['perfil']}"
                    batch.set(self.db.collection('niveis_risco').document(doc_id), {
                        'geohash': row['geohash'], 'perfil': row['perfil'],
                        'score': float(row['score']), 'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    if (i + 1) % 400 == 0: batch.commit(); batch = self.db.batch()
                batch.commit()

       
            df_final.to_csv("dados_publicos_safedriver.csv", index=False)
            
            stats = df_final['perfil'].value_counts().to_dict()
            campos = {
                "🧹 Filtro": f"{limpeza_pct:.1f}% de dados irrelevantes removidos",
                "🚶 A pé, Bike e Ônibus": f"{stats.get('pe', 0)} alertas ativos",
                "🚗 Carro e Moto": f"{stats.get('carro', 0)} alertas ativos",
                "📍 Locais monitorados": f"{len(grid)} pontos de risco",
                "✅ Sincronização": "App e BI disponiveis",
                "⏱️ Tempo": f"{(datetime.now() - inicio_timer).seconds}s"
            }
            self._notificar_discord("Radar SafeDriver: Trajetos Atualizados", campos, "sucesso")
        else:
            self._notificar_discord("Radar SafeDriver", {"Status": "Sem alertas relevantes nesta quinzena."}, "sucesso")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
