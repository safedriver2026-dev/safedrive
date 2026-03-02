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
PIPILINE DADOS SSP - EXTRAÇÃO, LIMPEZA E PREVISÃO
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
      
        self.subtipos_alvo = [self._limpar_texto(s) for s in [
            'Via Pública', 'Transeunte', 'Acostamento', 'Área de Descanso', 'Balança', 
            'Ciclofaixa', 'De Frente a Residência da Vitíma', 'Feira Livre', 
            'Interior de Veículo de Carga', 'Interior de Veículo de Particular', 
            'Posto de Auxílio', 'Posto de Fiscalização', 'Posto Policial', 'Praça', 
            'Praça de Pedágio', 'Semáforo', 'Túnel/Viaduto/Ponte', 'Veículo em Movimento'
        ]]

    def _limpar_texto(self, texto):
        if not isinstance(texto, str): return str(texto)
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

    def _categorizar_crime(self, nat):
        n = self._limpar_texto(nat)
        if 'FURTO' in n and 'VEICULO' in n: return 'FURTO VEICULO', 1.0
        if 'FURTO' in n and 'CARGA' in n: return 'FURTO CARGA', 1.2
        if 'ROUBO' in n and 'VEICULO' in n: return 'ROUBO VEICULO', 3.0
        if 'ROUBO' in n and 'CARGA' in n: return 'ROUBO CARGA', 3.5
        if 'EXTORSAO' in n and 'SEQUESTRO' in n: return 'SEQUESTRO', 5.0
        if 'LATROCINIO' in n: return 'LATROCINIO', 5.0
        if 'ROUBO' in n and 'OUTROS' in n: return 'ROUBO RUA', 2.0
        if 'FURTO' in n and 'OUTROS' in n: return 'FURTO RUA', 0.8
        return None, 0

    def _notificar_discord(self, titulo, campos, tipo="sucesso"):
        webhook_url = os.environ.get('DISCORD_ERRO') if tipo == "erro" else os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        cor = 0x3498db if tipo == "sucesso" else 0xe74c3c 
        
        embed = {
            "username": "SafeDriver Navigation AI",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/3064/3064155.png",
            "embeds": [{
                "title": f"🧭 {titulo}",
                "description": "Sincronização de Telemetria Multi-Modal (Padrão Google Maps).",
                "color": cor,
                "fields": [{"name": k, "value": str(v), "inline": True} for k, v in campos.items()],
                "footer": {"text": "SafeDriver Core • Inteligência em Mobilidade"},
                "thumbnail": {"url": "https://cdn-icons-png.flaticon.com/512/235/235861.png"}
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
        inicio = datetime.now()
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{inicio.year}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            excel = pd.ExcelFile(io.BytesIO(r.content))
            df_total = pd.DataFrame()

            for sheet in excel.sheet_names:
                df_raw = excel.parse(sheet, header=None)
                h_idx = -1
                for i in range(min(len(df_raw), 50)):
                    if 'NATUREZA_APURADA' in [self._limpar_texto(str(v)) for v in df_raw.iloc[i].values]:
                        h_idx = i; break
                if h_idx != -1:
                    sheet_df = excel.parse(sheet, skiprows=h_idx)
                    sheet_df.columns = [self._limpar_texto(str(c)) for c in sheet_df.columns]
                    df_total = pd.concat([df_total, sheet_df], ignore_index=True)

            total_bruto = len(df_total)
        except Exception as e:
            self._notificar_discord("Falha de Conexão SSP", {"Erro": str(e)}, "erro")
            return

      
        df_total['LATITUDE'] = pd.to_numeric(df_total['LATITUDE'], errors='coerce')
        df_total['LONGITUDE'] = pd.to_numeric(df_total['LONGITUDE'], errors='coerce')
        df_clean = df_total[(df_total['LATITUDE'] != 0) & (df_total['LONGITUDE'] != 0)].dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        
        res = df_clean['NATUREZA_APURADA'].apply(self._categorizar_crime)
        df_clean['CATEGORIA'] = [x[0] for x in res]
        df_clean['PESO'] = [x[1] for x in res]
        df_clean['SUB_LIMPO'] = df_clean['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto)
        
        df_filtrado = df_clean[(df_clean['CATEGORIA'].notna()) & (df_clean['SUB_LIMPO'].isin(self.subtipos_alvo))].copy()
        limpeza_pct = ((total_bruto - len(df_filtrado)) / total_bruto * 100) if total_bruto > 0 else 0

        if not df_filtrado.empty:
         
            def mapear_modais(row):
                cat, sub = row['CATEGORIA'], row['SUB_LIMPO']
                
          
                if 'RUA' in cat or sub == 'TRANSEUNTE':
                    return ['pedestre', 'onibus', 'bicicleta']
                
            
                return ['carro', 'motociclista']

            df_filtrado['PERFIL_LIST'] = df_filtrado.apply(mapear_modais, axis=1)
            df_final = df_filtrado.explode('PERFIL_LIST').rename(columns={'PERFIL_LIST': 'PERFIL'})
            
            df_final['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df_final['LATITUDE'], df_final['LONGITUDE'])]

            grid = df_final.groupby(['GEOHASH', 'PERFIL']).agg({'PESO': 'sum'}).reset_index()
            grid['SCORE'] = (grid['PESO'] * 1.5 + 0.5).clip(0.5, 10.0).round(2)

            if self.db:
                batch = self.db.batch()
                for i, row in grid.iterrows():
                    batch.set(self.db.collection('niveis_risco').document(f"{row['GEOHASH']}_{row['PERFIL']}"), {
                        'geohash': row['GEOHASH'], 'perfil': row['PERFIL'],
                        'score': float(row['SCORE']), 'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    if (i + 1) % 400 == 0: batch.commit(); batch = self.db.batch()
                batch.commit()

            df_final.to_csv("dados_publicos_safedriver.csv", index=False)
            
            stats = df_final['PERFIL'].value_counts().to_dict()
            campos = {
                "🧹 Limpeza": f"Eliminou {limpeza_pct:.1f}% de dados irrelevantes",
                "🚶 Pedestre/Ônibus/Bike": f"{stats.get('pedestre', 0)} alertas",
                "🚗 Carro/Moto": f"{stats.get('carro', 0)} alertas",
                "📍 Zonas Totais": f"{len(grid)} pontos",
                "🚀 Status dos dados": "✅ Sincronizado",
                "⏱️ Tempo": f"{(datetime.now() - inicio).seconds}s"
            }
            self._notificar_discord("Dashboard Multi-Modal SafeDriver", campos, "sucesso")
        else:
            self._notificar_discord("Aviso", {"Status": "Sem incidentes na malha atual."}, "erro")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
