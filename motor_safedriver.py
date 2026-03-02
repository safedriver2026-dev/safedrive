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
PIPELINE DE EXTRAÇÃO, LIMPEZA E PREVISÃO DE DADOS DA SSP
    """
    def __init__(self):
        self.db = self._iniciar_persistencia()
        
        
        self.tipos_alvo = [self._limpar_texto(t) for t in ['Via Pública', 'Rodovia/Estrada']]
        self.subtipos_alvo = [self._limpar_texto(s) for s in [
            'Via Pública', 'Transeunte', 'Acostamento', 'Área de Descanso', 'Balança', 
            'Ciclofaixa', 'De Frente a Residência da Vitíma', 'Feira Livre', 
            'Interior de Veículo de Carga', 'Interior de Veículo de Particular', 
            'Posto de Auxílio', 'Posto de Fiscalização', 'Posto Policial', 'Praça', 
            'Praça de Pedágio', 'Semáforo', 'Túnel/Viaduto/Ponte', 'Veículo em Movimento'
        ]]

        self.pesos = {'FURTO': 1.0, 'ROUBO': 2.5, 'EXTORSAO': 5.0, 'LATROCINIO': 5.0}

    def _limpar_texto(self, texto):
        if not isinstance(texto, str): return str(texto)
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

    def _categorizar_crime(self, nat):
     
        n = self._limpar_texto(nat)
        if 'FURTO' in n and 'VEICULO' in n: return 'FURTO DE VEICULO'
        if 'FURTO' in n and 'CARGA' in n: return 'FURTO DE CARGA'
        if 'ROUBO' in n and 'VEICULO' in n: return 'ROUBO DE VEICULO'
        if 'ROUBO' in n and 'CARGA' in n: return 'ROUBO DE CARGA'
        if 'EXTORSAO' in n and 'SEQUESTRO' in n: return 'EXTORSAO/SEQUESTRO'
        if 'LATROCINIO' in n: return 'LATROCINIO'
        return None

    def _notificar_discord(self, titulo, campos, tipo="sucesso"):
        webhook_url = os.environ.get('DISCORD_ERRO') if tipo == "erro" else os.environ.get('DISCORD_SUCESSO')
        if not webhook_url: return
        cor = 0x3498db if tipo == "sucesso" else 0xe74c3c # Azul para Mobilidade
        
        embed = {
            "username": "SafeDriver AI Intelligence",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/3064/3064155.png",
            "embeds": [{
                "title": f"🧭 {titulo}",
                "color": cor,
                "fields": [{"name": k, "value": str(v), "inline": True} for k, v in campos.items()],
                "footer": {"text": "SafeDriver Intelligence • Mobilidade Segura"},
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
        inicio = datetime.now()
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{inicio.year}.xlsx"
        
        try:
            r = requests.get(url, timeout=120)
            excel = pd.ExcelFile(io.BytesIO(r.content))
            df_bruto = pd.DataFrame()

            for sheet in excel.sheet_names:
                temp = excel.parse(sheet, header=None)
                found = -1
                for i in range(min(len(temp), 50)):
                    if 'NATUREZA_APURADA' in [self._limpar_texto(str(v)) for v in temp.iloc[i].values]:
                        found = i; break
                if found != -1:
                    sheet_df = excel.parse(sheet, skiprows=found)
                    sheet_df.columns = [self._limpar_texto(str(c)) for c in sheet_df.columns]
                    df_bruto = pd.concat([df_bruto, sheet_df], ignore_index=True)

            total_original = len(df_bruto)
        except Exception as e:
            self._notificar_discord("Falha na Telemetria", {"Erro": str(e)}, "erro")
            return

        # 1. Limpeza de Dados Sujos
        df_bruto['LATITUDE'] = pd.to_numeric(df_bruto['LATITUDE'], errors='coerce')
        df_bruto['LONGITUDE'] = pd.to_numeric(df_bruto['LONGITUDE'], errors='coerce')
        df_limpo = df_bruto[(df_bruto['LATITUDE'].notna()) & (df_bruto['LATITUDE'] != 0)].copy()
        
        # 2. Categorizacao e Filtros de Mobilidade
        df_limpo['CATEGORIA'] = df_limpo['NATUREZA_APURADA'].apply(self._categorizar_crime)
        df_limpo['SUB_LIMPO'] = df_limpo['DESCR_SUBTIPOLOCAL'].apply(self._limpar_texto)
        
        df_filtrado = df_limpo[
            (df_limpo['CATEGORIA'].notna()) & 
            (df_limpo['SUB_LIMPO'].isin(self.subtipos_alvo))
        ].copy()

        # Calculo de Eliminacao de Ruido 
        eliminados = total_original - len(df_filtrado)
        pct_limpeza = (eliminados / total_original * 100) if total_original > 0 else 0

        if not df_filtrado.empty:
        
            def classificar(row):
                cat, sub = row['CATEGORIA'], row['SUB_LIMPO']
                peso = self.pesos['FURTO'] if 'FURTO' in cat else self.pesos['ROUBO']
                if 'LATROCINIO' in cat or 'EXTORSAO' in cat: peso = self.pesos['LATROCINIO']
                perfil = 'pedestre' if any(x in sub for x in ['TRANSEUNTE', 'PRACA', 'CICLOFAIXA']) else 'motorista'
                return pd.Series([perfil, peso])

            df_filtrado[['PERFIL', 'PESO_LEGAL']] = df_filtrado.apply(classificar, axis=1)
            df_filtrado['GEOHASH'] = [gh.encode(la, lo, precision=6) for la, lo in zip(df_filtrado['LATITUDE'], df_filtrado['LONGITUDE'])]

            grid = df_filtrado.groupby(['GEOHASH', 'PERFIL']).agg({'PESO_LEGAL': 'sum'}).reset_index()
            grid['SCORE'] = (grid['PESO_LEGAL'] * 1.5 + 0.5).clip(0.5, 10.0).round(2)

            if self.db:
                batch = self.db.batch()
                for i, row in grid.iterrows():
                    batch.set(self.db.collection('niveis_risco').document(f"{row['GEOHASH']}_{row['PERFIL']}"), {
                        'geohash': row['GEOHASH'], 'perfil': row['PERFIL'],
                        'score': float(row['SCORE']), 'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    if (i + 1) % 400 == 0: batch.commit(); batch = self.db.batch()
                batch.commit()

            # Relatorio de Saude do Processo
            resumo_counts = df_filtrado['CATEGORIA'].value_counts().head(3).to_dict()
            resumo_str = "\n".join([f"🔹 {k}: {v}" for k, v in resumo_counts.items()])
            
            campos = {
                "🛰️ Malha Analisada": f"{total_original} registros",
                "🧹 Limpeza de Ruido": f"Eliminou {pct_limpeza:.1f}% de dados irrelevantes",
                "📍 Zonas Detectadas": f"{len(grid)} quadrantes",
                "🛡️ Score Medio": f"{grid['SCORE'].mean():.2f}",
                "🚨 Principais Alertas": resumo_str,
                "⏱️ Latencia": f"{(datetime.now() - inicio).seconds}s"
            }
            self._notificar_discord("Dashboard de Inteligencia SafeDriver", campos, "sucesso")
            
            cols_pbi = ['NATUREZA_APURADA', 'NOME_MUNICIPIO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 'LATITUDE', 'LONGITUDE', 'PERFIL', 'SCORE']
            df_filtrado.rename(columns={'SCORE_RISCO': 'SCORE'}, errors='ignore')
            df_filtrado.to_csv("dados_publicos_safedriver.csv", index=False)
        else:
            self._notificar_discord("Alerta de Vacancia", {"Status": "Nenhum crime relevante nesta malha", "Limpeza": f"{pct_limpeza:.1f}%"}, "erro")

if __name__ == "__main__":
    SafeDriverEngine().executar_pipeline()
