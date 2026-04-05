import os
import io
import json
import requests
import pandas as pd
import numpy as np
import h3
import shap
import matplotlib.pyplot as plt
import hashlib
from pathlib import Path
from datetime import datetime
from geopy.geocoders import Nominatim
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria",
            "docs": self.raiz / "documentacao"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        ano_inicial = 2022
        self.anos = list(range(ano_inicial, datetime.now().year + 1))
        self.agente = {'User-Agent': 'SafeDriver-Industrial-V14'}
        self.webhook = os.environ.get("DISCORD_WEBHOOK")
        self.manifesto_auditoria = {}

    def calcular_assinatura(self, caminho):
        sha256 = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""):
                sha256.update(bloco)
        return sha256.hexdigest()

    def enviar_discord(self, titulo, mensagem, cor):
        if not self.webhook: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "timestamp": datetime.now().isoformat()
            }]
        }
        requests.post(self.webhook, json=payload, timeout=10)

    def filtrar_planilha_util(self, conteudo):
        excel = pd.ExcelFile(io.BytesIO(conteudo))
        for aba in excel.sheet_names:
            df_temp = excel.parse(aba, nrows=10)
            df_temp.columns = [str(c).upper().strip() for c in df_temp.columns]
            if any(k in df_temp.columns for k in ['LATITUDE', 'NATUREZA_APURADA', 'RUBRICA']):
                return excel.parse(aba)
        return None

    def processar_ia_explicavel(self, df):
        df.columns = [c.lower().strip() for c in df.columns]
        
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_limpo'] = df[col_crime].fillna('').astype(str).upper()
        
        df.loc[df['crime_limpo'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime_limpo'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            local_limpo = df[col_local].fillna('').astype(str).upper()
            df.loc[(local_limpo.str.contains('VIA PÚBLICA')) & (df['crime_limpo'].str.contains('CELULAR|PESSOA')), 'perfil'] = 'Pedestre'
        
        df['peso_severidade'] = df['crime_limpo'].apply(lambda x: 15 if 'ROUBO' in x else 2)

        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        
        if df.empty: raise ValueError("Dados geograficos insuficientes")

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        fato = df.groupby(['h3_index', 'desc_periodo', 'perfil'])['peso_severidade'].sum().reset_index()
        fato['lat'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        fato['lon'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        fato['perfil_idx'] = fato['perfil'].astype('category').cat.codes
        fato['periodo_idx'] = fato['desc_periodo'].astype('category').cat.codes

        X = fato[['lat', 'lon', 'perfil_idx', 'periodo_idx']]
        y = fato['peso_severidade']
        
        modelo_lgb = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        modelo_cat = CatBoostRegressor(iterations=100, silent=True).fit(X, y)

        explicador = shap.TreeExplainer(modelo_lgb)
        valores_shap = explicador.shap_values(X)
        for i, col in enumerate(X.columns):
            fato[f'influencia_{col}'] = valores_shap[:, i]
            
        fato['score_risco'] = (modelo_lgb.predict(X) + modelo_cat.predict(X)) / 2

        caminho_ouro = self.pastas["ouro"] / "base_final_looker.csv"
        fato.to_csv(caminho_ouro, index=False)
        self.manifesto_auditoria["camada_ouro"] = self.calcular_assinatura(caminho_ouro)
        
        return len(fato)

    def iniciar(self):
        try:
            pool = []
            relatorio_op = []
            
            for ano in self.anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    res = requests.get(url, headers=self.agente, timeout=120)
                    if res.status_code == 200:
                        df = self.filtrar_planilha_util(res.content)
                        if df is not None:
                            caminho_bruto = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                            df.to_parquet(caminho_bruto)
                            self.manifesto_auditoria[f"bronze_{ano}"] = self.calcular_assinatura(caminho_bruto)
                            pool.append(df)
                            relatorio_op.append(f"Sucesso {ano}: {len(df)} registros")
                except:
                    relatorio_op.append(f"Falha {ano}: Conexao recusada")

            if not pool: raise Exception("Sem dados disponiveis")

            total = self.processar_ia_explicavel(pd.concat(pool))
            
            with open(self.pastas["auditoria"] / "manifesto.json", "w") as f:
                json.dump(self.manifesto_auditoria, f, indent=4)

            self.enviar_discord("Relatorio Operacional", "\n".join(relatorio_op), 3447003)
            self.enviar_discord("Relatorio Executivo", f"Areas Monitoradas: {total}\nAuditagem: Concluida\nModelos: LGBM + CatBoost", 3066993)

        except Exception as e:
            self.enviar_discord("Relatorio de Erro Critico", str(e), 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
