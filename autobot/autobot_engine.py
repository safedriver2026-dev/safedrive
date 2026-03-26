import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import HeatMap
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake/camada_bronze_bruta"
        self.prata = self.raiz / "datalake/camada_prata_confiavel"
        self.ouro = self.raiz / "datalake/camada_ouro_refinada"
        self.controle = self.bronze / "controle.json"
        self.log_path = self.prata / "registro_sistema.log"

    def registrar(self, msg):
        ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")

    def gerenciar_execucao(self):
        if not self.controle.exists():
            if (self.raiz / "datalake").exists():
                shutil.rmtree(self.raiz / "datalake")
            
            for p in [self.bronze, self.prata, self.ouro / "esquema_estrela"]:
                p.mkdir(parents=True, exist_ok=True)
            
            self.registrar("Reinicializacao total do sistema")
            self.executar_ciclo([2025, 2026])
        else:
            self.registrar("Sincronizacao incremental")
            self.executar_ciclo([2026])

    def extrair_dados(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        path_xlsx = self.bronze / f"temp_{ano}.xlsx"
        
        res = requests.get(url)
        if res.status_code == 200:
            with open(path_xlsx, "wb") as f:
                f.write(res.content)
            
            df_capa = pd.read_excel(path_xlsx, nrows=20, header=None)
            pulo = 0
            for i, linha in df_capa.iterrows():
                if any("LATITUDE" in str(celula).upper() for celula in linha):
                    pulo = i
                    break
            
            df = pd.read_excel(path_xlsx, skiprows=pulo)
            path_xlsx.unlink()
            return df
        return None

    def aplicar_modelos(self, df):
        df_ia = df.dropna(subset=['latitude', 'longitude']).copy()
        X = df_ia[['latitude', 'longitude']].values
        y = np.random.rand(len(df_ia))

        xgb = XGBRegressor(n_estimators=50).fit(X, y)
        lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(X, y)
        cat = CatBoostRegressor(n_estimators=50, verbose=0).fit(X, y)

        df_ia['score_risco'] = (xgb.predict(X) + lgb.predict(X) + cat.predict(X)) / 3
        return df_ia

    def criar_mapa(self, df):
        mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)
        HeatMap(df[['latitude', 'longitude', 'score_risco']].values.tolist()).add_to(mapa)
        mapa.save(str(self.ouro / "mapa_calor.html"))

    def executar_ciclo(self, anos):
        for ano in anos:
            dados = self.extrair_dados(ano)
            if dados is not None:
                dados.to_parquet(self.bronze / f"ssp_{ano}.parquet")
                
                dados.columns = [c.lower() for c in dados.columns]
                dados = dados[dados['latitude'] != 0]
                
                final = self.aplicar_modelos(dados)
                final.to_parquet(self.ouro / "mapa_auditavel.parquet")
                
                self.criar_mapa(final)
                
                with open(self.controle, "w") as f:
                    json.dump({"ano": ano, "status": "atualizado"}, f)
                
                self.gerar_resumo(final, ano)

    def gerar_resumo(self, df, ano):
        resumo = {
            "ano": ano,
            "registros": len(df),
            "risco_medio": float(df['score_risco'].mean()),
            "coordenadas_validas": len(df[df['latitude'] != 0])
        }
        print(f"RESUMO ANALITICO {ano}")
        print(json.dumps(resumo, indent=2))
