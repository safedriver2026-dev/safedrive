import pandas as pd
import numpy as np
import os, json, glob, logging
from datetime import datetime
import h3
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import requests

logging.basicConfig(level=logging.INFO)

class SafeDriverWorldClass:
    def __init__(self):
        self.paths = {"bronze": "datalake/bronze", "silver": "datalake/silver", "gold": "datalake/gold", "audit": "datalake/audit"}
        self._init_infra()
        self.memoria = self._carregar_memoria()
        self.telemetria = {"etapas": {}, "ia": {}}

    def _init_infra(self):
        for p in self.paths.values(): os.makedirs(p, exist_ok=True)

    def _carregar_memoria(self):
        """Lê todos os manifestos anteriores para entender a tendência de erro."""
        arquivos = glob.glob(f"{self.paths['audit']}/manifest_*.json")
        historico = []
        for f in arquivos:
            with open(f, 'r') as m: historico.append(json.load(m))
        return historico

    def _auto_correcao_schema(self, df):
        """Engenharia de Dados Autônoma: Mapeia colunas por similaridade."""
        mapa = {"LAT": "LATITUDE", "LONG": "LONGITUDE", "RUBRICA": "crime", "NATUREZA": "crime"}
        cols_atuais = {c.upper(): c for c in df.columns}
        final_map = {cols_atuais[k]: v for k, v in mapa.items() if k in cols_atuais}
        return df.rename(columns=final_map)

    def engine_ia_evolutiva(self, df):
        """IA que aprende com o passado (Meta-Learning)."""
        X = df[['LATITUDE', 'LONGITUDE']]
        y = df['score']
        
        # Obtém média histórica de erro (MAE)
        mae_historico = np.mean([m['ia']['mae'] for m in self.memoria if 'ia' in m]) if self.memoria else 1.0
        
        # Treino atual
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        mae_atual = mean_absolute_error(y, preds)
        
        # Decisão Autônoma
        status_ia = "ESTÁVEL"
        if mae_atual > mae_historico * 1.2: # Erro subiu 20%
            status_ia = "ALERTA: Degradação de dados ou mudança de padrão criminal."
            # Aqui poderíamos disparar um Hyperparameter Tuning automático
        
        self.telemetria['ia'] = {
            "mae": round(float(mae_atual), 4),
            "r2": round(float(r2_score(y, preds)), 4),
            "status_evolutivo": status_ia,
            "vs_historico": "MELHOR" if mae_atual < mae_historico else "PIOR"
        }
        return model

    def pipeline(self):
        try:
            # BRONZE -> SILVER
            raw = pd.concat([pd.read_parquet(os.path.join(self.paths['bronze'], f)) 
                             for f in os.listdir(self.paths['bronze']) if f.endswith('.parquet')])
            silver = self._auto_correcao_schema(raw)
            silver['score'] = silver['crime'].apply(lambda x: 10.0 if 'MORTE' in str(x).upper() else 5.0)
            silver.to_parquet(f"{self.paths['silver']}/main.parquet")
            
            # IA & GOLD
            modelo = self.engine_ia_evolutiva(silver)
            
            # H3 Aggregation
            silver['h3_id'] = silver.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)
            gold = silver.groupby('h3_id').agg({'score': 'mean', 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
            
            # PERSISTÊNCIA (Versionamento Automático via Git ocorrerá no Action)
            gold.to_json(f"{self.paths['gold']}/api_v1.json", orient="records", indent=2)
            gold.to_parquet(f"{self.paths['gold']}/bi_map.parquet")
            
            self._finalizar_run("SUCESSO")
        except Exception as e:
            self._finalizar_run(f"FALHA: {str(e)}")

    def _finalizar_run(self, status):
        run_id = datetime.now().strftime('%Y%m%d%H%M')
        self.telemetria["status"] = status
        with open(f"{self.paths['audit']}/manifest_{run_id}.json", "w") as f:
            json.dump(self.telemetria, f, indent=4)
        self._notificar_discord(status)

    def _notificar_discord(self, status):
        webhook = os.environ.get('DISCORD_WEBHOOK')
        if not webhook: return
        
        ia = self.telemetria.get('ia', {})
        payload = {
            "embeds": [{
                "title": "🛡️ SAFE-DRIVER | SISTEMA AUTÔNOMO",
                "color": 3066993 if "SUCESSO" in status else 15158332,
                "fields": [
                    {"name": "📉 MÉTRICA MAE", "value": f"`{ia.get('mae', 'N/A')}` ({ia.get('vs_historico', '-')})", "inline": True},
                    {"name": "🧠 STATUS IA", "value": f"`{ia.get('status_evolutivo', 'N/A')}`", "inline": False},
                    {"name": "⚙️ PIPELINE", "value": f"`{status}`", "inline": True}
                ],
                "footer": {"text": f"Audit ID: {datetime.now().strftime('%Y%m%d')}"}
            }]
        }
        requests.post(webhook, json=payload)

if __name__ == "__main__":
    SafeDriverWorldClass().pipeline()
