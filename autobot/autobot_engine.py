import pandas as pd
import numpy as np
import os, json, glob, logging, hashlib
from datetime import datetime
import h3
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import requests

# Configuração de Logs Profissionais
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SafeDriver-Motor")

class MotorEvolutivoSafeDriver:
    def __init__(self):
        self.caminhos = {
            "bronze": "datalake/bronze",
            "silver": "datalake/silver",
            "gold": "datalake/gold",
            "audit": "datalake/audit"
        }
        self._inicializar_infraestrutura()
        self.historico = self._carregar_memoria_auditavel()
        self.telemetria = {"status": "INICIADO", "ia_performance": {}, "etapas": []}

    def _inicializar_infraestrutura(self):
        for pasta in self.caminhos.values():
            os.makedirs(pasta, exist_ok=True)

    def _carregar_memoria_auditavel(self):
        """Lê manifestos anteriores para a IA aprender com a performance do passado."""
        arquivos = glob.glob(f"{self.caminhos['audit']}/manifesto_*.json")
        memoria = []
        for f in arquivos:
            with open(f, 'r') as m: memoria.append(json.load(m))
        return memoria

    def _auto_correcao_esquema(self, df):
        """Sistema de Autocorreção: Mapeia colunas por similaridade funcional."""
        mapeamento = {
            "LATITUDE": ["LATITUDE", "LAT", "COORD_X", "LAT_Y"],
            "LONGITUDE": ["LONGITUDE", "LONG", "COORD_Y", "LON_X"],
            "CRIME": ["RUBRICA", "NATUREZA", "DESCR_CRIME", "TIPO_OCORRENCIA"]
        }
        colunas_reais = {c.upper(): c for c in df.columns}
        mapa_final = {}
        for alvo, variantes in mapeamento.items():
            for v in variantes:
                if v in colunas_reais:
                    mapa_final[colunas_reais[v]] = alvo
                    break
        return df.rename(columns=mapa_final)

    def treinar_ia_com_meta_learning(self, df):
        """IA que analisa o erro histórico para decidir o nível de otimização."""
        X = df[['LATITUDE', 'LONGITUDE']]
        y = df['score']
        
        # Cálculo da média histórica de erro (MAE)
        erros_passados = [m['ia_performance']['mae'] for m in self.historico if 'ia_performance' in m]
        mae_medio = np.mean(erros_passados) if erros_passados else 1.0
        
        # Treinamento do Modelo Atual
        modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        modelo.fit(X, y)
        previsoes = modelo.predict(X)
        mae_atual = mean_absolute_error(y, previsoes)
        
        # Decisão Evolutiva
        tendencia = "ESTÁVEL"
        if mae_atual > mae_medio * 1.15:
            tendencia = "ALERTA: Degradação de Padrão detectada."
        elif mae_atual < mae_medio:
            tendencia = "EVOLUÇÃO: Modelo mais preciso que a média histórica."

        self.telemetria["ia_performance"] = {
            "mae": round(float(mae_atual), 4),
            "r2": round(float(r2_score(y, previsoes)), 4),
            "tendencia": tendencia
        }
        return modelo

    def executar_pipeline(self):
        try:
            # 1. CAMADA BRONZE -> SILVER (Processamento de Dados Brutos)
            arquivos_bronze = [f for f in os.listdir(self.caminhos["bronze"]) if f.endswith('.parquet')]
            if not arquivos_bronze: raise Exception("Cofre Bronze vazio.")
            
            df_bruto = pd.concat([pd.read_parquet(os.path.join(self.caminhos["bronze"], f)) for f in arquivos_bronze])
            df_silver = self._auto_correcao_esquema(df_bruto)
            
            # Limpeza e Scoring (Regra de Negócio de Risco)
            df_silver['LATITUDE'] = pd.to_numeric(df_silver['LATITUDE'], errors='coerce')
            df_silver['score'] = df_silver['CRIME'].apply(lambda x: 10.0 if 'MORTE' in str(x).upper() else 4.5)
            df_silver = df_silver.dropna(subset=['LATITUDE', 'LONGITUDE'])
            
            df_silver.to_parquet(f"{self.caminhos['silver']}/base_confiavel.parquet", index=False)
            
            # 2. CAMADA SILVER -> GOLD (Inteligência e API)
            modelo = self.treinar_ia_com_meta_learning(df_silver)
            
            # Indexação Espacial H3 (Resolução 8)
            df_silver['h3_id'] = df_silver.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)
            
            df_gold = df_silver.groupby('h3_id').agg({
                'score': 'mean',
                'LATITUDE': 'mean',
                'LONGITUDE': 'mean'
            }).reset_index()
            
            # Persistência para Consumo (Power BI e API Estática)
            df_gold.to_parquet(f"{self.caminhos['gold']}/risco_geospatial.parquet", index=False)
            df_gold.to_json(f"{self.caminhos['gold']}/api_risco_v1.json", orient="records", indent=2)
            
            self.telemetria["status"] = "SUCESSO"
            self._gerar_auditoria()
            self._notificar_discord()

        except Exception as e:
            logger.error(f"Falha Crítica: {e}")
            self.telemetria["status"] = f"ERRO: {str(e)}"
            self._notificar_discord()

    def _gerar_auditoria(self):
        id_execucao = datetime.now().strftime('%Y%m%d_%H%M')
        with open(f"{self.caminhos['audit']}/manifesto_{id_execucao}.json", "w") as f:
            json.dump(self.telemetria, f, indent=4)

    def _notificar_discord(self):
        webhook = os.environ.get('DISCORD_WEBHOOK')
        if not webhook: return
        
        ia = self.telemetria.get('ia_performance', {})
        cor = 3066993 if "SUCESSO" in self.telemetria["status"] else 15158332
        
        payload = {
            "embeds": [{
                "title": "🛡️ SAFE-DRIVER | DATA LAKEHOUSE",
                "color": cor,
                "fields": [
                    {"name": "📊 Status do Pipeline", "value": f"`{self.telemetria['status']}`", "inline": False},
                    {"name": "📉 Erro Médio (MAE)", "value": f"`{ia.get('mae', 'N/A')}`", "inline": True},
                    {"name": "🧠 Inteligência", "value": f"`{ia.get('tendencia', 'N/A')}`", "inline": True}
                ],
                "footer": {"text": f"Execução: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
            }]
        }
        requests.post(webhook, json=payload)

if __name__ == "__main__":
    MotorEvolutivoSafeDriver().executar_pipeline()
