import sys
import os
import boto3
import json
import logging
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from botocore.config import Config
from google.cloud import bigquery
from google.oauth2 import service_account

# Imports corrigidos para baterem com os seus arquivos físicos
from autobot.ingestao_bronze import IngestaoBronze
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SafeDriverMaestro:
    def __init__(self):
        self.tz = ZoneInfo("America/Sao_Paulo")
        self.inicio = datetime.now(self.tz)
        self.cal = CalendarioEstrategico()
        self.comunicador = ComunicadorSafeDriver()

    def _verificar_integridade_infra(self):
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
                config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()
            # Verifica se o modelo existe (ajustado para o nome 'geral')
            s3.head_object(Bucket=bucket, Key="datalake/modelos_ml/latest_cat_geral.pkl")
            return False 
        except Exception:
            return True 

    def run(self, force=False):
        try:
            logger.info("SafeDriver Maestro: Iniciando orquestração H3-L9.")

            # 1. BRONZE
            bronze = IngestaoBronze()
            novos_dados_ssp = bronze.executar_ingestao_continua()
            
            deve_rodar_calendario = self.cal.deve_rodar_hoje()
            precisa_reparo = self._verificar_integridade_infra()
            
            executar_pipeline = novos_dados_ssp or deve_rodar_calendario or precisa_reparo or force

            if not executar_pipeline:
                logger.info("Pipeline em repouso: Sem gatilhos ativos.")
                return

            if force: gatilho = "Execução Manual Forçada"
            elif novos_dados_ssp: gatilho = "Novos Dados SSP-SP"
            elif precisa_reparo: gatilho = "Reparo de Infraestrutura"
            else: gatilho = "Ciclo Estratégico (Sazonalidade)"

            logger.info(f"Gatilho: {gatilho}")

            # 2. PRATA
            prata = ProcessamentoPrata()
            resultado_prata = prata.executar_todos_os_anos(force=force)
            metricas_prata = resultado_prata.get("metricas", {})

            # 3. IA (Tweedie)
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha no Treinamento Evolutivo.")
            resumo_ia = treinador.obter_metricas_finais()

            # 4. OURO (Star Schema + View)
            ouro = CamadaOuroSafeDriver()
            relatorio_ouro = ouro.executar_predicao_atual()
            if not relatorio_ouro:
                raise RuntimeError("Falha na Sincronização BigQuery.")

            tempo_total = str(datetime.now(self.tz) - self.inicio).split(".")[0]

            self._disparar_feedback_discord(gatilho, tempo_total, metricas_prata, resumo_ia, relatorio_ouro)
            logger.info(f"SafeDriver finalizado com sucesso em {tempo_total}.")

        except Exception as e:
            logger.error(f"FALHA NO ORQUESTRADOR: {e}")
            self.comunicador.relatar_erro_critico("Orquestrador Maestro", e)
            sys.exit(1)

    def _disparar_feedback_discord(self, gatilho, tempo, m_prata, m_ia, r_ouro):
        # MAE Médio entre CatBoost e LGBM
        mae_medio = sum((i.get('mae_cat', 0) + i.get('mae_lgb', 0)) / 2 for i in m_ia) / len(m_ia) if m_ia else 0
        
        payload = {
            "content": "🚀 **SafeDriver - Pipeline de Dados e IA Concluído**",
            "embeds": [
                {
                    "title": "⚙️ Visão Executiva: Orquestração Maestro",
                    "color": 3066993,
                    "fields": [
                        {"name": "Trigger", "value": f"`{gatilho}`", "inline": True},
                        {"name": "Tempo Total", "value": f"`{tempo}`", "inline": True},
                        {"name": "Status", "value": "🟢 Sucesso", "inline": True}
                    ]
                },
                {
                    "title": "💎 Camada Prata: Cura Geográfica",
                    "color": 12370112,
                    "fields": [
                        {"name": "B.Os Processados", "value": f"{m_prata.get('processados', 0):,}", "inline": True},
                        {"name": "Malha Curada", "value": f"+ {m_prata.get('malha_resgatada', 0):,}", "inline": True}
                    ]
                },
                {
                    "title": "🧠 Camada IA: Treinamento Tweedie",
                    "color": 3447003,
                    "fields": [
                        {"name": "MAE Médio (Ensemble)", "value": f"`{mae_medio:.4f}`", "inline": True},
                        {"name": "Distribuição", "value": "Tweedie", "inline": True}
                    ]
                },
                {
                    "title": "🏆 Camada Ouro: Modelo Semântico DW",
                    "color": 16766720,
                    "fields": [
                        {"name": "DW Destino", "value": "Google BigQuery", "inline": True},
                        {"name": "H3 Sincronizados", "value": f"{r_ouro['embeds'][0]['fields'][0]['value']}", "inline": True}
                    ]
                }
            ]
        }
        self.comunicador.enviar_webhook(payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    SafeDriverMaestro().run(force=args.force)
