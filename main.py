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

# Imports atualizados para a nova estrutura de arquivos
from autobot.camada_bronze import CamadaBronzeSafeDriver
from autobot.camada_prata import CamadaPrataSafeDriver
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador_discord import ComunicadorSafeDriver

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
        """Verifica se os artefatos mínimos existem na nuvem."""
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=self.comunicador.webhook_sucesso, # Apenas placeholder
                config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()
            # Verifica o modelo mais importante
            s3.head_object(Bucket=bucket, Key="datalake/modelos_ml/latest_cat_geral.pkl")
            
            # Verifica o BigQuery
            gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            bq = bigquery.Client(credentials=credentials, project=os.getenv("BQ_PROJECT_ID"))
            tabela_ref = f"{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.fato_risco_h3_atual"
            bq.get_table(tabela_ref)
            
            return False # Infra OK, não precisa de reparo forçado
        except Exception:
            return True # Infra com falhas, gatilho de reparo ativado

    def run(self, force=False):
        try:
            logger.info("SafeDriver Maestro: Iniciando orquestração H3-L9.")

            # 1. BRONZE: Ingestão
            bronze = CamadaBronzeSafeDriver()
            novos_dados_ssp = bronze.executar_cdc()
            
            deve_rodar_calendario = self.cal.deve_rodar_hoje()
            precisa_reparo = self._verificar_integridade_infra()
            
            executar_pipeline = novos_dados_ssp or deve_rodar_calendario or precisa_reparo or force

            if not executar_pipeline:
                logger.info("Pipeline em repouso: Sem gatilhos ativos.")
                return

            # Definição do Gatilho para o Log/Discord
            if force: gatilho = "Execução Manual Forçada"
            elif novos_dados_ssp: gatilho = "Novos Dados SSP-SP"
            elif precisa_reparo: gatilho = "Reparo de Infraestrutura"
            else: gatilho = "Ciclo Estratégico (Sazonalidade)"

            logger.info(f"Gatilho Ativo: {gatilho}")

            # 2. PRATA: Processamento e Cura
            prata = CamadaPrataSafeDriver()
            # O método agora retorna um dicionário com métricas de resgate
            resultado_prata = prata.processar_limpeza_e_h3(force=force)
            
            # 3. IA: Treinamento Tweedie
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha no Treinamento Evolutivo Tweedie.")
            resumo_ia = treinador.obter_metricas_finais()

            # 4. OURO: Sincronização Estrela (BigQuery)
            ouro = CamadaOuroSafeDriver()
            relatorio_ouro = ouro.executar_predicao_atual()
            if not relatorio_ouro:
                raise RuntimeError("Falha na Sincronização BigQuery / Star Schema.")

            tempo_total = str(datetime.now(self.tz) - self.inicio).split(".")[0]

            # 5. FEEDBACK: Discord
            self._disparar_feedback_discord(gatilho, tempo_total, resultado_prata, resumo_ia, relatorio_ouro)
            
            logger.info(f"SafeDriver finalizado com sucesso em {tempo_total}.")

        except Exception as e:
            logger.error(f"FALHA NO ORQUESTRADOR: {e}")
            self.comunicador.relatar_erro_critico("Orquestrador Maestro", e)
            sys.exit(1)

    def _disparar_feedback_discord(self, gatilho, tempo, m_prata, m_ia, r_ouro):
        # Cálculo de MAE ponderado para o resumo
        mae_cat = sum(i.get('mae_cat', 0) for i in m_ia) / len(m_ia) if m_ia else 0
        mae_lgb = sum(i.get('mae_lgb', 0) for i in m_ia) / len(m_ia) if m_ia else 0
        
        # O retorno da Ouro agora é um dict de embeds, pegamos o valor de registros do campo certo
        registros_ouro = r_ouro['embeds'][0]['fields'][0]['value']

        payload = {
            "content": "🚀 **SafeDriver - Pipeline de Dados e IA Concluído**",
            "embeds": [
                {
                    "title": "⚙️ Visão Executiva: Orquestração Maestro",
                    "color": 3066993,
                    "fields": [
                        {"name": "Trigger", "value": f"`{gatilho}`", "inline": True},
                        {"name": "Tempo Total", "value": f"`{tempo}`", "inline": True},
                        {"name": "Status Final", "value": "🟢 Sucesso Total", "inline": True}
                    ]
                },
                {
                    "title": "💎 Camada Prata: Cura Geográfica",
                    "color": 12370112,
                    "fields": [
                        {"name": "Resgate de Bairros (Malha)", "value": f"+ {m_prata.get('malha_resgatada', 0):,}", "inline": True},
                        {"name": "Cura Semântica", "value": "Simbiótica/Injetada", "inline": True}
                    ]
                },
                {
                    "title": "🧠 Camada IA: Treinamento Tweedie",
                    "color": 3447003,
                    "fields": [
                        {"name": "MAE (CatBoost)", "value": f"`{mae_cat:.4f}`", "inline": True},
                        {"name": "MAE (LightGBM)", "value": f"`{mae_lgb:.4f}`", "inline": True},
                        {"name": "Distribuição", "value": "Tweedie (Zero-Inflated)", "inline": False}
                    ]
                },
                {
                    "title": "🏆 Camada Ouro: Modelo Semântico DW",
                    "color": 16766720,
                    "fields": [
                        {"name": "DW Destino", "value": "BigQuery (Star Schema)", "inline": True},
                        {"name": "H3 Sincronizados", "value": registros_ouro, "inline": True},
                        {"name": "View Analítica", "value": "`v_safedriver_analitico`", "inline": False}
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
