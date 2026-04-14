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

# Importação dos módulos do ecossistema SafeDriver
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
        """
        Valida se o ambiente precisa de intervenção usando o JSON da Service Account.
        """
        logger.info("AUDITORIA: Validando integridade da infraestrutura...")
        
        try:
            # 1. Check R2 (Cloudflare)
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
                config=Config(signature_version='s3v4')
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()
            s3.head_object(Bucket=bucket, Key="safedriver/modelos_ml/latest_cat_motorista.pkl")
            
            # 2. Check BigQuery (Google Cloud com JSON)
            gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            
            bq = bigquery.Client(credentials=credentials, project=os.getenv("BQ_PROJECT_ID"))
            tabela = f"{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.fato_risco_predicao_atual"
            bq.get_table(tabela)
            
            return False 
        except Exception as e:
            logger.warning(f"AUDITORIA: Infraestrutura incompleta ou credenciais inválidas: {e}")
            return True 

    def run(self, force=False):
        try:
            logger.info("SafeDriver Maestro: Iniciando orquestração.")

            bronze = IngestaoBronze()
            novos_dados_ssp = bronze.executar_ingestao_continua()
            
            deve_rodar_calendario = self.cal.deve_rodar_hoje()
            infra_quebrada = self._verificar_integridade_infra()
            
            executar_pipeline = novos_dados_ssp or deve_rodar_calendario or infra_quebrada or force

            if not executar_pipeline:
                logger.info("Pipeline em repouso: Tudo sincronizado.")
                return

            if force: gatilho = "Execução Manual Forçada"
            elif novos_dados_ssp: gatilho = "Novos Dados detectados na SSP-SP"
            elif infra_quebrada: gatilho = "Recuperação de Infraestrutura (Auto-Reparo)"
            else: gatilho = "Ciclo Estratégico (Calendário de Risco)"

            # --- PROCESSAMENTO ---
            prata = ProcessamentoPrata()
            resumo_prata = prata.executar_todos_os_anos(force=force) or {}

            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha no treinamento da IA.")
            resumo_ia = treinador.obter_metricas_finais() or {}

            ouro = CamadaOuroSafeDriver()
            if not ouro.executar_predicao_atual():
                raise RuntimeError("Falha na sincronização BigQuery.")

            # --- RELATÓRIO DISCORD ---
            fim = datetime.now(self.tz)
            tempo_total = str(fim - self.inicio).split(".")[0]

            self.comunicador.relatar_conclusao_rica(
                contexto={"gatilho": gatilho, "tempo_total": tempo_total},
                metricas_executivas={
                    "Crimes Processados": resumo_prata.get('total_linhas', 'N/A'),
                    "Municípios Cobertos": "Estado de São Paulo",
                    "Status da IA": "Modelos Evoluídos"
                },
                metricas_operacionais={
                    "Recuperados via H8": f"{resumo_prata.get('recuperados', 0)} end.",
                    "Erro Médio (MAE)": f"{resumo_ia.get('mae', 0.0):.4f}",
                    "Sincronização BQ": "Sucesso (Upsert)"
                }
            )
            logger.info(f"✅ SafeDriver finalizado em {tempo_total}.")

        except Exception as e:
            msg_erro = str(e)
            logger.error(f"FALHA: {msg_erro}")
            self.comunicador.relatar_alerta_critico(
                modulo="Orquestrador Maestro", 
                status="Interrupção do Fluxo", 
                detalhe_erro=msg_erro
            )
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDriver Maestro")
    parser.add_argument('--force', action='store_true', help="Força reprocessamento total.")
    args = parser.parse_args()
    SafeDriverMaestro().run(force=args.force)
