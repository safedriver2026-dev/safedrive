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
        Auditor de Integridade: Valida se o ambiente precisa de reparo (Backfill).
        Utiliza o JSON da Service Account para checar o BigQuery.
        """
        logger.info("AUDITORIA: Validando integridade da infraestrutura (R2 e BigQuery)...")
        
        try:
            # 1. Verificação no Cloudflare R2
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
                config=Config(signature_version='s3v4')
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()
            # Checa se o arquivo de tracker ou o modelo mais recente existem
            s3.head_object(Bucket=bucket, Key="safedriver/modelos_ml/latest_cat_motorista.pkl")
            
            # 2. Verificação no BigQuery usando o JSON das Secrets
            gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            
            bq = bigquery.Client(credentials=credentials, project=os.getenv("BQ_PROJECT_ID"))
            tabela_ref = f"{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.fato_risco_predicao_atual"
            bq.get_table(tabela_ref)
            
            return False # Infraestrutura íntegra
        except Exception as e:
            logger.warning(f"AUDITORIA: Anomalia detectada ou infraestrutura incompleta: {e}")
            return True # Necessita de execução para auto-reparo

    def run(self, force=False):
        try:
            logger.info("SafeDriver Maestro: Iniciando orquestração do ecossistema.")

            # ETAPA 0: Ingestão e Monitoramento de Mudanças (CDC)
            bronze = IngestaoBronze()
            novos_dados_ssp = bronze.executar_ingestao_continua()
            
            deve_rodar_calendario = self.cal.deve_rodar_hoje()
            precisa_reparo = self._verificar_integridade_infra()
            
     
            executar_pipeline = novos_dados_ssp or deve_rodar_calendario or precisa_reparo or force

            if not executar_pipeline:
                logger.info("Pipeline em repouso: Dados sincronizados e sem gatilhos ativos.")
                return

  
            if force: 
                gatilho = "Execução Manual (Retroativo/Forçado)"
            elif novos_dados_ssp: 
                gatilho = "Novos Dados detectados na SSP-SP"
            elif precisa_reparo: 
                gatilho = "Reparo Automático de Infraestrutura"
            else: 
                gatilho = "Ciclo Estratégico (Calendário de Risco)"

            logger.info(f"Gatilho acionado: {gatilho}")

            
            prata = ProcessamentoPrata()
            resumo_prata = prata.executar_todos_os_anos(force=force) or {}

       
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha crítica no treinamento dos modelos de IA.")
            resumo_ia = treinador.obter_metricas_finais() or {}

         
            ouro = CamadaOuroSafeDriver()
            if not ouro.executar_predicao_atual():
                raise RuntimeError("Falha na sincronização dos dados com o BigQuery.")

           
            
            tempo_total = str(datetime.now(self.tz) - self.inicio).split(".")[0]

 
            contexto = {
                "gatilho": gatilho,
                "tempo_total": tempo_total
            }

            metricas_executivas = {
                "Linhas Processadas": f"{resumo_prata.get('total_linhas', 0):,}",
                "Status da Base": "Sincronizada (BigQuery)",
                "Inteligência IA": "Modelos Atualizados"
            }

            metricas_operacionais = {
                "Recuperados via H8": f"{resumo_prata.get('recuperados', 0)} registros",
                "Erro Médio (MAE)": f"{resumo_ia.get('mae', 0.0):.4f}",
                "Fuso Horário": "Brasília (BRT)"
            }

            self.comunicador.relatar_conclusao_rica(contexto, metricas_executivas, metricas_operacionais)
            logger.info(f"✅ SafeDriver finalizado com sucesso em {tempo_total}.")

        except Exception as e:
            msg_erro = str(e)
            logger.error(f"FALHA NO ORQUESTRADOR: {msg_erro}")
            self.comunicador.relatar_alerta_critico(
                modulo="Orquestrador Maestro", 
                status="Interrupção do Fluxo", 
                detalhe_erro=msg_erro
            )
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDriver Maestro - Orquestrador de Dados e IA")
    parser.add_argument('--force', action='store_true', help="Força o reprocessamento histórico de todas as camadas.")
    args = parser.parse_args()

    SafeDriverMaestro().run(force=args.force)
