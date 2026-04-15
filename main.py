import os
import logging
import argparse
import sys
import boto3
from botocore.config import Config
from datetime import datetime


try:
    from autobot.ingestao_bronze import IngestaoBronze
    from autobot.processamento_prata import ProcessamentoPrata
    from autobot.treinador_ia import TreinadorEvolutivo
    from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
    from autobot.comunicador import ComunicadorSafeDriver
except ImportError as e:
    print(f"Erro ao importar módulos do diretório 'autobot': {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def verificar_modelos_ia_existem():
    """Verifica no R2 se os modelos da IA já foram criados alguma vez."""
    try:
        s3 = boto3.client('s3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
       
        s3.head_object(Bucket=bucket, Key="datalake/modelos_ml/latest_cat_geral.pkl")
        return True
    except:
        return False

def orquestrar_pipeline(force_reprocess=False):
    logger.info(f"🛡️ SafeDriver Maestro: Iniciando orquestração (Modo Force: {force_reprocess}).")
    start_time = datetime.now()
    comunicador = ComunicadorSafeDriver()

    try:
        # --- ETAPA 1: BRONZE ---
        bronze = IngestaoBronze()
        novos_dados_bronze = bronze.executar_ingestao_continua(force=force_reprocess)

   
        modelos_existem = verificar_modelos_ia_existem()
        
       
        gatilho_prata = novos_dados_bronze or force_reprocess
        gatilho_ia_ouro = gatilho_prata or not modelos_existem

        if not gatilho_prata and not gatilho_ia_ouro:
            logger.info("😴 Decisão: Dados atualizados e Modelos intactos. Pipeline em repouso.")
            return

        # --- ETAPA 2: PRATA ---
        if gatilho_prata:
            logger.info("🚀 Gatilho Ativado: Iniciando Processamento Prata.")
            prata = ProcessamentoPrata()
            prata.executar_todos_os_anos(force=force_reprocess)
            logger.info(f"✅ Prata finalizada.")
        else:
            logger.info("⏭️ Prata pulada (Dados já atualizados).")

        # --- ETAPA 3: IA ---
        if gatilho_ia_ouro:
            if not modelos_existem and not gatilho_prata:
                logger.warning("⚠️ Modelos da IA ausentes detectados! Retomando o pipeline a partir do Treinamento.")
            
            logger.info("🧠 IA: Iniciando treinamento evolutivo (Produção Completa).")
            treinador = TreinadorEvolutivo(dev_mode=False) 
            sucesso_ia = treinador.treinar_modelo_mestre()
            
            if sucesso_ia:
                logger.info(f"📈 IA Treinada com Sucesso.")
            else:
                comunicador.relatar_erro_critico("Camada IA (Treinador)", "Falha no treinamento.")
                return
            
            # --- ETAPA 4: OURO ---
            logger.info("🏆 OURO: Sincronizando Data Warehouse e processando Explicabilidade (XAI).")
            ouro = CamadaOuroSafeDriver(dev_mode=False) 
            resultado_ouro = ouro.executar_predicao_atual()
            
            if resultado_ouro:
                duration = datetime.now() - start_time
                logger.info("✨ Pipeline SafeDriver finalizado com sucesso absoluto.")
                comunicador.enviar_webhook({
                    "embeds": [{
                        "title": "✅ Pipeline SafeDriver Concluído!",
                        "description": "Data Warehouse atualizado com sucesso.",
                        "color": 3066993,
                        "fields": [
                            {"name": "Status da Prata", "value": "Executada" if gatilho_prata else "Pulada (Cache)", "inline": True},
                            {"name": "Status IA/Ouro", "value": "Executada", "inline": True},
                            {"name": "Tempo", "value": f"`{duration}`", "inline": False}
                        ]
                    }]
                })
            else:
                comunicador.relatar_erro_critico("Camada Ouro", "Falha na sincronização BigQuery.")

    except Exception as e:
        logger.error(f"💥 FALHA CRÍTICA NO ORQUESTRADOR: {e}", exc_info=True)
        comunicador.relatar_erro_critico("Maestro Central (main.py)", str(e))
        sys.exit(1)

    logger.info(f"⏱️ Duração total do ciclo de dados: {datetime.now() - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()
    orquestrar_pipeline(force_reprocess=args.force)
