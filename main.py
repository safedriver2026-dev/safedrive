import os
import logging
import argparse
import sys
import boto3
from botocore.config import Config
from datetime import datetime

# Importação dos motores do SafeDriver
try:
    from autobot.ingestao_bronze import IngestaoBronze
    from autobot.processamento_prata import ProcessamentoPrata
    from autobot.treinador_ia import TreinadorEvolutivo
    from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
    from autobot.comunicador import ComunicadorSafeDriver
except ImportError as e:
    print(f"Erro ao importar módulos do diretório 'autobot': {e}")
    sys.exit(1)

# Configuração de Log
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def verificar_estado_remoto(prefixo_path):
    """Verifica a existência de um arquivo ou diretório no Cloudflare R2."""
    try:
        s3 = boto3.client('s3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        s3.head_object(Bucket=bucket, Key=prefixo_path)
        return True
    except:
        return False

def orquestrar_pipeline(force_reprocess=False):
    """
    Coordena o fluxo com Retomada Inteligente:
    Bronze -> Prata -> IA -> Ouro
    """
    logger.info(f"🛡️ SafeDriver Maestro: Iniciando orquestração (Modo Force: {force_reprocess}).")
    start_time = datetime.now()
    comunicador = ComunicadorSafeDriver()

    try:
        # --- ETAPA 1: BRONZE ---
        bronze = IngestaoBronze()
        novos_dados_bronze = bronze.executar_ingestao_continua(force=force_reprocess)

        # --- ANÁLISE DE ESTADO DO DATA LAKE ---
        # Verifica se temos os modelos treinados e se o arquivo final da ouro existe
        modelos_existem = verificar_estado_remoto("datalake/modelos_ml/latest_cat_geral.pkl")
        ouro_existe = verificar_estado_remoto("datalake/ouro/fato_risco_consolidada.parquet")
        
        # Lógica de Gatilhos
        gatilho_prata = novos_dados_bronze or force_reprocess
        gatilho_ia = gatilho_prata or not modelos_existem
        gatilho_ouro = gatilho_ia or not ouro_existe

        # Se tudo estiver OK, entra em repouso
        if not gatilho_prata and not gatilho_ia and not gatilho_ouro:
            logger.info("😴 Decisão: Data Lakehouse 100% atualizado. Pipeline em repouso.")
            return

        # --- ETAPA 2: PRATA (Transformação) ---
        if gatilho_prata:
            logger.info("🚀 Prata: Novos dados detectados. Iniciando processamento H3...")
            prata = ProcessamentoPrata()
            prata.executar_todos_os_anos(force=force_reprocess)
        else:
            logger.info("⏭️ Prata: Cache válido. Pulando transformação.")

        # --- ETAPA 3: IA (Treinamento) ---
        if gatilho_ia:
            logger.info("🧠 IA: Modelos ausentes ou desatualizados. Iniciando Treinador...")
            treinador = TreinadorEvolutivo(dev_mode=False) 
            if not treinador.treinar_modelo_mestre():
                comunicador.relatar_erro_critico("Treinador IA", "Falha no treinamento dos modelos.")
                return
        else:
            logger.info("⏭️ IA: Modelos de produção encontrados. Pulando treinamento.")

        # --- ETAPA 4: OURO (Predição e Sincronização) ---
        if gatilho_ouro:
            logger.info("🏆 OURO: Sincronizando Data Warehouse e Gerando Predições...")
            ouro = CamadaOuroSafeDriver(dev_mode=False) 
            if ouro.executar_predicao_atual():
                duration = datetime.now() - start_time
                logger.info(f"✨ SafeDriver finalizado com sucesso em {duration}.")
                
                # Notificação Executiva
                comunicador.enviar_webhook({
                    "embeds": [{
                        "title": "✅ Sistema SafeDriver Atualizado",
                        "color": 3066993,
                        "fields": [
                            {"name": "Transformação Prata", "value": "✅" if gatilho_prata else "⏭️ (Cache)", "inline": True},
                            {"name": "Treino de IA", "value": "✅" if gatilho_ia else "⏭️ (Cache)", "inline": True},
                            {"name": "Carga BigQuery", "value": "✅", "inline": True},
                            {"name": "Duração Total", "value": f"`{duration}`", "inline": False}
                        ],
                        "footer": {"text": "SafeDriver Autobot v2.0"}
                    }]
                })
            else:
                comunicador.relatar_erro_critico("Camada Ouro", "Erro na sincronização BigQuery.")

    except Exception as e:
        logger.error(f"💥 FALHA CRÍTICA NO ORQUESTRADOR: {e}", exc_info=True)
        comunicador.relatar_erro_critico("Maestro Central", str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()
    orquestrar_pipeline(force_reprocess=args.force)
