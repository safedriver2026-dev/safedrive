import os
import logging
import argparse
import sys
import boto3
from botocore.config import Config
from datetime import datetime

# Importação dos motores do ecossistema SafeDriver
try:
    from autobot.ingestao_bronze import IngestaoBronze
    from autobot.processamento_prata import ProcessamentoPrata
    from autobot.treinador_ia import TreinadorEvolutivo
    from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
    from autobot.comunicador import ComunicadorSafeDriver
    from autobot.calendario_estrategico import CalendarioEstrategico
except ImportError as e:
    print(f"❌ Erro de dependência: {e}")
    sys.exit(1)

# Configuração de Log de Alta Visibilidade
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validar_ambiente():
    """Verifica se todas as chaves de acesso estão presentes antes de iniciar."""
    keys = [
        "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
        "BQ_PROJECT_ID", "BQ_SERVICE_ACCOUNT_JSON", "DISCORD_SUCESSO"
    ]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        logger.error(f"🚫 Configuração incompleta. Variáveis ausentes: {missing}")
        sys.exit(1)
    logger.info("✅ Ambiente validado com sucesso.")

def verificar_estado_remoto(caminho_relativo):
    """Verifica a existência física de artefatos no Cloudflare R2."""
    try:
        s3 = boto3.client('s3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        # Busca padrão no diretório datalake
        s3.head_object(Bucket=bucket, Key=f"datalake/{caminho_relativo}".replace("//", "/"))
        return True
    except:
        return False

def orquestrar_pipeline(force_reprocess=False):
    """
    Coordena o ciclo de vida dos dados com Observabilidade e Retomada Inteligente.
    """
    validar_ambiente()
    logger.info(f"🛡️ SafeDriver Maestro: Iniciando ciclo (Force: {force_reprocess}).")
    start_time = datetime.now()
    
    comunicador = ComunicadorSafeDriver()
    cal = CalendarioEstrategico()

    # Telemetria de Execução
    stats = {
        "status_camadas": {"bronze": "⏭️", "prata": "⏭️", "ia": "⏭️", "ouro": "⏭️"},
        "hygiene": {"taxa_recuperacao": 100, "linhas_ouro": 0, "recuperado_grade": 0},
        "metrics_ia": {"mae": "N/A"},
        "duracao": "0s"
    }

    try:
        # 1. GATILHOS DE EXECUÇÃO
        deve_rodar_estrategico = cal.deve_rodar_hoje()
        
        # --- ETAPA 1: BRONZE (Ingestão Resiliente) ---
        bronze = IngestaoBronze()
        novos_dados_bronze = bronze.executar_ingestao_continua(force=force_reprocess)
        if novos_dados_bronze: 
            stats["status_camadas"]["bronze"] = "✅ Novo"

        # --- ANÁLISE DE ESTADO PARA DECISÃO DE ETAPAS ---
        ano_atual = datetime.now().year
        prata_existe = verificar_estado_remoto(f"prata/ssp_consolidada_{ano_atual}.parquet")
        modelos_existem = verificar_estado_remoto("modelos_ml/latest_cat_geral.pkl")
        
        gatilho_prata = novos_dados_bronze or force_reprocess or not prata_existe
        gatilho_ia = gatilho_prata or not modelos_existem
        
        # --- ETAPA 2: PRATA (Heavy Silver com Matriz de Gravidade) ---
        if gatilho_prata:
            logger.info("🚀 Prata: Processando crimes e injetando pesos de gravidade...")
            prata = ProcessamentoPrata()
            res_prata = prata.executar_todos_os_anos(force=force_reprocess)
            if res_prata:
                stats["status_camadas"]["prata"] = "✅ Atualizada"
            else:
                raise Exception("Erro crítico no processamento da Camada Prata.")
        
        # --- ETAPA 3: IA (Treino Evolutivo Tweedie) ---
        if gatilho_ia:
            logger.info("🧠 IA: Ajustando modelos preditivos ao histórico recente...")
            treinador = TreinadorEvolutivo()
            if treinador.treinar_modelo_mestre():
                stats["status_camadas"]["ia"] = "✅ Treinado"
                stats["metrics_ia"].update(treinador.obter_stats())
            else:
                raise Exception("Falha no treinamento dos modelos de Machine Learning.")

        # --- ETAPA 4: OURO (Scaffold de Risco e BigQuery) ---
        # A Ouro roda se houver nova IA, novos dados ou gatilho estratégico do calendário
        if stats["status_camadas"]["ia"] == "✅ Treinado" or deve_rodar_estrategico or force_reprocess:
            logger.info("🏆 Ouro: Sincronizando Star Schema e calculando pressão espacial...")
            ouro = CamadaOuroSafeDriver()
            if ouro.executar_predicao_atual():
                stats["status_camadas"]["ouro"] = "✅ Sincronizado"
                stats["hygiene"]["linhas_ouro"] = "Malha Completa"
            else:
                raise Exception("Erro na sincronização final com o BigQuery.")

        # FINALIZAÇÃO
        stats["duracao"] = str(datetime.now() - start_time).split(".")[0]
        logger.info(f"✨ Ciclo concluído com sucesso em {stats['duracao']}.")
        comunicador.enviar_relatorio_executivo(stats)

    except Exception as e:
        logger.error(f"💥 FALHA NO MAESTRO: {e}", exc_info=True)
        comunicador.relatar_erro_critico("Orquestrador Central", str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDriver Autobot Maestro")
    parser.add_argument('--force', action='store_true', help="Forçar reprocessamento de todo o Data Lake")
    args = parser.parse_args()
    
    orquestrar_pipeline(force_reprocess=args.force)
