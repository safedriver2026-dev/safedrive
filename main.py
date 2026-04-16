import os
import logging
import argparse
import sys
import boto3
from botocore.config import Config
from datetime import datetime

# Importação dos motores SafeDriver
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

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validar_ambiente():
    """Verifica se todas as chaves de acesso estão presentes antes de iniciar."""
    keys = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "BQ_SERVICE_ACCOUNT_JSON", "DISCORD_SUCESSO"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        logger.error(f"🚫 Variáveis de ambiente ausentes: {missing}")
        sys.exit(1)
    logger.info("✅ Validação de ambiente concluída.")

def verificar_existencia_r2(caminho_relativo):
    """Verifica fisicamente se um arquivo existe no storage R2."""
    try:
        s3 = boto3.client('s3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        # Assume 'datalake/' como base padrão se não detectar dinamicamente
        s3.head_object(Bucket=bucket, Key=f"datalake/{caminho_relativo}".replace("//", "/"))
        return True
    except:
        return False

def orquestrar_pipeline(force_reprocess=False):
    validar_ambiente()
    start_time = datetime.now()
    comunicador = ComunicadorSafeDriver()
    cal = CalendarioEstrategico()
    
    # Telemetria de Ciclo
    stats = {
        "status_camadas": {"bronze": "⏭️", "prata": "⏭️", "ia": "⏭️", "ouro": "⏭️"},
        "metrics_ia": {"mae": "N/A"},
        "duracao": "0s",
        "severidade_total": 0
    }

    try:
        # 1. GATILHOS DE EXECUÇÃO
        # Se o Calendário Estratégico diz que 'deve_rodar_hoje', forçamos a Ouro pelo menos
        deve_rodar_estrategico = cal.deve_rodar_hoje()
        
        # --- ETAPA 1: BRONZE ---
        bronze = IngestaoBronze()
        novos_dados_bronze = bronze.executar_ingestao_continua(force=force_reprocess)
        if novos_dados_bronze: stats["status_camadas"]["bronze"] = "✅ Novo"

        # --- VERIFICAÇÃO DE ESTADO ---
        ano_atual = datetime.now().year
        prata_existe = verificar_existencia_r2(f"prata/ssp_consolidada_{ano_atual}.parquet")
        modelos_existem = verificar_existencia_r2("modelos_ml/latest_cat_geral.pkl")
        
        # --- ETAPA 2: PRATA ---
        if novos_dados_bronze or force_reprocess or not prata_existe:
            logger.info("🚀 Iniciando Camada Prata (Matriz de Gravidade)...")
            prata = ProcessamentoPrata()
            if prata.executar_todos_os_anos(force=force_reprocess):
                stats["status_camadas"]["prata"] = "✅ Atualizada"
            else: raise Exception("Falha no processamento da Prata.")
        
        # --- ETAPA 3: IA ---
        if stats["status_camadas"]["prata"] == "✅ Atualizada" or not modelos_existem:
            logger.info("🧠 Iniciando Treinamento IA (Corte Cronológico)...")
            treinador = TreinadorEvolutivo()
            if treinador.treinar_modelo_mestre():
                stats["status_camadas"]["ia"] = "✅ Treinado"
                stats["metrics_ia"].update(treinador.obter_stats())
            else: raise Exception("Falha no treinamento da IA.")

        # --- ETAPA 4: OURO ---
        # A Ouro roda se houver novos dados, novos modelos ou se for um dia estratégico (Pagamento/Feriado)
        if stats["status_camadas"]["ia"] == "✅ Treinado" or deve_rodar_estrategico or force_reprocess:
            logger.info("🏆 Iniciando Camada Ouro (Star Schema BigQuery)...")
            ouro = CamadaOuroSafeDriver()
            if ouro.executar_predicao_atual():
                stats["status_camadas"]["ouro"] = "✅ Sincronizado"
            else: raise Exception("Falha na sincronização da Ouro.")

        # Finalização
        stats["duracao"] = str(datetime.now() - start_time).split(".")[0]
        comunicador.enviar_relatorio_executivo(stats)
        logger.info(f"✨ Ciclo concluído com sucesso em {stats['duracao']}.")

    except Exception as e:
        logger.error(f"💥 FALHA NO MAESTRO: {e}", exc_info=True)
        comunicador.relatar_erro_critico("Orquestrador Central", str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help="Forçar reprocessamento")
    args = parser.parse_args()
    orquestrar_pipeline(force_reprocess=args.force)
