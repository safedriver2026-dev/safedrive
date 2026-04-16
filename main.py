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
except ImportError as e:
    print(f"Erro ao importar módulos do diretório 'autobot': {e}")
    sys.exit(1)

# Configuração de Log de Alta Visibilidade
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def verificar_estado_remoto(prefixo_path):
    """Verifica a existência física de artefatos no Cloudflare R2."""
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
    Coordena o ciclo de vida dos dados com Observabilidade e Retomada Inteligente.
    """
    logger.info(f"🛡️ SafeDriver Maestro: Iniciando ciclo (Modo Force: {force_reprocess}).")
    start_time = datetime.now()
    comunicador = ComunicadorSafeDriver()

    # Telemetria centralizada para o Relatório Executivo
    stats = {
        "status_camadas": {
            "bronze": "⏭️ (Cache)", 
            "prata": "⏭️ (Cache)", 
            "ia": "⏭️ (Cache)", 
            "ouro": "⏭️ (Cache)"
        },
        "hygiene": {
            "linhas_in": 0, 
            "linhas_out": 0, 
            "taxa_recuperacao": 100, 
            "linhas_ouro": 0,
            "recuperado_grade": 0
        },
        "metrics_ia": {"mae": "Aguardando..."},
        "duracao": "0s"
    }

    try:
        # --- ETAPA 1: BRONZE (Ingestão) ---
        bronze = IngestaoBronze()
        # novos_dados_bronze indica se houve download ou mudança de tamanho na SSP
        novos_dados_bronze = bronze.executar_ingestao_continua(force=force_reprocess)
        if novos_dados_bronze: 
            stats["status_camadas"]["bronze"] = "✅ (Novo)"

        # --- ANÁLISE DE ESTADO FÍSICO (Verificação de Integridade no R2) ---
        prata_existe = verificar_estado_remoto("datalake/prata/ssp_consolidada_2022.parquet")
        modelos_existem = verificar_estado_remoto("datalake/modelos_ml/latest_cat_geral.pkl")
        ouro_existe = verificar_estado_remoto("datalake/ouro/fato_risco_consolidada.parquet")
        
        # Lógica de Gatilhos: Só reprocessa se houver dado novo, se for forçado ou se o arquivo sumiu
        gatilho_prata = novos_dados_bronze or force_reprocess or not prata_existe
        gatilho_ia = gatilho_prata or not modelos_existem
        gatilho_ouro = gatilho_ia or not ouro_existe

        # --- ETAPA 2: PRATA (Higiene e Normalização) ---
        prata = ProcessamentoPrata()
        if gatilho_prata:
            logger.info("🚀 Prata: Iniciando processamento e filtragem de qualidade...")
            metricas_prata = prata.executar_todos_os_anos(force=force_reprocess)
            
            if metricas_prata and isinstance(metricas_prata, dict):
                stats["status_camadas"]["prata"] = "✅ (Processado)"
                stats["hygiene"].update(metricas_prata)
            else:
                stats["status_camadas"]["prata"] = "⚠️ (Vazio/Erro)"
        else:
            logger.info("⏭️ Prata: Cache geográfico validado.")

        # --- ETAPA 3: IA (Treinamento Evolutivo) ---
        if gatilho_ia:
            logger.info("🧠 IA: Iniciando treinamento dos modelos Tweedie...")
            treinador = TreinadorEvolutivo(dev_mode=False) 
            if treinador.treinar_modelo_mestre():
                stats["status_camadas"]["ia"] = "✅ (Treinado)"
                stats["metrics_ia"].update(treinador.obter_stats())
            else:
                comunicador.relatar_erro_critico("Treinador IA", "Falha no treinamento.")
                return
        else:
            logger.info("⏭️ IA: Modelos de produção validados.")

        # --- ETAPA 4: OURO (Predição e BigQuery) ---
        if gatilho_ouro:
            logger.info("🏆 OURO: Sincronizando Data Warehouse e SHAP...")
            ouro = CamadaOuroSafeDriver(dev_mode=False) 
            if ouro.executar_predicao_atual():
                stats["status_camadas"]["ouro"] = "✅ (Sincronizado)"
                # Aqui a Ouro poderia retornar a contagem real de predições
                stats["hygiene"]["linhas_ouro"] = stats["hygiene"].get("linhas_out", 0)
            else:
                comunicador.relatar_erro_critico("Camada Ouro", "Erro na sincronização com BigQuery.")
                return
        else:
            logger.info("⏭️ Ouro: Data Warehouse atualizado.")
        
        # --- FINALIZAÇÃO E RELATÓRIO ---
        stats["duracao"] = str(datetime.now() - start_time).split(".")[0]
        
        # Recalcula a taxa de recuperação final para o relatório
        if stats["hygiene"]["linhas_in"] > 0:
            stats["hygiene"]["taxa_recuperacao"] = round(
                (stats["hygiene"]["linhas_out"] / stats["hygiene"]["linhas_in"]) * 100, 1
            )

        logger.info(f"✨ SafeDriver finalizado. Enviando telemetria para o Discord...")
        comunicador.enviar_relatorio_executivo(stats)

    except Exception as e:
        logger.error(f"💥 FALHA CRÍTICA NO ORQUESTRADOR: {e}", exc_info=True)
        comunicador.relatar_erro_critico("Maestro Central", str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maestro SafeDriver")
    parser.add_argument('--force', action='store_true', default=False, help="Forçar reprocessamento total")
    args = parser.parse_args()
    
    orquestrar_pipeline(force_reprocess=args.force)
