import os
import logging
import argparse
import sys
import boto3
from botocore.config import Config
from datetime import datetime

from autobot.ingestao_bronze import IngestaoBronze
from autobot.processamento_prata import ProcessadorPrata
from autobot.treinador_ia import TreinadorModelos
from autobot.ia_sincronizacao_ouro import SincronizadorOuro
from autobot.comunicador import ComunicadorDiscord
from autobot.calendario_estrategico import CalendarioEstrategico

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConfiguracaoOrquestrador:
    CHAVES_AMBIENTE = [
        "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
        "BQ_PROJECT_ID", "BQ_SERVICE_ACCOUNT_JSON", "DISCORD_SUCESSO"
    ]
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "").strip()

class OrquestradorPipeline:
    def __init__(self):
        self.configuracao = ConfiguracaoOrquestrador()
        self._validar_ambiente()
        self.comunicador = ComunicadorDiscord()
        self.calendario = CalendarioEstrategico()
        self.cliente_armazenamento = self._inicializar_cliente_armazenamento()

    def _validar_ambiente(self):
        chaves_ausentes = [chave for chave in self.configuracao.CHAVES_AMBIENTE if not os.getenv(chave)]
        if chaves_ausentes:
            logger.error(f"Configuracao incompleta. Variaveis ausentes: {chaves_ausentes}")
            sys.exit(1)
        logger.info("Ambiente validado com exito.")

    def _inicializar_cliente_armazenamento(self):
        return boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )

    def _verificar_existencia_artefato(self, caminho_relativo: str) -> bool:
        try:
            chave_completa = f"datalake/{caminho_relativo}".replace("//", "/")
            self.cliente_armazenamento.head_object(Bucket=self.configuracao.NOME_BUCKET, Key=chave_completa)
            return True
        except Exception:
            return False

    def executar(self, forcar_reprocessamento: bool = False):
        logger.info(f"Iniciando ciclo operacional. Forcar reprocessamento: {forcar_reprocessamento}")
        tempo_inicio = datetime.now()
        
        estatisticas = {
            "status_camadas": {"bronze": "⏭️", "prata": "⏭️", "ia": "⏭️", "ouro": "⏭️"},
            "hygiene": {"taxa_recuperacao": 100, "linhas_ouro": 0, "recuperado_grade": 0},
            "metrics_ia": {"mae": "N/A"},
            "duracao": "0s"
        }

        try:
            execucao_estrategica = self.calendario.validar_execucao_automatica()
            
            bronze = IngestaoBronze()
            novos_dados_bronze = bronze.executar_ingestao_continua(forcar_execucao=forcar_reprocessamento)
            
            if novos_dados_bronze: 
                estatisticas["status_camadas"]["bronze"] = "✅ Novo"

            ano_atual = datetime.now().year
            prata_disponivel = self._verificar_existencia_artefato(f"prata/ssp_consolidada_{ano_atual}.parquet")
            modelos_disponiveis = self._verificar_existencia_artefato("modelos_ml/latest_cat_geral.pkl")
            
            acionar_prata = novos_dados_bronze or forcar_reprocessamento or not prata_disponivel
            acionar_ia = acionar_prata or not modelos_disponiveis
            
            if acionar_prata:
                logger.info("Processando camada Prata.")
                prata = ProcessadorPrata()
                resultado_prata = prata.executar_completo(forcar_execucao=forcar_reprocessamento)
                if resultado_prata:
                    estatisticas["status_camadas"]["prata"] = "✅ Atualizada"
                else:
                    raise RuntimeError("Falha no processamento da camada Prata.")
            
            if acionar_ia:
                logger.info("Ajustando modelos preditivos.")
                treinador = TreinadorModelos()
                if treinador.treinar_modelos():
                    estatisticas["status_camadas"]["ia"] = "✅ Treinado"
                    estatisticas["metrics_ia"].update(treinador.obter_estatisticas())
                else:
                    raise RuntimeError("Falha no treinamento dos modelos.")

            if estatisticas["status_camadas"]["ia"] == "✅ Treinado" or execucao_estrategica or forcar_reprocessamento:
                logger.info("Sincronizando camada Ouro.")
                ouro = SincronizadorOuro()
                if ouro.executar_pipeline_preditivo():
                    estatisticas["status_camadas"]["ouro"] = "✅ Sincronizado"
                    estatisticas["hygiene"]["linhas_ouro"] = "Malha Completa"
                else:
                    raise RuntimeError("Falha na sincronizacao com o BigQuery.")

            estatisticas["duracao"] = str(datetime.now() - tempo_inicio).split(".")[0]
            logger.info(f"Ciclo concluido em {estatisticas['duracao']}.")
            self.comunicador.enviar_relatorio_operacional(estatisticas)

        except Exception as erro:
            logger.error(f"Interrupcao do fluxo principal: {erro}", exc_info=True)
            self.comunicador.reportar_falha_critica("Orquestrador Central", str(erro))
            sys.exit(1)

if __name__ == "__main__":
    analisador = argparse.ArgumentParser()
    analisador.add_argument('--force', action='store_true')
    argumentos = analisador.parse_args()
    
    orquestrador = OrquestradorPipeline()
    orquestrador.executar(forcar_reprocessamento=argumentos.force)
