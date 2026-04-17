import logging
import holidays
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoCalendario:
    ESTADO = 'SP'
    FUSO_HORARIO = "America/Sao_Paulo"
    DIAS_PAGAMENTO_INICIO = (5, 10)
    DIAS_PAGAMENTO_FIM = (19, 22)

class CalendarioEstrategico:
    def __init__(self, data_referencia=None):
        self.configuracao = ConfiguracaoCalendario()
        self.fuso = ZoneInfo(self.configuracao.FUSO_HORARIO)
        self.hoje = data_referencia if data_referencia else datetime.now(self.fuso)
        self.feriados_estaduais = holidays.BR(state=self.configuracao.ESTADO)

    def verificar_periodo_pagamento(self) -> int:
        dia = self.hoje.day
        inicio = self.configuracao.DIAS_PAGAMENTO_INICIO
        fim = self.configuracao.DIAS_PAGAMENTO_FIM
        
        if (inicio[0] <= dia <= inicio[1]) or (fim[0] <= dia <= fim[1]):
            return 1
        return 0

    def verificar_proximidade_feriado(self) -> int:
        amanha = self.hoje + timedelta(days=1)
        depois_amanha = self.hoje + timedelta(days=2)
        
        datas_criticas = [self.hoje, amanha, depois_amanha]
        
        if any(data in self.feriados_estaduais for data in datas_criticas):
            return 1
        return 0

    def verificar_final_de_semana(self) -> int:
        if self.hoje.weekday() >= 5:
            return 1
        return 0

    def obter_contexto_ia(self) -> dict:
        return {
            "IS_PAGAMENTO": self.verificar_periodo_pagamento(),
            "IS_FERIADO": self.verificar_proximidade_feriado(),
            "IS_FDS": self.verificar_final_de_semana()
        }

    def validar_execucao_automatica(self) -> bool:
        if self.hoje.weekday() == 6:
            logger.info("Execucao autorizada: Ciclo semanal de domingo.")
            return True
        
        if self.verificar_periodo_pagamento() or self.verificar_proximidade_feriado():
            logger.info("Execucao autorizada: Contexto de alta movimentacao financeira ou feriado.")
            return True
        
        logger.info("Execucao postergada: Sem gatilhos estrategicos ativos.")
        return False

if __name__ == "__main__":
    calendario = CalendarioEstrategico()
    logger.info(f"Data de Analise: {calendario.hoje.strftime('%d/%m/%Y')}")
    logger.info(f"Contexto para Modelagem: {calendario.obter_contexto_ia()}")
    logger.info(f"Gatilho de Operacao: {calendario.validar_execucao_automatica()}")
