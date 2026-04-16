import pandas as pd
from datetime import datetime, timedelta
import holidays
import logging

logger = logging.getLogger(__name__)

class CalendarioEstrategico:
    def __init__(self, data_ref=None):
        # Permitir injeção de data é essencial para fazer Backtesting (testar o passado)
        self.hoje = data_ref if data_ref else datetime.now()
        
        # Feriados oficiais do Estado de São Paulo
        self.feriados_sp = holidays.BR(state='SP')

    def is_pagamento(self):
        """
        Feature ML (0 ou 1). Identifica períodos de injeção de liquidez:
        - 5º dia útil (geralmente entre o dia 5 e 10).
        - Vales e Adiantamentos (dias 19 a 22).
        """
        dia = self.hoje.day
        if (5 <= dia <= 10) or (19 <= dia <= 22):
            return 1
        return 0

    def is_vespera_feriado(self):
        """
        Feature ML (0 ou 1). Identifica a proximidade de feriados.
        Crucial para o modelo detetar esvaziamento residencial e fluxo em autoestradas.
        """
        amanha = self.hoje + timedelta(days=1)
        depois_amanha = self.hoje + timedelta(days=2)
        
        tem_feriado = (self.hoje in self.feriados_sp) or \
                      (amanha in self.feriados_sp) or \
                      (depois_amanha in self.feriados_sp)
        
        return 1 if tem_feriado else 0

    def is_fds(self):
        """Feature ML (0 ou 1). Identifica fins de semana."""
        return 1 if self.hoje.weekday() >= 5 else 0 # 5 = Sábado, 6 = Domingo

    def obter_contexto_ia(self):
        """
        Substitui os antigos 'Multiplicadores'.
        Retorna um dicionário de features puras para injetar diretamente no DataFrame da Camada Ouro.
        """
        contexto = {
            "IS_PAGAMENTO": self.is_pagamento(),
            "IS_FERIADO": self.is_vespera_feriado(),
            "IS_FDS": self.is_fds()
        }
        return contexto

    def deve_rodar_hoje(self):
        """
        Gatilho de DevOps para o GitHub Actions (O Batedor).
        Retorna True se for Domingo OU se houver contexto de alta tensão.
        """
        if self.hoje.weekday() == 6:
            logger.info("DevOps: Domingo detetado. Execução de rotina autorizada.")
            return True
        
        if self.is_pagamento() or self.is_vespera_feriado():
            logger.info("DevOps: Contexto de tensão detetado (Pagamento/Feriado). Execução tática autorizada.")
            return True
        
        logger.info("DevOps: Sem gatilhos estratégicos. Pipeline em repouso.")
        return False

if __name__ == "__main__":
    cal = CalendarioEstrategico()
    print(f"Data Base: {cal.hoje.strftime('%d/%m/%Y')}")
    print(f"Gatilho GitHub Actions: {cal.deve_rodar_hoje()}")
    print(f"Features Injetadas na IA: {cal.obter_contexto_ia()}")
