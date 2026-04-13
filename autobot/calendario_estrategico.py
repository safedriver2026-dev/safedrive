import pandas as pd
from datetime import datetime, timedelta
import holidays
import logging

logger = logging.getLogger(__name__)

class CalendarioEstrategico:
    def __init__(self):
        # Define a data de hoje (ou pode ser injetada para testes)
        self.hoje = datetime.now()
        # Feriados oficiais de São Paulo (Estaduais e Nacionais)
        self.feriados_sp = holidays.BR(state='SP')

    def eh_semana_de_pagamento(self):
        """
        Identifica os períodos de maior circulação de dinheiro:
        - Início do mês (1 ao 7): Salários do 5º dia útil.
        - Meio do mês (19 ao 22): Adiantamentos e vales.
        """
        dia = self.hoje.day
        return (1 <= dia <= 7) or (19 <= dia <= 22)

    def eh_vespera_de_feriado(self):
        """
        Verifica se amanhã ou depois de amanhã é feriado.
        Essencial para antecipar o aumento de risco em rotas de saída e áreas residenciais.
        """
        amanha = self.hoje + timedelta(days=1)
        depois_amanha = self.hoje + timedelta(days=2)
        
        # Verifica se hoje, amanhã ou depois consta na lista de feriados
        return (self.hoje in self.feriados_sp) or \
               (amanha in self.feriados_sp) or \
               (depois_amanha in self.feriados_sp)

    def obter_multiplicadores(self):
        """
        Retorna dicionário com pesos de ajuste para a Predição de Risco.
        Estes valores multiplicam o score bruto da IA na Camada Ouro.
        """
        pesos = {
            "comercial": 1.0,     # Impacto em TOTAL_NAO_RES_H3
            "residencial": 1.0,   # Impacto em INDICE_RESIDENCIAL
            "geral": 1.0          # Multiplicador final do Score
        }
        
        # Ajuste para Semana de Pagamento (Foco em áreas comerciais/bancárias)
        if self.eh_semana_de_pagamento():
            pesos["comercial"] = 1.25  # +25% de risco em áreas comerciais
            pesos["geral"] = 1.10      # +10% de elevação basal
            logger.info("Contexto: Semana de Pagamento detectada.")

        # Ajuste para Vésperas de Feriado (Foco em residências vazias e movimento geral)
        if self.eh_vespera_de_feriado():
            pesos["residencial"] = 1.20 # +20% de risco em áreas residenciais
            pesos["geral"] = 1.15       # +15% de elevação basal
            logger.info("Contexto: Proximidade de Feriado detectada.")
            
        return pesos

    def deve_rodar_hoje(self):
        """
        Lógica de decisão para o GitHub Actions (Batedor).
        Retorna True se for Domingo OU se houver contexto estratégico.
        """
        # 1. Ciclo Padrão: Domingo (weekday 6)
        if self.hoje.weekday() == 6:
            logger.info("Decisão: Domingo detectado. Execução semanal obrigatória.")
            return True
        
        # 2. Ciclo de Antecipação: Pagamento ou Feriado
        if self.eh_semana_de_pagamento() or self.eh_vespera_de_feriado():
            logger.info("Decisão: Contexto estratégico detectado. Execução antecipada autorizada.")
            return True
        
        # Caso contrário, mantém o pipeline em repouso
        logger.info("Decisão: Sem gatilhos estratégicos para hoje. Pipeline em repouso.")
        return False

if __name__ == "__main__":
    # Teste rápido de console
    cal = CalendarioEstrategico()
    print(f"Data: {cal.hoje.strftime('%d/%m/%Y')}")
    print(f"Deve rodar hoje? {cal.deve_rodar_hoje()}")
    print(f"Multiplicadores: {cal.obter_multiplicadores()}")
