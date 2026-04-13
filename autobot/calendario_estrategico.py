import pandas as pd
from datetime import datetime, timedelta
import holidays

class CalendarioEstrategico:
    def __init__(self):
        self.hoje = datetime.now()
        self.feriados_br = holidays.BR(state='SP')

    def eh_semana_de_pagamento(self):
        """Pagamento geralmente ocorre entre o 1º e o 5º dia útil + dia 20."""
        dia = self.hoje.day
     
        return (1 <= dia <= 7) or (19 <= dia <= 22)

    def eh_vespera_de_feriado(self):
        """Verifica se amanhã ou depois é feriado (antecipação)."""
        amanha = self.hoje + timedelta(days=1)
        depois_amanha = self.hoje + timedelta(days=2)
        return amanha in self.feriados_br or depois_amanha in self.feriados_br

    def obter_multiplicadores(self):
        """Define o peso extra baseado no contexto atual."""
        pesos = {"comercial": 1.0, "residencial": 1.0, "geral": 1.0}
        
        if self.eh_semana_de_pagamento():
            pesos["comercial"] = 1.25 
            pesos["geral"] = 1.10
            
        if self.eh_vespera_de_feriado():
            pesos["geral"] = 1.20
            pesos["residencial"] = 1.15
            
        return pesos
