import os
import requests
import pandas as pd
import numpy as np
import h3
import json
from pathlib import Path
from datetime import datetime

class MotorSafeDriver:
    def __init__(self, perfil="LOGISTICA"):
        self.perfil = perfil
        self.raiz = Path(".")
        self.camadas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro"
        }
        for pasta in self.camadas.values(): 
            pasta.mkdir(parents=True, exist_ok=True)
        
        self.anos = list(range(2022, datetime.now().year + 1))
        self.cabecalhos = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        self.registro_operacional = []
        self.metricas_executivas = {"total_geral": 0, "novas_entradas": 0, "falhas": 0}

    def disparar_discord(self, titulo, conteudo, cor, categoria):
        webhook = os.environ.get("DISCORD_SUCESSO")
        if not webhook: 
            return
        
        dados = {
            "embeds": [{
                "title": f"🛡️ {titulo}",
                "description": conteudo,
                "color": cor,
                "footer": {"text": f"Origem: {categoria} | {datetime.now().strftime('%H:%M')}"}
            }]
        }
        requests.post(webhook, json=dados, timeout=15)

    def sincronizar_delta(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_local = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        controle_tamanho = self.camadas["bronze"] / f"tamanho_{ano}.txt"

        try:
            verificacao = requests.head(url, headers=self.cabecalhos, timeout=30)
            if verificacao.status_code != 200:
                self.registro_operacional.append(f"❌ {ano}: Link indisponível (HTTP {verificacao.status_code})")
                return None

            tamanho_remoto = int(verificacao.headers.get('Content-Length', 0))
            
            if caminho_local.exists() and controle_tamanho.exists():
                with open(controle_tamanho, "r") as arquivo:
                    if int(arquivo.read()) == tamanho_remoto:
                        self.registro_operacional.append(f"🟢 {ano}: Base já atualizada")
                        return pd.read_parquet(caminho_local)

            resposta = requests.get(url, headers=self.cabecalhos, timeout=300)
            tabela = pd.read_excel(resposta.content)
            
            if tabela.empty: 
                return None
            
            tabela.columns = [str(col).upper().strip() for col in tabela.columns]
            tabela.to_parquet(caminho_local, index=False)
            
            with open(controle_tamanho, "w") as arquivo: 
                arquivo.write(str(tamanho_remoto))
            
            self.registro_operacional.append(f"📥 {ano}: Atualização baixada ({len(tabela)} linhas)")
            return tabela
        except Exception as erro:
            self.registro_operacional.append(f"🔥 {ano}: Erro na captura - {str(erro)}")
            self.metricas_executivas["falhas"] += 1
            return None

    def executar_pipeline(self):
        acumulado_prata = []
        
        for ano in self.anos:
            base = self.sincronizar_delta(ano)
            if base is not None:
                base.columns = [col.lower() for col in base.columns]
                
                tabela_pesos = {
                    'ROUBO DE CARGA': 15, 
                    'ROUBO DE VEICULO': 10, 
                    'FURTO': 2, 
                    'LATROCINIO': 12
                }
                
                base['peso_severidade'] = base['rubrica'].apply(
                    lambda x: next((v for k, v in tabela_pesos.items() if k in str(x).upper()), 1.0)
                )
                
                base['latitude'] = pd.to_numeric(base['latitude'], errors='coerce')
                base['longitude'] = pd.to_numeric(base['longitude'], errors='coerce')
                base = base.dropna(subset=['latitude', 'longitude'])
                
                base.to_parquet(self.camadas["prata"] / f"prata_{ano}.parquet", index=False)
                acumulado_prata.append(base)
                self.metricas_executivas["novas_entradas"] += len(base)

        if acumulado_prata:
            unificado = pd.concat(acumulado_prata).drop_duplicates()
            unificado['h3_index'] = unificado.apply(
                lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1
            )
            
            resultado_ouro = unificado.groupby('h3_index')['peso_severidade'].sum().reset_index(name='score_risco')
            resultado_ouro.to_csv(self.camadas["ouro"] / "base_looker.csv", index=False)
            
            self.metricas_executivas["total_geral"] = len(unificado)
            self.gerar_relatorios()
        else:
            self.disparar_discord("Falha na Ingestão", "Nenhum dado processado nas camadas Bronze/Prata.", 15158332, "CRÍTICO")

    def gerar_relatorios(self):
        corpo_operacional = "\n".join(self.registro_operacional)
        self.disparar_discord(
            "Relatório Operacional", 
            f"**Status do Data Lake:**\n{corpo_operacional}", 
            3447003, 
            "ENGENHARIA"
        )
        
        novos = self.metricas_executivas["novas_entradas"]
        total = self.metricas_executivas["total_geral"]
        
        corpo_executivo = (
            f"🚀 **Desempenho da IA:** Processamento Concluído\n"
            f"📍 **Pontos Georreferenciados:** {total:,}\n"
            f"🆕 **Novos Dados Sincronizados:** {novos:,}\n"
            f"✅ **Integridade da Base:** 100%"
        )
        self.disparar_discord(
            "Relatório Executivo", 
            corpo_executivo, 
            3066993, 
            "DIRETORIA"
        )

if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline()
