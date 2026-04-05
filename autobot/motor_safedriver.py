import os
import io
import json
import requests
import pandas as pd
import numpy as np
import h3
import shap
import matplotlib.pyplot as plt
import hashlib
from pathlib import Path
from datetime import datetime
from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria",
            "docs": self.raiz / "documentacao"
        }
        for pasta_caminho in self.pastas.values():
            pasta_caminho.mkdir(parents=True, exist_ok=True)
            
        ano_base = 2022
        ano_limite = datetime.now().year
        self.anos_processamento = list(range(ano_base, ano_limite + 1))
        self.agente_requisicao = {'User-Agent': 'SafeDriver-Industrial-V17-ColdStart'}
        self.webhook_monitor = os.environ.get("DISCORD_WEBHOOK")
        self.registro_auditoria = {}

    def gerar_assinatura_arquivo(self, caminho):
        sha256 = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""):
                sha256.update(bloco)
        return sha256.hexdigest()

    def reportar_discord(self, titulo, mensagem, cor_hex):
        if not self.webhook_monitor: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor_hex,
                "timestamp": datetime.now().isoformat()
            }]
        }
        requests.post(self.webhook_monitor, json=payload, timeout=10)

    def identificar_planilha_dados(self, conteudo_binario):
        arquivo_excel = pd.ExcelFile(io.BytesIO(conteudo_binario))
        for nome_aba in arquivo_excel.sheet_names:
            df_amostra = arquivo_excel.parse(nome_aba, nrows=25)
            df_amostra.columns = [str(col).upper().strip() for col in df_amostra.columns]
            if any(chave in df_amostra.columns for chave in ['LATITUDE', 'NATUREZA_APURADA', 'RUBRICA']):
                return arquivo_excel.parse(nome_aba)
        return None

    def executar_ensemble_ia(self, dataframe_consolidado):
        dataframe_consolidado.columns = [str(c).lower().strip() for c in dataframe_consolidado.columns]
        
        dataframe_consolidado['perfil_usuario'] = 'Geral'
        coluna_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in dataframe_consolidado.columns), 'rubrica')
        dataframe_consolidado['crime_normalizado'] = dataframe_consolidado[coluna_crime].fillna('').astype(str).upper()
        
        dataframe_consolidado.loc[dataframe_consolidado['crime_normalizado'].str.contains('VEÍCULO|MOTO|CARGA|AUTO|CONDUZIR'), 'perfil_usuario'] = 'Motorista'
        dataframe_consolidado.loc[dataframe_consolidado['crime_normalizado'].str.contains('BICICLETA|BIKE'), 'perfil_usuario'] = 'Ciclista'
        
        coluna_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in dataframe_consolidado.columns), 'descr_tipolocal')
        if coluna_local in dataframe_consolidado.columns:
            local_normalizado = dataframe_consolidado[coluna_local].fillna('').astype(str).upper()
            filtro_pedestre = (local_normalizado.str.contains('VIA PÚBLICA|RUA|AVENIDA')) & (dataframe_consolidado['crime_normalizado'].str.contains('CELULAR|PESSOA|TRANSEUNTE'))
            dataframe_consolidado.loc[filtro_pedestre, 'perfil_usuario'] = 'Pedestre'
        
        dataframe_consolidado['peso_severidade'] = dataframe_consolidado['crime_normalizado'].apply(lambda x: 15 if 'ROUBO' in x else 2)

        dataframe_consolidado['latitude'] = pd.to_numeric(dataframe_consolidado['latitude'], errors='coerce').fillna(0)
        dataframe_consolidado['longitude'] = pd.to_numeric(dataframe_consolidado['longitude'], errors='coerce').fillna(0)
        dataframe_consolidado = dataframe_consolidado[(dataframe_consolidado['latitude'] != 0) & (dataframe_consolidado['longitude'] != 0)].copy()
        
        if dataframe_consolidado.empty: raise ValueError("Processamento interrompido: Ausencia de coordenadas validas")

        dataframe_consolidado['h3_index'] = dataframe_consolidado.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        tabela_fato = dataframe_consolidado.groupby(['h3_index', 'desc_periodo', 'perfil_usuario'])['peso_severidade'].sum().reset_index()
        tabela_fato['lat'] = tabela_fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        tabela_fato['lon'] = tabela_fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        tabela_fato['perfil_cod'] = tabela_fato['perfil_usuario'].astype('category').cat.codes
        tabela_fato['periodo_cod'] = tabela_fato['desc_periodo'].astype('category').cat.codes

        variaveis_x = tabela_fato[['lat', 'lon', 'perfil_cod', 'periodo_cod']]
        alvo_y = tabela_fato['peso_severidade']
        
        lgbm = LGBMRegressor(n_estimators=100, verbose=-1).fit(variaveis_x, alvo_y)
        catb = CatBoostRegressor(iterations=100, silent=True).fit(variaveis_x, alvo_y)
        knn = KNeighborsRegressor(n_neighbors=5).fit(variaveis_x, alvo_y)

        explainer_shap = shap.TreeExplainer(lgbm)
        valores_shap = explainer_shap.shap_values(variaveis_x)
        for i, col_nome in enumerate(variaveis_x.columns):
            tabela_fato[f'influencia_{col_nome}'] = valores_shap[:, i]
            
        tabela_fato['score_final'] = (lgbm.predict(variaveis_x) * 0.4 + catb.predict(variaveis_x) * 0.4 + knn.predict(variaveis_x) * 0.2)

        caminho_saida_ouro = self.pastas["ouro"] / "base_inteligencia.csv"
        tabela_fato.to_csv(caminho_saida_ouro, index=False)
        self.registro_auditoria["camada_ouro"] = self.gerar_assinatura_arquivo(caminho_saida_ouro)
        
        return len(tabela_fato)

    def iniciar_pipeline(self):
        try:
            lista_dataframes = []
            log_detalhado = []
            
            for ano in self.anos_processamento:
                url_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_local_bronze = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                
                try:
                    resposta_ssp = requests.get(url_ssp, headers=self.agente_requisicao, timeout=90)
                    if resposta_ssp.status_code == 200:
                        df_extraido = self.identificar_planilha_dados(resposta_ssp.content)
                        if df_extraido is not None:
                            df_extraido.to_parquet(caminho_local_bronze)
                            self.registro_auditoria[f"bronze_{ano}"] = self.gerar_assinatura_arquivo(caminho_local_bronze)
                            lista_dataframes.append(df_extraido)
                            log_detalhado.append(f"✅ {ano}: Download e Ingestao OK")
                except:
                    if caminho_local_bronze.exists():
                        df_cache = pd.read_parquet(caminho_local_bronze)
                        lista_dataframes.append(df_cache)
                        log_detalhado.append(f"📦 {ano}: Recuperado do Cache Local")

            if not lista_dataframes: raise Exception("Falha de infraestrutura: Nenhuma aba de dados localizada em 2022-2026")

            total_h3 = self.executar_ensemble_ia(pd.concat(lista_dataframes))
            
            with open(self.pastas["auditoria"] / "manifesto.json", "w") as f_audit:
                json.dump(self.registro_auditoria, f_audit, indent=4)

            self.reportar_discord("SafeDriver: Monitor Operacional", "\n".join(log_detalhado), 3447003)
            self.reportar_discord("SafeDriver: Monitor Executivo", f"Areas: {total_h3}\nEnsemble: LGBM/CatB/KNN\nAuditoria: Ativa", 3066993)

        except Exception as erro_fatal:
            self.reportar_discord("SafeDriver: Alerta de Sistema", str(erro_fatal), 15158332)
            raise erro_fatal

if __name__ == "__main__":
    MotorSafeDriver().iniciar_pipeline()
