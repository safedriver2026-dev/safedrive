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
from lightgbm import LGBMRegressor

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
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        # ESCALABILIDADE: Intervalo dinâmico de anos
        self.anos = list(range(2022, datetime.now().year + 1))
        self.agente = {'User-Agent': 'SafeDriver-Industrial-V12'}
        self.webhook = os.environ.get("DISCORD_WEBHOOK")
        self.manifesto_auditoria = {}

    def gerar_hash(self, caminho_ficheiro):
        sha256_hash = hashlib.sha256()
        with open(caminho_ficheiro, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def enviar_discord(self, titulo, mensagem, cor):
        if not self.webhook: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "Monitorização SafeDriver"}
            }]
        }
        requests.post(self.webhook, json=payload, timeout=10)

    def extrair_dados_reais(self, conteudo_excel):
        excel = pd.ExcelFile(io.BytesIO(conteudo_excel))
        for aba in excel.sheet_names:
            df_teste = excel.parse(aba, nrows=10)
            df_teste.columns = [str(c).upper().strip() for c in df_teste.columns]
            if any(k in df_teste.columns for k in ['LATITUDE', 'NATUREZA_APURADA', 'RUBRICA']):
                return excel.parse(aba)
        return None

    def processar_camada_ouro(self, df):
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 1. Classificação de Perfis (Vetor de Exposição)
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_texto'] = df[col_crime].fillna('').astype(str).upper()
        
        df.loc[df['crime_texto'].str.contains('VEÍCULO|MOTO|CARRO|CARGA'), 'perfil'] = 'Motorista'
        df.loc[df['crime_texto'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            local_texto = df[col_local].fillna('').astype(str).upper()
            mask_pedestre = (local_texto.str.contains('VIA PÚBLICA')) & (df['crime_texto'].str.contains('CELULAR|PESSOA'))
            df.loc[mask_pedestre, 'perfil'] = 'Pedestre'
        
        df['peso_severidade'] = df['crime_texto'].apply(lambda x: 15 if 'ROUBO' in x else 2)

        # 2. Filtragem Geográfica
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        
        # 3. Agregação H3 e Modelo Estrela
        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        fato_risco = df.groupby(['h3_index', 'desc_periodo', 'perfil'])['peso_severidade'].sum().reset_index()
        fato_risco['lat'] = fato_risco['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        fato_risco['lon'] = fato_risco['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        fato_risco['perfil_idx'] = fato_risco['perfil'].astype('category').cat.codes
        fato_risco['periodo_idx'] = fato_risco['desc_periodo'].astype('category').cat.codes

        # 4. IA Explicável (SHAP)
        X = fato_risco[['lat', 'lon', 'perfil_idx', 'periodo_idx']]
        y = fato_risco['peso_severidade']
        modelo = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        
        valores_shap = shap.TreeExplainer(modelo).shap_values(X)
        for i, col in enumerate(X.columns):
            fato_risco[f'shap_{col}'] = valores_shap[:, i]
            
        fato_risco['score_predito'] = modelo.predict(X)

        # 5. Exportação e Trilha de Auditoria
        caminho_ouro = self.pastas["ouro"] / "base_final.csv"
        fato_risco.to_csv(caminho_ouro, index=False)
        self.manifesto_auditoria["ouro"] = self.gerar_hash(caminho_ouro)
        
        return len(fato_risco)

    def iniciar(self):
        try:
            acumulado = []
            log_operacional = []
            
            for ano in self.anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    res = requests.get(url, headers=self.agente, timeout=120)
                    if res.status_code == 200:
                        df = self.extrair_dados_reais(res.content)
                        if df is not None:
                            caminho_bruto = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                            df.to_parquet(caminho_bruto)
                            self.manifesto_auditoria[f"bronze_{ano}"] = self.gerar_hash(caminho_bruto)
                            acumulado.append(df)
                            log_operacional.append(f"✅ {ano}: {len(df)} registos processados.")
                except:
                    log_operacional.append(f"⚠️ {ano}: Erro na captura.")

            if not acumulado: raise Exception("Bases de dados inacessíveis.")

            total = self.processar_camada_ouro(pd.concat(acumulado))
            
            # Grava Manifesto de Auditoria
            with open(self.pastas["auditoria"] / "manifesto.json", "w") as f:
                json.dump(self.manifesto_auditoria, f, indent=4)

            # Relatórios Discord
            self.enviar_discord("📊 Relatório Operacional", "\n".join(log_operacional), 3447003)
            self.enviar_discord("🚀 Relatório Executivo", f"Pipeline concluído com sucesso.\n📍 Áreas Geográficas: {total}\n🔒 Trilha de Auditoria: Gerada.", 3066993)

        except Exception as e:
            self.enviar_discord("🚨 Erro Crítico", f"Falha no processamento: `{str(e)}`", 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
