import io
import os
import json
import math
import hashlib
import logging
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import pygeohash as gh
import requests
import firebase_admin
import xgboost as xgb

from firebase_admin import credentials, firestore
from prophet import Prophet
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from autobot.config import (
    CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, SUBTIPOS_LOCAL_PERMITIDOS, LIMITES_SP,
    ESQUEMA_RAW_CANONICO, ESQUEMA_TRUSTED, COLUNAS_REFINED_EVENTOS,
    PALAVRAS_CHAVE_PERFIL, MAPA_SEMANTICO_COLUNAS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class MotorSafeDriver:
    def __init__(self, habilitar_firestore=True, forcar_recarga=False):
        self.data_execucao = pd.Timestamp(datetime.now().date())
        self.janela_inicio = self.data_execucao - pd.Timedelta(days=730)
        self.periodo_historico = range(self.janela_inicio.year, self.data_execucao.year + 1)
        self.forcar_recarga = forcar_recarga
        
        self.sessao_web = self._criar_sessao_resiliente()
        self.banco_nuvem = self._estabelecer_conexao_nuvem() if habilitar_firestore else None
        
        self.auditoria = {
            "timestamp": str(datetime.now()), "volume_raw": 0, "volume_trusted": 0, 
            "volume_refined": 0, "falhas_integridade": 0, "malha_motorista": 0, 
            "malha_motociclista": 0, "malha_pedestre": 0, "malha_ciclista": 0, 
            "documentos_sincronizados": 0, "documentos_atualizados": 0, 
            "novos_dados_baixados": False, "mae_modelo": 0.0, "rmse_modelo": 0.0
        }
        
        for pasta in ['raw', 'trusted', 'refined', 'metadata', 'reports']:
            os.makedirs(f'datalake/{pasta}', exist_ok=True)

    def _estabelecer_conexao_nuvem(self):
        chave = os.environ.get('FIREBASE_JSON')
        if not chave or not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave))
            firebase_admin.initialize_app(credenciais)
        return firestore.client()

    def _criar_sessao_resiliente(self):
        sessao = requests.Session()
        retentativas = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        adaptador = HTTPAdapter(max_retries=retentativas)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        sessao.headers.update({'User-Agent': 'Mozilla/5.0 SafeDriver/8.0 (MLOps)'})
        return sessao

    def _notificar_sucesso(self):
        endereco_webhook = os.environ.get('DISCORD_SUCESSO')
        if not endereco_webhook: return
        status_extracao = "Novos arquivos detectados e processados." if self.auditoria['novos_dados_baixados'] else "Processamento incremental MLOps concluido."
        pacote_dados = {
            "embeds": [{
                "title": "Pipeline SafeDriver MLOps Executado",
                "description": status_extracao,
                "color": 3066993,
                "fields": [
                    {"name": "Volumetria", "value": f"RAW: {self.auditoria['volume_raw']:,}\nTRUSTED: {self.auditoria['volume_trusted']:,}\nREFINED: {self.auditoria['volume_refined']:,}", "inline": False},
                    {"name": "Performance do Modelo", "value": f"MAE: {self.auditoria['mae_modelo']}\nRMSE: {self.auditoria['rmse_modelo']}", "inline": False},
                    {"name": "Firestore Delta Sync", "value": f"Lotes avaliados: {self.auditoria['documentos_sincronizados']:,}\nAlterados: {self.auditoria['documentos_atualizados']:,}", "inline": False}
                ]
            }]
        }
        self.sessao_web.post(endereco_webhook, json=pacote_dados)

    def _notificar_erro(self, diagnostico_falha):
        endereco_webhook = os.environ.get('DISCORD_ERRO')
        if not endereco_webhook: return
        pacote_dados = {"embeds": [{"title": "Interrupcao Operacional", "color": 15158332, "fields": [{"name": "Diagnostico", "value": diagnostico_falha, "inline": False}]}]}
        requests.post(endereco_webhook, json=pacote_dados)

    def _higienizar_texto(self, texto):
        if pd.isna(texto): return ""
        norm = unicodedata.normalize('NFKD', str(texto))
        return "".join([c for c in norm if not unicodedata.combining(c)]).upper().strip()

    def _verificar_atualizacao(self, url, ano):
        meta_path = f"datalake/metadata/tamanho_{ano}.json"
        try:
            head = self.sessao_web.head(url, timeout=30, allow_redirects=True)
            tamanho_nuvem = int(head.headers.get('Content-Length', 0))
            if os.path.exists(meta_path) and not self.forcar_recarga:
                with open(meta_path, 'r') as f:
                    if json.load(f).get('tamanho') == tamanho_nuvem: return False, tamanho_nuvem
            return True, tamanho_nuvem
        except: return True, 0

    def _coalescer_colunas(self, df, nome_canonico, aliases):
        aliases_norm = [self._higienizar_texto(a) for a in aliases]
        colunas_existentes = [col for col in aliases_norm if col in df.columns]
        if not colunas_existentes: return df
        serie_final = pd.Series(np.nan, index=df.index)
        if nome_canonico in df.columns: serie_final = serie_final.combine_first(df[nome_canonico])
        for coluna in colunas_existentes: serie_final = serie_final.combine_first(df[coluna])
        df[nome_canonico] = serie_final
        return df

    def _construir_raw_operacional(self, df_raw, ano):
        df = df_raw.copy()
        df.columns = [self._higienizar_texto(c) for c in df.columns]
        
        for canonico, aliases in MAPA_SEMANTICO_COLUNAS.items():
            df = self._coalescer_colunas(df, canonico, aliases)

        for col in ESQUEMA_RAW_CANONICO.keys():
            if col not in df.columns and col != "ANO_BASE":
                df[col] = pd.Series(dtype='object')

        df["DESCR_TIPOLOCAL"] = df["DESCR_TIPOLOCAL"].astype(str).replace('nan', '')
        df["DESCR_SUBTIPOLOCAL"] = df["DESCR_SUBTIPOLOCAL"].astype(str).replace('nan', '')
        
        df["DESCR_SUBTIPOLOCAL"] = df["DESCR_SUBTIPOLOCAL"].replace('', np.nan).combine_first(df["DESCR_TIPOLOCAL"].replace('', np.nan)).fillna('')
        mascara_subtipo = df["DESCR_SUBTIPOLOCAL"].map(self._higienizar_texto).isin(SUBTIPOS_LOCAL_PERMITIDOS)
        df.loc[(df["DESCR_TIPOLOCAL"] == '') & mascara_subtipo, "DESCR_TIPOLOCAL"] = "VIA PUBLICA"

        df = df[list(ESQUEMA_RAW_CANONICO.keys() - {"ANO_BASE"})].copy()
        df.to_parquet(f"datalake/raw/ssp_{ano}.parquet", index=False)
        return df

    def _processar_trusted_refined(self, df_raw, ano):
        df = df_raw.copy()
        for col in ESQUEMA_TRUSTED.keys():
            if col not in df.columns and col != 'ANO_BASE': df[col] = np.nan

        vol_inicial = len(df)
        df['ANO_BASE'] = str(ano)
        
        for col, tipo in ESQUEMA_TRUSTED.items():
            if col not in df.columns: continue
            if tipo == 'string': df[col] = df[col].astype(str).replace('nan', '')
            elif tipo == 'float': df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors="coerce")
            elif tipo == 'datetime': df[col] = pd.to_datetime(df[col], errors='coerce')
            elif tipo == 'int': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        df['DATA_OCORRENCIA_BO'] = df['DATA_OCORRENCIA_BO'].dt.normalize()
        m_tempo = df['DATA_OCORRENCIA_BO'].between(self.janela_inicio, self.data_execucao)
        
        m_geo = (df['LATITUDE'].notna() & df['LONGITUDE'].notna() &
                 (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0) &
                 df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
                 df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1]))
        
        df_trusted = df[m_tempo & m_geo].copy()
        df_trusted = df_trusted[list(ESQUEMA_TRUSTED.keys())]
        
        self.auditoria['falhas_integridade'] += (vol_inicial - len(df_trusted))
        self.auditoria['volume_trusted'] += len(df_trusted)

        if 'DESCR_TIPOLOCAL' not in df_trusted.columns: df_trusted['DESCR_TIPOLOCAL'] = 'VIA PUBLICA'

        m_negocio = (df_trusted['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys()) &
                     df_trusted['DESCR_TIPOLOCAL'].isin(TIPOS_LOCAL_PERMITIDOS) &
                     df_trusted['DESCR_SUBTIPOLOCAL'].isin(SUBTIPOS_LOCAL_PERMITIDOS))
        
        df_refined = df_trusted[m_negocio][COLUNAS_REFINED_EVENTOS].copy()
        self.auditoria['volume_refined'] += len(df_refined)
        
        return df_trusted, df_refined

    def _inferir_perfil_contextual(self, linha):
        perfis = set()
        txt = f"{linha.get('NATUREZA_APURADA','')} {linha.get('DESCR_CONDUTA','')} {linha.get('DESCR_SUBTIPOLOCAL','')} {linha.get('RUBRICA','')}".upper()
        for p, palavras in PALAVRAS_CHAVE_PERFIL.items():
            if any(pal in txt for pal in palavras): perfis.add(p)
        if not perfis:
            perfis.update(CATALOGO_CRIMES.get(linha.get('NATUREZA_APURADA'), {}).get('perfis', []))
        return list(perfis) if perfis else ['Indefinido']

    def _classificar_turno(self, hora_str):
        try:
            h = int(str(hora_str).split(':')[0])
            if 0 <= h < 6: return 'Madrugada'
            if 6 <= h < 12: return 'Manha'
            if 12 <= h < 18: return 'Tarde'
            return 'Noite'
        except: return 'Noite'

    def _construir_features_preditivas(self, df_refinado):
        df = df_refinado.copy()
        df['perfis_afetados'] = df.apply(self._inferir_perfil_contextual, axis=1)
        df = df.explode('perfis_afetados').dropna(subset=['perfis_afetados'])
        df['geohash'] = [gh.encode(lat, lon, precision=7) for lat, lon in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(self._classificar_turno)
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        
        painel = df.groupby(['geohash', 'perfis_afetados', 'turno', 'DATA_OCORRENCIA_BO'], as_index=False).agg(
            target_dia=('peso', 'sum'), lat=('LATITUDE', 'mean'), lon=('LONGITUDE', 'mean')
        ).sort_values(['geohash', 'perfis_afetados', 'turno', 'DATA_OCORRENCIA_BO'])

        chaves = ['geohash', 'perfis_afetados', 'turno']
        painel['lag_7d'] = painel.groupby(chaves)['target_dia'].shift(7).fillna(0)
        painel['lag_14d'] = painel.groupby(chaves)['target_dia'].shift(14).fillna(0)
        painel['media_30d'] = painel.groupby(chaves)['target_dia'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0)
        
        painel['target_futuro_7d'] = painel.groupby(chaves)['target_dia'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        
        serie_macro = painel.groupby('DATA_OCORRENCIA_BO')['target_dia'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'target_dia': 'y'})
        modelo_tendencia = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(serie_macro)
        projecao = modelo_tendencia.predict(modelo_tendencia.make_future_dataframe(periods=14))[['ds', 'yhat']]
        
        painel = painel.merge(projecao, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
        painel['fator_tendencia'] = painel['yhat'] / max(painel['yhat'].mean(), 1.0)
        
        self.projecao_prophet = projecao
        return painel

    def _treinar_modelo(self, painel):
        corte_treino = painel['DATA_OCORRENCIA_BO'].max() - pd.Timedelta(days=7)
        corte_val = corte_treino - pd.Timedelta(days=30)
        
        painel_valido = painel.dropna(subset=['target_futuro_7d']).copy()
        treino = painel_valido[painel_valido['DATA_OCORRENCIA_BO'] <= corte_val]
        teste = painel_valido[painel_valido['DATA_OCORRENCIA_BO'] > corte_val]
        
        features = ['lat', 'lon', 'lag_7d', 'lag_14d', 'media_30d', 'fator_tendencia']
        
        enc_p = LabelEncoder().fit(painel['perfis_afetados'])
        enc_t = LabelEncoder().fit(painel['turno'])
        
        for df_ in [treino, teste, painel]:
            df_.loc[:, 'p_enc'] = enc_p.transform(df_['perfis_afetados'])
            df_.loc[:, 't_enc'] = enc_t.transform(df_['turno'])
            
        features.extend(['p_enc', 't_enc'])

        modelo = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        modelo.fit(treino[features], treino['target_futuro_7d'])
        
        if not teste.empty:
            previsoes = np.clip(modelo.predict(teste[features]), 0, None)
            self.auditoria['mae_modelo'] = round(float(mean_absolute_error(teste['target_futuro_7d'], previsoes)), 3)
            self.auditoria['rmse_modelo'] = round(float(math.sqrt(mean_squared_error(teste['target_futuro_7d'], previsoes))), 3)

        modelo.fit(painel_valido[features], painel_valido['target_futuro_7d'])

        futuro_grid = painel.sort_values('DATA_OCORRENCIA_BO').groupby(['geohash', 'perfis_afetados', 'turno']).tail(1).copy()
        futuro_grid['lag_7d'] = futuro_grid['target_dia']
        futuro_grid['lag_14d'] = futuro_grid.groupby(['geohash', 'perfis_afetados', 'turno'])['target_dia'].shift(7).fillna(0)
        futuro_grid['fator_tendencia'] = self.projecao_prophet.iloc[-1]['yhat'] / max(self.projecao_prophet['yhat'].mean(), 1.0)
        
        futuro_grid['score_preditivo'] = np.clip(modelo.predict(futuro_grid[features]), 0, None)
        
        escala = max(futuro_grid['score_preditivo'].quantile(0.95), 1.0)
        futuro_grid['score_normalizado'] = ((futuro_grid['score_preditivo'] / escala) * 10).clip(0.5, 10.0).round(2)
        futuro_grid['routing_penalty'] = (1.0 + (futuro_grid['score_normalizado'] * 0.20)).round(2)

        self.auditoria['malha_motorista'] = len(futuro_grid[futuro_grid['perfis_afetados'] == 'Motorista'])
        self.auditoria['malha_motociclista'] = len(futuro_grid[futuro_grid['perfis_afetados'] == 'Motociclista'])
        self.auditoria['malha_pedestre'] = len(futuro_grid[futuro_grid['perfis_afetados'] == 'Pedestre'])
        self.auditoria['malha_ciclista'] = len(futuro_grid[futuro_grid['perfis_afetados'] == 'Ciclista'])

        return futuro_grid[['geohash', 'perfis_afetados', 'turno', 'score_normalizado', 'routing_penalty']]

    def _sincronizacao_firestore(self, malha):
        colecao = self.banco_nuvem.collection('niveis_risco')
        documentos_atuais = {doc.id: doc.to_dict().get('hash_registro') for doc in colecao.stream()}
        
        lote = self.banco_nuvem.batch()
        operacoes = 0
        ids_vivos = set()

        self.auditoria['documentos_sincronizados'] = len(malha)

        for _, linha in malha.iterrows():
            doc_id = f"{linha['geohash']}_{linha['perfis_afetados']}_{linha['turno']}"
            ids_vivos.add(doc_id)
            
            payload = {
                'score': round(float(linha['score_normalizado']), 2),
                'routing_penalty': round(float(linha['routing_penalty']), 2),
                'geohash': linha['geohash'], 
                'geohash_prefix_4': linha['geohash'][:4],
                'perfil': linha['perfis_afetados'],
                'periodo': linha['turno']
            }
            hash_payload = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
            payload['hash_registro'] = hash_payload
            payload['ultima_atualizacao'] = firestore.SERVER_TIMESTAMP
            
            if doc_id not in documentos_atuais or documentos_atuais[doc_id] != hash_payload:
                lote.set(colecao.document(doc_id), payload, merge=True)
                operacoes += 1
                self.auditoria['documentos_atualizados'] += 1
                
                if operacoes >= 450: 
                    lote.commit()
                    lote = self.banco_nuvem.batch()
                    operacoes = 0

        ids_obsoletos = set(documentos_atuais.keys()) - ids_vivos
        for doc_id in ids_obsoletos:
            lote.delete(colecao.document(doc_id))
            operacoes += 1
            if operacoes >= 450:
                lote.commit()
                lote = self.banco_nuvem.batch()
                operacoes = 0

        if operacoes > 0:
            lote.commit()

    def _gerar_documentacao_runbook(self):
        data_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho_md = f"datalake/reports/runbook_{data_str}.md"
        
        conteudo = f"""# SafeDriver Data & AI Pipeline Runbook
**Data da Execucao:** {self.auditoria['timestamp']}

## 1. Integridade do Pipeline (Data Engineering)
- **Registros Ingeridos (RAW):** {self.auditoria['volume_raw']:,}
- **Registros Higienizados (TRUSTED):** {self.auditoria['volume_trusted']:,}
- **Eventos Analiticos (REFINED):** {self.auditoria['volume_refined']:,}
- **Perda por Qualidade (Data Drops):** {self.auditoria['falhas_integridade']:,}
- **Status da Fonte (SSP):** {'Novos dados consumidos' if self.auditoria['novos_dados_baixados'] else 'Nenhuma mutacao detectada (Cache utilizado)'}

## 2. Modelagem Preditiva (Foresight AI)
- **MAE (Mean Absolute Error):** {self.auditoria['mae_modelo']}
- **RMSE (Root Mean Squared Error):** {self.auditoria['rmse_modelo']}
- **Janela de Contexto:** {self.janela_inicio.date()} ate {self.data_execucao.date()}

## 3. Dispersao Geografica do Risco (Output Size)
- **Motorista:** {self.auditoria['malha_motorista']:,} quadrantes ativados
- **Motociclista:** {self.auditoria['malha_motociclista']:,} quadrantes ativados
- **Ciclista:** {self.auditoria['malha_ciclista']:,} quadrantes ativados
- **Pedestre:** {self.auditoria['malha_pedestre']:,} quadrantes ativados

## 4. Estado da Nuvem (Firebase Sync)
- **Documentos Verificados:** {self.auditoria['documentos_sincronizados']:,}
- **Mutacoes Escritas (Delta Update):** {self.auditoria['documentos_atualizados']:,}
"""
        with open(caminho_md, 'w', encoding='utf-8') as f:
            f.write(conteudo)

    def executar_pipeline_completo(self):
        df_master = pd.DataFrame()
        try:
            for ano in self.periodo_historico:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_raw = f'datalake/raw/ssp_{ano}.parquet'
                
                baixar, tamanho = self._verificar_atualizacao(url, ano)
                
                if baixar or not os.path.exists(caminho_raw):
                    res = self.sessao_web.get(url, timeout=120)
                    if res.status_code != 200: raise ConnectionError(f"HTTP {res.status_code}")
                    leitura_previa = pd.read_excel(io.BytesIO(res.content), nrows=50, header=None)
                    linha_cabecalho = next((i for i, row in leitura_previa.iterrows() if any(t in [self._higienizar_texto(str(c)) for c in row.values] for t in ['NUM_BO', 'LATITUDE', 'NATUREZA_APURADA'])), None)
                    if linha_cabecalho is None: raise ValueError("Header ausente.")
                    
                    df_temp = pd.read_excel(io.BytesIO(res.content), skiprows=linha_cabecalho, dtype=str)
                    df_temp = self._construir_raw_operacional(df_temp, ano)
                    with open(f"datalake/metadata/tamanho_{ano}.json", 'w') as f: json.dump({'tamanho': tamanho}, f)
                    self.auditoria['novos_dados_baixados'] = True
                else:
                    df_temp = pd.read_parquet(caminho_raw)
                    df_temp = self._construir_raw_operacional(df_temp, ano)

                self.auditoria['volume_raw'] += len(df_temp)
                df_trusted, df_refined = self._processar_trusted_refined(df_temp, ano)
                df_trusted.to_parquet(f'datalake/trusted/ssp_trusted_{ano}.parquet', index=False)
                
                if not df_refined.empty: df_master = pd.concat([df_master, df_refined])

            painel_features = self._construir_features_preditivas(df_master)
            malha_futura = self._treinar_modelo(painel_features)
            malha_futura.to_parquet("datalake/refined/malha_analitica_routing.parquet", index=False)
            
            if self.banco_nuvem: self._sincronizacao_firestore(malha_futura)
            
            self._gerar_documentacao_runbook()
            self._notificar_sucesso()

        except Exception as e:
            self._notificar_erro(str(e))
            raise

if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline_completo()
