import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json, unicodedata
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, SUBTIPOS_LOCAL_PERMITIDOS, LIMITES_SP, ESQUEMA_TRUSTED, COLUNAS_REFINED, PALAVRAS_CHAVE_PERFIL

# Bibliotecas de Inteligência Artificial
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

class MotorSafeDriver:
    
    def __init__(self):
        self.ano_vigente = datetime.now().year
        self.periodo_historico = range(2022, self.ano_vigente + 1)
        self.banco_nuvem = self._estabelecer_conexao_nuvem()
        self.sessao_web = self._criar_sessao_resiliente()
        
        self.auditoria = {
            "volume_raw": 0, "volume_trusted": 0, "volume_refined": 0,
            "falhas_integridade": 0, "malha_motorista": 0, "malha_motociclista": 0,
            "malha_pedestre": 0, "malha_ciclista": 0, "documentos_sincronizados": 0,
            "documentos_atualizados": 0, "novos_dados": False
        }
        
        for pasta in ['raw', 'trusted', 'refined', 'metadata']:
            os.makedirs(f'datalake/{pasta}', exist_ok=True)

    def _estabelecer_conexao_nuvem(self):
        chave_secreta = os.environ.get('FIREBASE_JSON')
        if not chave_secreta or not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave_secreta))
            firebase_admin.initialize_app(credenciais)
        return firestore.client()

    def _criar_sessao_resiliente(self):
        sessao = requests.Session()
        retentativas = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])
        adaptador = HTTPAdapter(max_retries=retentativas)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        return sessao

    def _notificar_sucesso(self):
        endereco_webhook = os.environ.get('DISCORD_SUCESSO')
        if not endereco_webhook: return
        
        pacote_dados = {
            "embeds": [{
                "title": "Relatório Semanal Autobot SafeDriver",
                "color": 3066993,
                "fields": [
                    {"name": "🌊 Data Lake", "value": f"**RAW:** {self.auditoria['volume_raw']:,}\n**TRUSTED:** {self.auditoria['volume_trusted']:,}\n**REFINED:** {self.auditoria['volume_refined']:,}", "inline": False},
                    {"name": "🎯 Qualificação Espacial", "value": f"Motoristas: {self.auditoria['malha_motorista']:,}\nMotos: {self.auditoria['malha_motociclista']:,}\nPedestres: {self.auditoria['malha_pedestre']:,}\nCiclistas: {self.auditoria['malha_ciclista']:,}", "inline": False},
                    {"name": "☁️ Sincronização com Firestore", "value": f"Lotes avaliados: {self.auditoria['documentos_sincronizados']:,}\n**Novos/Alterados (Escritos): {self.auditoria['documentos_atualizados']:,}**", "inline": False}
                ]
            }]
        }
        self.sessao_web.post(endereco_webhook, json=pacote_dados)

    def _notificar_erro(self, diagnostico_falha):
        endereco_webhook = os.environ.get('DISCORD_ERRO')
        if not endereco_webhook: return
        pacote_dados = {"embeds": [{"title": "⚠️ Interrupção Operacional", "color": 15158332, "fields": [{"name": "🛑 Diagnóstico", "value": diagnostico_falha, "inline": False}]}]}
        requests.post(endereco_webhook, json=pacote_dados)

    def _higienizar_texto(self, texto_bruto):
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _verificar_necessidade_download(self, endereco_arquivo, ano_referencia):
        caminho_metadados = f"datalake/metadata/tamanho_{ano_referencia}.json"
        try:
            cabecalho_resposta = self.sessao_web.head(endereco_arquivo, timeout=30, allow_redirects=True)
            tamanho_nuvem = int(cabecalho_resposta.headers.get('Content-Length', 0))
            if os.path.exists(caminho_metadados):
                with open(caminho_metadados, 'r') as arquivo_leitura:
                    if json.load(arquivo_leitura).get('tamanho') == tamanho_nuvem: return False, tamanho_nuvem
            return True, tamanho_nuvem
        except: return True, 0

    def _inferir_perfil_contextual(self, linha):
        # NLP Básico: Extrai perfis cruzando o contexto textual rico da camada Trusted
        perfis_identificados = set()
        contexto_textual = f"{linha.get('NATUREZA_APURADA','')} {linha.get('DESCR_CONDUTA','')} {linha.get('DESCR_SUBTIPOLOCAL','')} {linha.get('RUBRICA','')}".upper()

        for perfil, palavras in PALAVRAS_CHAVE_PERFIL.items():
            if any(palavra in contexto_textual for palavra in palavras):
                perfis_identificados.add(perfil)

        # Fallback para a heurística caso o texto seja pobre
        if not perfis_identificados:
            perfis_base = CATALOGO_CRIMES.get(linha.get('NATUREZA_APURADA'), {}).get('perfis', [])
            perfis_identificados.update(perfis_base)

        return list(perfis_identificados) if perfis_identificados else ['Indefinido']

    def _processar_camadas_dados(self, dataframe_bruto, ano_referencia):
        for coluna in ESQUEMA_TRUSTED.keys():
            if coluna not in dataframe_bruto.columns and coluna != 'ANO_BASE': dataframe_bruto[coluna] = np.nan

        volume_inicial = len(dataframe_bruto)
        dataframe_bruto['ANO_BASE'] = str(ano_referencia)
        
        for coluna, tipo_dado in ESQUEMA_TRUSTED.items():
            if tipo_dado == 'string': dataframe_bruto[coluna] = dataframe_bruto[coluna].apply(self._higienizar_texto)
            elif tipo_dado == 'float': dataframe_bruto[coluna] = pd.to_numeric(dataframe_bruto[coluna].astype(str).str.replace(',', '.'), errors="coerce")
            elif tipo_dado == 'datetime': dataframe_bruto[coluna] = pd.to_datetime(dataframe_bruto[coluna], errors='coerce')
            elif tipo_dado == 'int': dataframe_bruto[coluna] = pd.to_numeric(dataframe_bruto[coluna], errors='coerce').fillna(0).astype(int)

        mascara_geografica = (
            dataframe_bruto['LATITUDE'].notna() & dataframe_bruto['LONGITUDE'].notna() &
            (dataframe_bruto['LATITUDE'] != 0) & (dataframe_bruto['LONGITUDE'] != 0) &
            dataframe_bruto['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            dataframe_bruto['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        
        dataframe_trusted = dataframe_bruto[mascara_geografica].copy()
        dataframe_trusted = dataframe_trusted[list(ESQUEMA_TRUSTED.keys())]
        
        self.auditoria['falhas_integridade'] += (volume_inicial - len(dataframe_trusted))
        self.auditoria['volume_trusted'] += len(dataframe_trusted)

        if 'DESCR_TIPOLOCAL' not in dataframe_trusted.columns: dataframe_trusted['DESCR_TIPOLOCAL'] = 'VIA PUBLICA'

        mascara_negocio = (
            dataframe_trusted['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys()) &
            dataframe_trusted['DESCR_TIPOLOCAL'].isin(TIPOS_LOCAL_PERMITIDOS) &
            dataframe_trusted['DESCR_SUBTIPOLOCAL'].isin(SUBTIPOS_LOCAL_PERMITIDOS)
        )
        
        # O Refined agora recebe os campos extras (Conduta, Rubrica) para a inferência de IA
        dataframe_refinado = dataframe_trusted[mascara_negocio][COLUNAS_REFINED].copy()
        self.auditoria['volume_refined'] += len(dataframe_refinado)
        
        return dataframe_trusted, dataframe_refinado

    def _treinar_ensemble_ia(self, dataframe_consolidado):
        # 1. Aplica o NLP na camada Trusted para criar os vetores de perfis
        dataframe_consolidado['perfis_afetados'] = dataframe_consolidado.apply(self._inferir_perfil_contextual, axis=1)
        df_expandido = dataframe_consolidado.explode('perfis_afetados').dropna(subset=['perfis_afetados'])
        
        df_expandido['codigo_geohash'] = [gh.encode(lat, lon, precision=7) for lat, lon in zip(df_expandido['LATITUDE'], df_expandido['LONGITUDE'])]
        
        def classificar_turno(hora_texto):
            try:
                hora_inteira = int(str(hora_texto).split(':')[0])
                return 'Madrugada' if 0<=hora_inteira<6 else 'Manhã' if 6<=hora_inteira<12 else 'Tarde' if 12<=hora_inteira<18 else 'Noite'
            except: return 'Indefinido'
        df_expandido['turno_operacional'] = df_expandido['HORA_OCORRENCIA_BO'].apply(classificar_turno)
        
        # Label Encoding para alimentar o XGBoost
        encoder_turno = LabelEncoder()
        encoder_perfil = LabelEncoder()
        df_expandido['turno_enc'] = encoder_turno.fit_transform(df_expandido['turno_operacional'])
        df_expandido['perfil_enc'] = encoder_perfil.fit_transform(df_expandido['perfis_afetados'])
        df_expandido['peso_estatistico'] = df_expandido['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))

        # 2. Oráculo Prophet (Prevê a tensão macro-criminal)
        serie_temporal = df_expandido.groupby('DATA_OCORRENCIA_BO').size().reset_index()
        serie_temporal.columns = ['ds', 'y']
        
        modelo_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        modelo_prophet.fit(serie_temporal)
        
        tendencia_prophet = modelo_prophet.predict(modelo_prophet.make_future_dataframe(periods=7))[['ds', 'yhat']]
        df_expandido = df_expandido.merge(tendencia_prophet, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
        df_expandido['fator_prophet'] = df_expandido['yhat'].fillna(df_expandido['yhat'].mean())

        # 3. XGBoost (Combina Espaço, Perfil e a Tendência do Prophet)
        df_treino = df_expandido.groupby(['codigo_geohash', 'LATITUDE', 'LONGITUDE', 'perfil_enc', 'turno_enc', 'fator_prophet']).agg({'peso_estatistico': 'sum'}).reset_index()
        
        X = df_treino[['LATITUDE', 'LONGITUDE', 'perfil_enc', 'turno_enc', 'fator_prophet']]
        y = df_treino['peso_estatistico']
        
        modelo_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        modelo_xgb.fit(X, y)
        
        # Inferencia e Clipping (Mapeamento preditivo final)
        df_treino['score_preditivo'] = modelo_xgb.predict(X)
        df_treino['score_preditivo'] = (df_treino['score_preditivo'] * 2.3).clip(0.5, 10.0).round(2)
        
        # Restaura os textos legíveis
        df_treino['perfis_afetados'] = encoder_perfil.inverse_transform(df_treino['perfil_enc'])
        df_treino['turno_operacional'] = encoder_turno.inverse_transform(df_treino['turno_enc'])
        
        self.auditoria['malha_motorista'] = len(df_treino[df_treino['perfis_afetados'] == 'Motorista'])
        self.auditoria['malha_motociclista'] = len(df_treino[df_treino['perfis_afetados'] == 'Motociclista'])
        self.auditoria['malha_pedestre'] = len(df_treino[df_treino['perfis_afetados'] == 'Pedestre'])
        self.auditoria['malha_ciclista'] = len(df_treino[df_treino['perfis_afetados'] == 'Ciclista'])
        
        return df_treino[['codigo_geohash', 'perfis_afetados', 'turno_operacional', 'score_preditivo']], df_expandido

    def _sincronizacao_delta_firestore(self, malha_final):
        # Recupera os dados atuais para evitar escritas caras e redundantes (Smart Sync)
        colecao_risco = self.banco_nuvem.collection('niveis_risco')
        documentos_atuais = {doc.id: doc.to_dict().get('score') for doc in colecao_risco.stream()}
        
        lote_nuvem = self.banco_nuvem.batch()
        operacoes_pendentes = 0

        self.auditoria['documentos_sincronizados'] = len(malha_final)

        for indice, linha in malha_final.iterrows():
            doc_id = f"{linha['codigo_geohash']}_{linha['perfis_afetados']}_{linha['turno_operacional']}"
            novo_score = float(linha['score_preditivo'])
            
            # Escreve apenas se o documento não existir ou se a IA recalcular um score diferente
            if doc_id not in documentos_atuais or documentos_atuais[doc_id] != novo_score:
                referencia_doc = colecao_risco.document(doc_id)
                lote_nuvem.set(referencia_doc, {
                    'score': novo_score, 'geohash': linha['codigo_geohash'], 
                    'perfil': linha['perfis_afetados'], 'periodo': linha['turno_operacional'],
                    'ultima_atualizacao': firestore.SERVER_TIMESTAMP
                }, merge=True)
                
                operacoes_pendentes += 1
                self.auditoria['documentos_atualizados'] += 1
                
                if operacoes_pendentes % 450 == 0: 
                    lote_nuvem.commit()
                    lote_nuvem = self.banco_nuvem.batch()
                    operacoes_pendentes = 0
                    
        if operacoes_pendentes > 0:
            lote_nuvem.commit()

    def executar_pipeline_completo(self):
        dataframe_mestre_refinado = pd.DataFrame()
        
        try:
            for ano_alvo in self.periodo_historico:
                endereco_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano_alvo}.xlsx"
                caminho_raw = f'datalake/raw/ssp_{ano_alvo}.parquet'
                caminho_trusted = f'datalake/trusted/ssp_trusted_{ano_alvo}.parquet'
                
                realizar_download, tamanho_arquivo = self._verificar_necessidade_download(endereco_ssp, ano_alvo)
                
                if realizar_download or not os.path.exists(caminho_raw):
                    requisicao_dados = self.sessao_web.get(endereco_ssp, timeout=120)
                    if requisicao_dados.status_code != 200: raise ConnectionError(f"Protocolo web recusado. Status: {requisicao_dados.status_code}.")

                    leitura_previa = pd.read_excel(io.BytesIO(requisicao_dados.content), nrows=50, header=None)
                    linha_cabecalho = next((indice for indice, linha in leitura_previa.iterrows() if any(termo in [self._higienizar_texto(str(c)) for c in linha.values] for termo in ['NUM_BO', 'LATITUDE', 'NATUREZA_APURADA'])), None)
                            
                    if linha_cabecalho is None: raise ValueError("Distúrbio tabular: Cabeçalho não encontrado.")

                    tabela_temporaria = pd.read_excel(io.BytesIO(requisicao_dados.content), skiprows=linha_cabecalho, dtype=str)
                    tabela_temporaria.columns = [self._higienizar_texto(c) for c in tabela_temporaria.columns]
                    
                    mapeamento_correcao = {'NUMERO_BO': 'NUM_BO', 'N_BO': 'NUM_BO', 'LAT': 'LATITUDE', 'LON': 'LONGITUDE', 'DATA_FATO': 'DATA_OCORRENCIA_BO', 'HORA_FATO': 'HORA_OCORRENCIA_BO'}
                    tabela_temporaria.rename(columns=mapeamento_correcao, inplace=True)
                    
                    tabela_temporaria.to_parquet(caminho_raw, index=False)
                    with open(f"datalake/metadata/tamanho_{ano_alvo}.json", 'w') as arquivo_escrita: json.dump({'tamanho': tamanho_arquivo}, arquivo_escrita)
                    
                    self.auditoria['novos_dados'] = True
                else:
                    tabela_temporaria = pd.read_parquet(caminho_raw)

                self.auditoria['volume_raw'] += len(tabela_temporaria)
                
                tabela_trusted, tabela_refinada = self._processar_camadas_dados(tabela_temporaria, ano_alvo)
                tabela_trusted.to_parquet(caminho_trusted, index=False)
                dataframe_mestre_refinado = pd.concat([dataframe_mestre_refinado, tabela_refinada])

            if not self.auditoria['novos_dados'] and os.path.exists("datalake/refined/malha_analitica.parquet"): return

            # Treina o Ensemble de IA (Prophet + XGBoost) e recupera a malha preditiva
            malha_final, base_inteligencia = self._treinar_ensemble_ia(dataframe_mestre_refinado)
            
            # Executa a Sincronização Delta (Econômica e Estratégica)
            if self.banco_nuvem:
                self._sincronizacao_delta_firestore(malha_final)

            base_inteligencia.to_parquet("datalake/refined/malha_analitica.parquet", index=False)
            self._notificar_sucesso()

        except Exception as erro_critico:
            self._notificar_erro(diagnostico_falha=str(erro_critico))

if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline_completo()
