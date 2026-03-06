import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json, unicodedata
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from config import CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, SUBTIPOS_LOCAL_PERMITIDOS, LIMITES_SP, COLUNAS_TEXTO, COLUNAS_REFINED

class MotorSafeDriver:
    """ Arquitetura principal para extração, saneamento e predição de risco viário. """
    
    def __init__(self):
        # Definição do horizonte temporal e construção do Data Lake
        self.ano_vigente = datetime.now().year
        self.periodo_historico = range(2022, self.ano_vigente + 1)
        self.banco_nuvem = self._estabelecer_conexao_nuvem()
        self.metricas_auditoria = {"extraidos": 0, "qualificados": 0, "sincronizados": 0, "novos_dados": False}
        
        for pasta in ['raw', 'trusted', 'refined', 'metadata']:
            os.makedirs(f'datalake/{pasta}', exist_ok=True)

    def _estabelecer_conexao_nuvem(self):
        # Autenticação segura com a infraestrutura NoSQL
        chave_secreta = os.environ.get('FIREBASE_JSON')
        if not chave_secreta or not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave_secreta))
            firebase_admin.initialize_app(credenciais)
        return firestore.client()

    def _notificar_monitoramento(self, operacao_bem_sucedida=True, mensagem_erro=""):
        # Transmissão de status executivo para os canais de telemetria
        endereco_webhook = os.environ.get('DISCORD_SUCESSO' if operacao_bem_sucedida else 'DISCORD_ERRO')
        if not endereco_webhook: return
        
        pacote_dados = {
            "embeds": [{
                "title": f"🛡️ Motor SafeDriver: {'Sucesso Operacional' if operacao_bem_sucedida else 'Falha na Execução'}",
                "color": 3066993 if operacao_bem_sucedida else 15158332,
                "fields": [
                    {"name": "📥 Extraídos", "value": f"{self.metricas_auditoria['extraidos']:,}", "inline": True},
                    {"name": "💎 Qualificados", "value": f"{self.metricas_auditoria['qualificados']:,}", "inline": True},
                    {"name": "🚀 Sincronizados", "value": f"{self.metricas_auditoria['sincronizados']:,}", "inline": True}
                ],
                "description": f"**Status:** {mensagem_erro or 'Conformidade total do pipeline.'}"
            }]
        }
        requests.post(endereco_webhook, json=pacote_dados)

    def _higienizar_texto(self, texto_bruto):
        # Normalização de caracteres para garantir integridade estrutural
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): 
            return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _verificar_necessidade_download(self, endereco_arquivo, ano_referencia):
        # Validação de metadados HTTP para prevenção de downloads redundantes
        caminho_metadados = f"datalake/metadata/tamanho_{ano_referencia}.json"
        try:
            cabecalho = requests.head(endereco_arquivo, timeout=30)
            tamanho_nuvem = int(cabecalho.headers.get('Content-Length', 0))
            if os.path.exists(caminho_metadados):
                with open(caminho_metadados, 'r') as arquivo_leitura:
                    if json.load(arquivo_leitura).get('tamanho') == tamanho_nuvem: return False, tamanho_nuvem
            return True, tamanho_nuvem
        except: return True, 0

    def _processar_camadas_dados(self, dataframe_bruto, ano_referencia):
        # Garantia de contrato de colunas para evitar anomalias de schema
        for coluna in COLUNAS_REFINED:
            if coluna not in dataframe_bruto.columns and coluna != 'ANO_BASE':
                dataframe_bruto[coluna] = np.nan

        # Tipagem estrutural para armazenamento colunar estável
        dataframe_bruto['NUM_BO'] = dataframe_bruto['NUM_BO'].astype(str)
        dataframe_bruto['DATA_OCORRENCIA_BO'] = dataframe_bruto['DATA_OCORRENCIA_BO'].astype(str)
        dataframe_bruto['HORA_OCORRENCIA_BO'] = dataframe_bruto['HORA_OCORRENCIA_BO'].astype(str)
        dataframe_bruto['ANO_BASE'] = int(ano_referencia)

        # Conversão numérica e tratamento rigoroso de coordenadas
        dataframe_bruto['LATITUDE'] = pd.to_numeric(dataframe_bruto['LATITUDE'], errors="coerce")
        dataframe_bruto['LONGITUDE'] = pd.to_numeric(dataframe_bruto['LONGITUDE'], errors="coerce")

        # Normalização textual restrita aos atributos categóricos
        for coluna in COLUNAS_TEXTO:
            dataframe_bruto[coluna] = dataframe_bruto[coluna].apply(self._higienizar_texto)

        # Saneamento geográfico e eliminação de coordenadas nulas ou zeradas
        mascara_geografica = (
            dataframe_bruto['LATITUDE'].notna() & dataframe_bruto['LONGITUDE'].notna() &
            (dataframe_bruto['LATITUDE'] != 0) & (dataframe_bruto['LONGITUDE'] != 0) &
            dataframe_bruto['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            dataframe_bruto['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        dataframe_confiavel = dataframe_bruto[mascara_geografica].copy()

        # Filtragem direcional baseada nos critérios de negócio
        mascara_negocio = (
            dataframe_confiavel['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys()) &
            dataframe_confiavel['DESCR_TIPOLOCAL'].isin(TIPOS_LOCAL_PERMITIDOS) &
            dataframe_confiavel['DESCR_SUBTIPOLOCAL'].isin(SUBTIPOS_LOCAL_PERMITIDOS)
        )
        dataframe_refinado = dataframe_confiavel[mascara_negocio].copy()

        # Tipagem temporal definitiva para integração analítica
        dataframe_refinado['DATA_OCORRENCIA_BO'] = pd.to_datetime(dataframe_refinado['DATA_OCORRENCIA_BO'], errors='coerce')
        
        return dataframe_refinado[COLUNAS_REFINED]

    def _calcular_predicao_risco(self, dataframe_consolidado):
        # Modelagem preditiva e distribuição multimodal de ameaças
        def distribuir_perfis(natureza_crime):
            perfil_alvo = CATALOGO_CRIMES.get(natureza_crime, {}).get('perfil', 'OUTROS')
            return ['Motorista', 'Pedestre'] if perfil_alvo == 'TODOS' else [perfil_alvo.capitalize()]

        dataframe_consolidado['perfis_afetados'] = dataframe_consolidado['NATUREZA_APURADA'].apply(distribuir_perfis)
        dataframe_expandido = dataframe_consolidado.explode('perfis_afetados').dropna(subset=['perfis_afetados'])
        
        # Geocodificação de precisão urbana
        dataframe_expandido['codigo_geohash'] = [gh.encode(lat, lon, precision=7) for lat, lon in zip(dataframe_expandido['LATITUDE'], dataframe_expandido['LONGITUDE'])]
        
        # Classificação cronológica para análise de sazonalidade
        def classificar_turno(hora_texto):
            try:
                hora_inteira = int(str(hora_texto).split(':')[0])
                return 'Madrugada' if 0<=hora_inteira<6 else 'Manhã' if 6<=hora_inteira<12 else 'Tarde' if 12<=hora_inteira<18 else 'Noite'
            except: return 'Indefinido'
        
        dataframe_expandido['turno_operacional'] = dataframe_expandido['HORA_OCORRENCIA_BO'].apply(classificar_turno)
        dataframe_expandido['peso_estatistico'] = dataframe_expandido['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        
        # Agrupamento e processamento de densidade de risco
        malha_risco = dataframe_expandido.groupby(['codigo_geohash', 'perfis_afetados', 'turno_operacional']).agg({'peso_estatistico': 'sum'}).reset_index()
        malha_risco['score_preditivo'] = (malha_risco['peso_estatistico'] * 2.3).clip(0.5, 10.0).round(2)
        return malha_risco, dataframe_expandido

    def executar_pipeline_completo(self):
        # Orquestração integral do ecossistema de dados
        dataframe_mestre = pd.DataFrame()
        try:
            for ano_alvo in self.periodo_historico:
                endereco_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano_alvo}.xlsx"
                caminho_raw = f'datalake/raw/ssp_{ano_alvo}.parquet'
                
                realizar_download, tamanho_arquivo = self._verificar_necessidade_download(endereco_ssp, ano_alvo)
                if realizar_download or not os.path.exists(caminho_raw):
                    requisicao_dados = requests.get(endereco_ssp, timeout=120)
                    tabela_temporaria = pd.read_excel(io.BytesIO(requisicao_dados.content), skiprows=1)
                    tabela_temporaria.columns = [self._higienizar_texto(c) for c in tabela_temporaria.columns]
                    
                    # Forçamento rigoroso de string para o Boletim de Ocorrência na extração imediata
                    tabela_temporaria['NUM_BO'] = tabela_temporaria['NUM_BO'].astype(str)
                    tabela_temporaria.to_parquet(caminho_raw, index=False)
                    
                    with open(f"datalake/metadata/tamanho_{ano_alvo}.json", 'w') as arquivo_escrita: 
                        json.dump({'tamanho': tamanho_arquivo}, arquivo_escrita)
                    self.metricas_auditoria['novos_dados'] = True
                else:
                    tabela_temporaria = pd.read_parquet(caminho_raw)

                self.metricas_auditoria['extraidos'] += len(tabela_temporaria)
                
                # Execução do funil de saneamento
                tabela_refinada = self._processar_camadas_dados(tabela_temporaria, ano_alvo)
                dataframe_mestre = pd.concat([dataframe_mestre, tabela_refinada])

            if not self.metricas_auditoria['novos_dados'] and os.path.exists("datalake/refined/malha_analitica.parquet"):
                return

            self.metricas_auditoria['qualificados'] = len(dataframe_mestre)
            malha_final, base_inteligencia = self._calcular_predicao_risco(dataframe_mestre)
            self.metricas_auditoria['sincronizados'] = len(malha_final)

            # Persistência atômica no armazenamento em nuvem
            if self.banco_nuvem:
                lote_nuvem = self.banco_nuvem.batch()
                for indice, linha in malha_final.iterrows():
                    identificador_documento = f"{linha['codigo_geohash']}_{linha['perfis_afetados']}_{linha['turno_operacional']}"
                    lote_nuvem.set(self.banco_nuvem.collection('niveis_risco').document(identificador_documento), {
                        'score': float(linha['score_preditivo']), 'geohash': linha['codigo_geohash'], 
                        'perfil': linha['perfis_afetados'], 'periodo': linha['turno_operacional'],
                        'sincronizacao': firestore.SERVER_TIMESTAMP
                    }, merge=True)
                    if (indice + 1) % 450 == 0: 
                        lote_nuvem.commit()
                        lote_nuvem = self.banco_nuvem.batch()
                lote_nuvem.commit()

            base_inteligencia.to_parquet("datalake/refined/malha_analitica.parquet", index=False)
            self._notificar_monitoramento(operacao_bem_sucedida=True)

        except Exception as erro_critico:
            self._notificar_monitoramento(operacao_bem_sucedida=False, mensagem_erro=str(erro_critico))

if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline_completo()
