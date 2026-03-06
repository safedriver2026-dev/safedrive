import pandas as pd
import numpy as np
import pygeohash as gh
import os, io, requests, json, unicodedata
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from config import CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, SUBTIPOS_LOCAL_PERMITIDOS, LIMITES_SP, ESQUEMA_TRUSTED, COLUNAS_REFINED

class MotorSafeDriver:
    """ Infraestrutura de extração, paridade de schema e modelagem de risco. """
    
    def __init__(self):
        # Parametrização temporal e alocação de armazenamento em múltiplas camadas
        self.ano_vigente = datetime.now().year
        self.periodo_historico = range(2022, self.ano_vigente + 1)
        self.banco_nuvem = self._estabelecer_conexao_nuvem()
        
        # Dicionário de auditoria expandido para refletir o Data Lake (Medallion)
        self.auditoria = {
            "volume_raw": 0,
            "volume_trusted": 0,
            "volume_refined": 0,
            "falhas_integridade": 0,
            "malha_motorista": 0,
            "malha_motociclista": 0,
            "malha_pedestre": 0,
            "malha_ciclista": 0,
            "documentos_sincronizados": 0,
            "novos_dados": False
        }
        
        for pasta in ['raw', 'trusted', 'refined', 'metadata']:
            os.makedirs(f'datalake/{pasta}', exist_ok=True)

    def _estabelecer_conexao_nuvem(self):
        # Autenticação segura com infraestrutura NoSQL
        chave_secreta = os.environ.get('FIREBASE_JSON')
        if not chave_secreta or not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave_secreta))
            firebase_admin.initialize_app(credenciais)
        return firestore.client()

    def _notificar_sucesso(self):
        # Emissão de relatório analítico detalhando o funil do Data Lake e matriz de modais
        endereco_webhook = os.environ.get('DISCORD_SUCESSO')
        if not endereco_webhook: return
        
        pacote_dados = {
            "embeds": [{
                "title": "📊 Relatório Executivo: Data Lake, Qualidade e Sincronização",
                "color": 3066993,
                "fields": [
                    {"name": "🌊 Saúde do Data Lake (Funil)", "value": f"**RAW (Bruto):** {self.auditoria['volume_raw']:,} registos\n**TRUSTED (Limpo):** {self.auditoria['volume_trusted']:,} registos\n**REFINED (Negócio):** {self.auditoria['volume_refined']:,} registos", "inline": False},
                    {"name": "⚙️ Qualidade e Expurgo", "value": f"Anomalias e Coordenadas Inválidas Expurgadas: {self.auditoria['falhas_integridade']:,}", "inline": False},
                    {"name": "🚗 Risco Veicular", "value": f"Motoristas: {self.auditoria['malha_motorista']:,}\nMotociclistas: {self.auditoria['malha_motociclista']:,}", "inline": True},
                    {"name": "🚶 Risco Vulneráveis", "value": f"Pedestres: {self.auditoria['malha_pedestre']:,}\nCiclistas: {self.auditoria['malha_ciclista']:,}", "inline": True},
                    {"name": "☁️ Sincronização Firestore", "value": f"{self.auditoria['documentos_sincronizados']:,} coleções espaciais sincronizadas com sucesso.", "inline": False}
                ],
                "footer": {"text": "Monitorização concluída. Data Lake e Nuvem operacionais."}
            }]
        }
        requests.post(endereco_webhook, json=pacote_dados)

    def _notificar_erro(self, diagnostico_falha):
        # Bloqueio de operação por corrupção de malha ou erro de requisição
        endereco_webhook = os.environ.get('DISCORD_ERRO')
        if not endereco_webhook: return
        
        pacote_dados = {
            "embeds": [{
                "title": "⚠️ Ruptura de Qualidade Operacional",
                "color": 15158332,
                "fields": [
                    {"name": "🛑 Causa Primária", "value": diagnostico_falha, "inline": False},
                    {"name": "☁️ Intervenção Automática", "value": "Sincronização com o Firestore paralisada para proteger o schema e a integridade da base de dados.", "inline": False}
                ]
            }]
        }
        requests.post(endereco_webhook, json=pacote_dados)

    def _higienizar_texto(self, texto_bruto):
        # Expulsão de caracteres especiais e acentuação da matriz tabular
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): 
            return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _verificar_necessidade_download(self, endereco_arquivo, ano_referencia):
        # Leitura assíncrona de metadados para redução de consumo de banda
        cabecalho_navegador = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        caminho_metadados = f"datalake/metadata/tamanho_{ano_referencia}.json"
        
        try:
            cabecalho_resposta = requests.head(endereco_arquivo, headers=cabecalho_navegador, timeout=30, allow_redirects=True)
            tamanho_nuvem = int(cabecalho_resposta.headers.get('Content-Length', 0))
            if os.path.exists(caminho_metadados):
                with open(caminho_metadados, 'r') as arquivo_leitura:
                    if json.load(arquivo_leitura).get('tamanho') == tamanho_nuvem: 
                        return False, tamanho_nuvem
            return True, tamanho_nuvem
        except: 
            return True, 0

    def _processar_camadas_dados(self, dataframe_bruto, ano_referencia):
        # Injeção de paridade para colunas flutuantes da fonte governamental
        for coluna in ESQUEMA_TRUSTED.keys():
            if coluna not in dataframe_bruto.columns and coluna != 'ANO_BASE':
                dataframe_bruto[coluna] = np.nan

        volume_inicial = len(dataframe_bruto)
        dataframe_bruto['ANO_BASE'] = str(ano_referencia)
        
        # Tipagem dinâmica forçada e higienização em massa
        for coluna, tipo_dado in ESQUEMA_TRUSTED.items():
            if tipo_dado == 'string':
                dataframe_bruto[coluna] = dataframe_bruto[coluna].apply(self._higienizar_texto)
            elif tipo_dado == 'float':
                dataframe_bruto[coluna] = pd.to_numeric(dataframe_bruto[coluna].astype(str).str.replace(',', '.'), errors="coerce")
            elif tipo_dado == 'datetime':
                dataframe_bruto[coluna] = pd.to_datetime(dataframe_bruto[coluna], errors='coerce')
            elif tipo_dado == 'int':
                dataframe_bruto[coluna] = pd.to_numeric(dataframe_bruto[coluna], errors='coerce').fillna(0).astype(int)

        # Restrição de perímetro espacial da camada limpa
        mascara_geografica = (
            dataframe_bruto['LATITUDE'].notna() & dataframe_bruto['LONGITUDE'].notna() &
            (dataframe_bruto['LATITUDE'] != 0) & (dataframe_bruto['LONGITUDE'] != 0) &
            dataframe_bruto['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            dataframe_bruto['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        
        # Camada TRUSTED: Todo o contexto de negócio padronizado
        dataframe_trusted = dataframe_bruto[mascara_geografica].copy()
        
        # Remoção de colunas nativas da SSP que não pertencem ao contrato Trusted
        dataframe_trusted = dataframe_trusted[list(ESQUEMA_TRUSTED.keys())]
        
        # Auditoria do funil (O que foi expurgado entre RAW e TRUSTED)
        self.auditoria['falhas_integridade'] += (volume_inicial - len(dataframe_trusted))
        self.auditoria['volume_trusted'] += len(dataframe_trusted)

        # Camada REFINED: Apenas os vetores estritos necessários para a predição viária
        if 'DESCR_TIPOLOCAL' not in dataframe_trusted.columns:
            dataframe_trusted['DESCR_TIPOLOCAL'] = 'VIA PUBLICA'

        mascara_negocio = (
            dataframe_trusted['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys()) &
            dataframe_trusted['DESCR_TIPOLOCAL'].isin(TIPOS_LOCAL_PERMITIDOS) &
            dataframe_trusted['DESCR_SUBTIPOLOCAL'].isin(SUBTIPOS_LOCAL_PERMITIDOS)
        )
        dataframe_refinado = dataframe_trusted[mascara_negocio][COLUNAS_REFINED].copy()
        
        self.auditoria['volume_refined'] += len(dataframe_refinado)
        
        return dataframe_trusted, dataframe_refinado

    def _calcular_predicao_risco(self, dataframe_consolidado):
        # Explosão matricial para cobertura de múltiplos perfis por boletim
        dataframe_consolidado['perfis_afetados'] = dataframe_consolidado['NATUREZA_APURADA'].apply(
            lambda crime: CATALOGO_CRIMES.get(crime, {}).get('perfis', ['Indefinido'])
        )
        dataframe_expandido = dataframe_consolidado.explode('perfis_afetados').dropna(subset=['perfis_afetados'])
        
        # Codificação de coordenadas flutuantes para quarteirões geohash
        dataframe_expandido['codigo_geohash'] = [gh.encode(lat, lon, precision=7) for lat, lon in zip(dataframe_expandido['LATITUDE'], dataframe_expandido['LONGITUDE'])]
        
        # Identificação paramétrica de ciclos solares
        def classificar_turno(hora_texto):
            try:
                hora_inteira = int(str(hora_texto).split(':')[0])
                return 'Madrugada' if 0<=hora_inteira<6 else 'Manhã' if 6<=hora_inteira<12 else 'Tarde' if 12<=hora_inteira<18 else 'Noite'
            except: return 'Indefinido'
        
        dataframe_expandido['turno_operacional'] = dataframe_expandido['HORA_OCORRENCIA_BO'].apply(classificar_turno)
        dataframe_expandido['peso_estatistico'] = dataframe_expandido['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        
        # Cálculo colunar ponderado limitando o teto matemático a 10
        malha_risco = dataframe_expandido.groupby(['codigo_geohash', 'perfis_afetados', 'turno_operacional']).agg({'peso_estatistico': 'sum'}).reset_index()
        malha_risco['score_preditivo'] = (malha_risco['peso_estatistico'] * 2.3).clip(0.5, 10.0).round(2)
        
        # Contabilização fragmentada de impacto nos modais
        self.auditoria['malha_motorista'] = len(malha_risco[malha_risco['perfis_afetados'] == 'Motorista'])
        self.auditoria['malha_motociclista'] = len(malha_risco[malha_risco['perfis_afetados'] == 'Motociclista'])
        self.auditoria['malha_pedestre'] = len(malha_risco[malha_risco['perfis_afetados'] == 'Pedestre'])
        self.auditoria['malha_ciclista'] = len(malha_risco[malha_risco['perfis_afetados'] == 'Ciclista'])
        
        return malha_risco, dataframe_expandido

    def executar_pipeline_completo(self):
        # Regência do funil integral desde o download até a replicação Firebase
        dataframe_mestre_refinado = pd.DataFrame()
        
        try:
            for ano_alvo in self.periodo_historico:
                endereco_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano_alvo}.xlsx"
                caminho_raw = f'datalake/raw/ssp_{ano_alvo}.parquet'
                caminho_trusted = f'datalake/trusted/ssp_trusted_{ano_alvo}.parquet'
                
                realizar_download, tamanho_arquivo = self._verificar_necessidade_download(endereco_ssp, ano_alvo)
                
                if realizar_download or not os.path.exists(caminho_raw):
                    cabecalho_navegador = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    requisicao_dados = requests.get(endereco_ssp, headers=cabecalho_navegador, timeout=120)
                    
                    if requisicao_dados.status_code != 200:
                        raise ConnectionError(f"Protocolo HTTP bloqueado. Resposta do servidor governamental: {requisicao_dados.status_code}.")

                    # Varredura inteligente do início tabular
                    leitura_previa = pd.read_excel(io.BytesIO(requisicao_dados.content), nrows=50, header=None)
                    linha_cabecalho = 0
                    cabecalho_encontrado = False
                    
                    for indice, linha in leitura_previa.iterrows():
                        linha_texto = [self._higienizar_texto(str(celula)) for celula in linha.values]
                        if any(termo in linha_texto for termo in ['NUM_BO', 'LATITUDE', 'NATUREZA_APURADA', 'NUMERO_BO']):
                            linha_cabecalho = indice
                            cabecalho_encontrado = True
                            break
                            
                    if not cabecalho_encontrado:
                        raise ValueError("Ruído extremo na matriz. Cabeçalho governamental irreconhecível.")

                    # Blindagem da ingestão convertendo todos os bytes em cadeia string primitiva
                    tabela_temporaria = pd.read_excel(io.BytesIO(requisicao_dados.content), skiprows=linha_cabecalho, dtype=str)
                    tabela_temporaria.columns = [self._higienizar_texto(c) for c in tabela_temporaria.columns]
                    
                    mapeamento_correcao = {
                        'NUMERO_BO': 'NUM_BO', 'N_BO': 'NUM_BO', 'BOLETIM': 'NUM_BO',
                        'LAT': 'LATITUDE', 'LON': 'LONGITUDE',
                        'DATA_FATO': 'DATA_OCORRENCIA_BO', 'HORA_FATO': 'HORA_OCORRENCIA_BO'
                    }
                    tabela_temporaria.rename(columns=mapeamento_correcao, inplace=True)
                    
                    if 'NUM_BO' not in tabela_temporaria.columns:
                        raise KeyError(f"Identificador primário ausente no escaneamento tabular do ficheiro do ano {ano_alvo}.")
                    
                    tabela_temporaria.to_parquet(caminho_raw, index=False)
                    with open(f"datalake/metadata/tamanho_{ano_alvo}.json", 'w') as arquivo_escrita: 
                        json.dump({'tamanho': tamanho_arquivo}, arquivo_escrita)
                    
                    self.auditoria['novos_dados'] = True
                else:
                    tabela_temporaria = pd.read_parquet(caminho_raw)

                # Registo da volumetria inicial no Data Lake
                self.auditoria['volume_raw'] += len(tabela_temporaria)
                
                # Execução da tipagem ampla gerando as partições Trusted (histórica) e Refined (preditiva)
                tabela_trusted, tabela_refinada = self._processar_camadas_dados(tabela_temporaria, ano_alvo)
                
                # Materialização autônoma da camada Trusted
                tabela_trusted.to_parquet(caminho_trusted, index=False)
                
                dataframe_mestre_refinado = pd.concat([dataframe_mestre_refinado, tabela_refinada])

            # Aborto condicionado à ausência de vetores de alteração nas partições locais
            if not self.auditoria['novos_dados'] and os.path.exists("datalake/refined/malha_analitica.parquet"):
                return

            malha_final, base_inteligencia = self._calcular_predicao_risco(dataframe_mestre_refinado)
            self.auditoria['documentos_sincronizados'] = len(malha_final)

            # Transação paralela em blocos para a instância Firebase
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
            self._notificar_sucesso()

        except Exception as erro_critico:
            self._notificar_erro(diagnostico_falha=str(erro_critico))

if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline_completo()
