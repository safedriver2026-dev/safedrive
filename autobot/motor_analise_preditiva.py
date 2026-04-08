import sys
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback
import hashlib
import polars as pl
import pandas as pd
import numpy as np
import h3
import gc
import holidays
import warnings
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

print("[INICIALIZACAO] Motor Autônomo SafeDriver ativado. Carregando módulos semânticos e preditivos...", flush=True)
warnings.filterwarnings("ignore")

class MotorSafeDriverCloud:
    def __init__(self):
        self.raiz = Path(".")
        self.bucket_nome = os.environ.get("GCP_BUCKET_NAME")
        
        if not self.bucket_nome:
            raise ValueError("[ERRO_FATAL] A variável de ambiente GCP_BUCKET_NAME não foi injetada pelo GitHub Actions.")
            
        self.pastas = {
            "bruto": self.raiz / "datalake" / "raw",
            "processado": self.raiz / "datalake" / "prata",
            "refinado": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria",
            "relatorios": self.raiz / "datalake" / "relatorios"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        self.hoje = datetime.now()
        
        # Apontamento restaurado para o projeto de faturamento onde o balde reside fisicamente
        self.storage_client = storage.Client(project="sandbox-suprimentos")
        
        self.assinaturas_seguranca = {}
        self.anomalias_detectadas = 0
        self.registros_brutos = 0
        self.registros_validados = 0
        self.registros_descartados = 0

        ano_atual = self.hoje.year
        calendario_feriados = holidays.Brazil(subdiv='SP', years=[ano_atual - 2, ano_atual - 1, ano_atual])
        self.datas_feriados = list(calendario_feriados.keys())

    def gerar_assinatura_criptografica(self, caminho):
        sha256_hash = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco_bytes in iter(lambda: f.read(4096), b""):
                sha256_hash.update(bloco_bytes)
        return sha256_hash.hexdigest()

    def validar_perimetro_geografico(self, df):
        return df.filter(
            (pl.col("LAT").is_between(-25.5, -19.5)) & 
            (pl.col("LON").is_between(-53.5, -44.0))
        )

    def criar_sessao_cacada(self):
        sessao = requests.Session()
        estrategia_retry = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[403, 404, 429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adaptador = HTTPAdapter(max_retries=estrategia_retry)
        sessao.mount("https://", adaptador)
        sessao.mount("http://", adaptador)
        
        sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel, text/html, */*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })
        return sessao

    @staticmethod
    def achatar_texto(texto):
        t_sem_acento = ''.join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^A-Z0-9]', '', t_sem_acento.upper())

    def normalizador_semantico(self, df, ano):
        df = df.rename({c: str(c).upper().strip() for c in df.columns})
        
        dicionario_sinonimos = {
            'NUMEROBOLETIM': 'NUM_BO', 'NUMERODOBOLETIM': 'NUM_BO', 'NUMEROBO': 'NUM_BO', 
            'NUMERODOBO': 'NUM_BO', 'BOLETIM': 'NUM_BO', 'BO': 'NUM_BO', 'NBO': 'NUM_BO', 'NBOLETIM': 'NUM_BO', 'NUMBO': 'NUM_BO',
            'DATAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATADOFATO': 'DATA_OCORRENCIA_BO', 
            'DATADAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATADEOCORRENCIA': 'DATA_OCORRENCIA_BO', 
            'DATAREGCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATA': 'DATA_OCORRENCIA_BO',
            'NATUREZA': 'RUBRICA', 'NATUREZAJURIDICA': 'RUBRICA', 'CRIME': 'RUBRICA', 'DELITO': 'RUBRICA',
            'DESCRICAOLOCAL': 'DESCR_TIPOLOCAL', 'TIPOLOCAL': 'DESCR_TIPOLOCAL', 
            'LOCALFATO': 'DESCR_TIPOLOCAL', 'AMBIENTE': 'DESCR_TIPOLOCAL',
            'LAT': 'LATITUDE', 'LONG': 'LONGITUDE', 'LON': 'LONGITUDE', 
            'LONGITUD': 'LONGITUDE', 'COORDX': 'LONGITUDE', 'COORDY': 'LATITUDE'
        }

        mapa_renomeacao = {}
        colunas_atuais = df.columns
        
        for col_original in colunas_atuais:
            col_achatada = self.achatar_texto(col_original)
            if col_achatada in dicionario_sinonimos:
                alvo = dicionario_sinonimos[col_achatada]
                if alvo not in colunas_atuais and alvo not in mapa_renomeacao.values():
                    mapa_renomeacao[col_original] = alvo

        if mapa_renomeacao:
            df = df.rename(mapa_renomeacao)

        if "NUM_BO" not in df.columns:
            print(f"[AVISO_ESQUEMA] Identificador NUM_BO não localizado de forma colunar no ano {ano}. Gerando chaves primárias sintéticas.", flush=True)
            df = df.with_columns([
                pl.format("VIRTUAL_BO_{}_{}", pl.lit(ano), pl.int_range(0, pl.len())).alias("NUM_BO")
            ])
            
        return df

    def executar_extracao_dados(self):
        print("[COLETOR_DADOS] Iniciando protocolo de Caçada Contínua (Sincronização Resiliente).", flush=True)
        ano_atual = self.hoje.year
        anos_foco = [ano_atual - 2, ano_atual - 1, ano_atual]
        
        sessao = self.criar_sessao_cacada()
        arquivos_baixados_com_sucesso = 0

        for ano in anos_foco:
            arquivo_bruto = self.pastas["bruto"] / f"ssp_{ano}.parquet"
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            
            if arquivo_bruto.exists() and ano < ano_atual:
                self.assinaturas_seguranca[f"ssp_{ano}"] = self.gerar_assinatura_criptografica(arquivo_bruto)
                print(f"[COLETOR_DADOS] Arquivo do ano {ano} validado em cache local.", flush=True)
                arquivos_baixados_com_sucesso += 1
                continue

            try:
                print(f"[COLETOR_DADOS] Forçando extração do servidor SSP para o ano {ano}...", flush=True)
                r = sessao.get(url, stream=True, verify=False, timeout=180) 
                
                if r.status_code == 200:
                    temp_xlsx = self.pastas["bruto"] / "temp.xlsx"
                    with open(temp_xlsx, 'wb') as f:
                        for pedaco in r.iter_content(chunk_size=2*1024*1024): f.write(pedaco)
                    
                    import fastexcel
                    excel = fastexcel.read_excel(str(temp_xlsx))
                    
                    df_abas_validas = []
                    indicadores_matriz = ['NUMBO', 'NUMEROBO', 'BOLETIM', 'LATITUDE', 'DATAOCORRENCIA']
                    
                    for aba in excel.sheet_names:
                        df_temp = pl.read_excel(str(temp_xlsx), sheet_name=aba, engine="calamine")
                        colunas_achatadas = [self.achatar_texto(c) for c in df_temp.columns]
                        
                        if any(ind in colunas_achatadas for ind in indicadores_matriz) and len(df_temp.columns) > 5:
                            df_temp = self.normalizador_semantico(df_temp, ano)
                            df_temp = df_temp.with_columns(pl.all().cast(pl.String))
                            df_abas_validas.append(df_temp)
                            print(f"[COLETOR_DADOS] Matriz validada e extraída da aba: '{aba}'.", flush=True)
                            
                    if df_abas_validas:
                        df_novo = pl.concat(df_abas_validas, how="diagonal")
                        print(f"[COLETOR_DADOS] Fusão de {len(df_abas_validas)} aba(s) concluída para o ano {ano}.", flush=True)
                    else:
                        aba_mais_colunas = max(excel.sheet_names, key=lambda x: len(pl.read_excel(str(temp_xlsx), sheet_name=x, engine="calamine").columns))
                        print(f"[ALERTA_ESQUEMA] Identificadores ausentes. Selecionando por densidade colunar na aba: '{aba_mais_colunas}'.", flush=True)
                        df_novo = pl.read_excel(str(temp_xlsx), sheet_name=aba_mais_colunas, engine="calamine")
                        df_novo = self.normalizador_semantico(df_novo, ano)
                        df_novo = df_novo.with_columns(pl.all().cast(pl.String))
                    
                    if arquivo_bruto.exists():
                        df_final = pl.concat([pl.read_parquet(arquivo_bruto), df_novo], how="diagonal")
                        df_final = df_final.unique(subset=["NUM_BO"], keep="last")
                    else:
                        df_final = df_novo.unique(subset=["NUM_BO"])

                    df_final.write_parquet(arquivo_bruto)
                    self.assinaturas_seguranca[f"ssp_{ano}"] = self.gerar_assinatura_criptografica(arquivo_bruto)
                    os.remove(temp_xlsx)
                    gc.collect()
                    arquivos_baixados_com_sucesso += 1
                    print(f"[COLETOR_DADOS] Ingestão do ano {ano} salva com sucesso.", flush=True)
                else:
                    print(f"[ALERTA_REDE] O servidor SSP recusou todas as tentativas para o ano {ano} (HTTP {r.status_code}).", flush=True)
            except Exception as e:
                print(f"[ALERTA_REDE] Falha catastrófica ao processar o ano {ano}: {str(e)}", flush=True)

        if arquivos_baixados_com_sucesso == 0 and not list(self.pastas["bruto"].glob("*.parquet")):
            raise RuntimeError("[ERRO_FATAL] O Protocolo de Caçada esgotou todas as tentativas e o repositório permanece vazio.")

    def executar_limpeza_dados(self):
        print("[REFINADOR_DADOS] Executando rotinas de higienização, isolamento geoespacial e rastreio volumétrico.", flush=True)
        arquivos = list(self.pastas["bruto"].glob("*.parquet"))
        
        lista_lazyframes = [pl.scan_parquet(str(f)) for f in arquivos]
        lf = pl.concat(lista_lazyframes, how="diagonal")
        
        self.registros_brutos = lf.select(pl.len()).collect().item()
        
        if "DESCR_TIPOLOCAL" not in lf.columns:
            lf = lf.with_columns(pl.lit("NAO_INFORMADO").alias("DESCR_TIPOLOCAL"))

        df_processado = (
            lf.with_columns([
                pl.col("DATA_OCORRENCIA_BO").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_DT"),
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
                pl.col("DESCR_TIPOLOCAL").fill_null("OUTROS")
            ])
            .filter(pl.col("LAT").is_not_null() & pl.col("DATA_DT").is_not_null())
            .unique(subset=["NUM_BO"]) 
            .collect()
        )

        df_processado = self.validar_perimetro_geografico(df_processado)

        self.registros_validados = df_processado.height
        self.registros_descartados = self.registros_brutos - self.registros_validados

        print("[PROTOCOLO_PRIVACIDADE] Anonimizando identificadores públicos. Destruindo NUM_BO e gerando chaves sintéticas.", flush=True)
        df_processado = df_processado.with_columns(
            pl.format("SEC_KEY_{}", pl.col("NUM_BO").fill_null("SINTETICO").hash(42)).alias("ID_ANONIMO")
        ).drop("NUM_BO")

        coords = df_processado.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        
        df_final = df_processado.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")
        df_final.write_parquet(self.pastas["processado"] / "camada_prata.parquet")
        return df_final

    def identificar_ciclo_pagamento(self, df):
        return df.with_columns(
            pl.when(
                (pl.col("DATA_DT").dt.day().is_between(28, 31)) | 
                (pl.col("DATA_DT").dt.day().is_between(1, 7))
            )
            .then(1)
            .otherwise(0)
            .alias("FLG_DIA_PAGAMENTO")
        )

    def aplicar_suavizacao_vizinhanca(self, df_fato):
        mapa_inc = dict(zip(df_fato['H3'], df_fato['INCIDENTES']))
        suavizado = []
        for h, inc in zip(df_fato['H3'], df_fato['INCIDENTES']):
            vizinhos = h3.grid_disk(h, 1)
            peso_viz = sum(mapa_inc.get(v, 0) for v in vizinhos if v != h) * 0.4
            suavizado.append(inc + peso_viz)
        return df_fato.with_columns([pl.Series("RISCO_GEO", suavizado)])

    def detectar_anomalias_estatisticas(self, df_fato):
        arr_inc = df_fato.select("INCIDENTES").to_numpy().ravel()
        media = np.mean(arr_inc)
        desvio = np.std(arr_inc)
        if desvio > 0:
            pontuacoes_z = np.abs((arr_inc - media) / desvio)
            self.anomalias_detectadas = int(np.sum(pontuacoes_z > 3))
        else:
            self.anomalias_detectadas = 0

    def compilar_relatorio_operacional(self, margem_acerto):
        print("[SINTESE_DADOS] Compilando relatório operacional e executivo.", flush=True)
        tempo_decorrido = round((datetime.now() - self.hoje).total_seconds(), 2)
        data_formatada = self.hoje.strftime('%Y-%m-%d %H:%M:%S')
        
        relatorio = f"""# 📊 Relatório Executivo e Operacional - Sistema SafeDriver

## 1. Resumo Executivo
* **Data da Operação:** {data_formatada}
* **Status do Sistema:** OPERACIONAL
* **Precisão do Modelo Preditivo (R²):** {margem_acerto:.4f}
* **Anomalias Estatísticas Críticas Isoladas:** {self.anomalias_detectadas}

---

## 2. Funil de Processamento de Dados (Data Pipeline)
* **[Camada Bruta] Total de Registros Capturados:** {self.registros_brutos}
* **[Filtros de Qualidade] Registros Descartados (Duplicidade/Erro Geo):** {self.registros_descartados}
* **[Camada Prata] Registros Únicos Validados:** {self.registros_validados}

---

## 3. Relatório Operacional Algorítmico
* **Tempo Total de Execução:** {tempo_decorrido} segundos
* **Conformidade de Privacidade (LGPD):** ATIVA (Identificadores permanentemente destruídos).
* **Ciclo Econômico e Sazonalidade:** Ativos e Injetados.
* **Mecanismo de Reconhecimento:** Scanner Colunar Multi-Aba.
* **Mecanismo de Fusão Preditiva:** Ativo (CatBoost Regressor 70% + LightGBM Regressor 30%)

---

## 4. Auditoria de Segurança Criptográfica (SHA-256)
*Garantia de imutabilidade dos dados extraídos da fonte oficial.*
"""
        for arquivo, assinatura in self.assinaturas_seguranca.items():
            relatorio += f"* **{arquivo}**: `{assinatura}`\n"
            
        caminho_arquivo = self.pastas["relatorios"] / "relatorio_executivo_safedriver.md"
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write(relatorio)
            
        return relatorio

    def executar_predicao_inteligente(self, df):
        print("[NUCLEO_PREDITIVO] Extraindo tensores temporais, ciclo de pagamento e calibrando rede preditiva...", flush=True)
        col_crime = 'NATUREZA_APURADA' if 'NATUREZA_APURADA' in df.columns else 'RUBRICA'
        
        df_engenharia = df.with_columns([
            pl.col(col_crime).str.contains("ROUBO").alias("IS_ROUBO"),
            pl.col("DATA_DT").dt.month().alias("MES"),
            pl.col("DATA_DT").dt.weekday().alias("DIA_SEMANA"),
            pl.col("DATA_DT").dt.hour().alias("HORA"),
            pl.col("DATA_DT").dt.date().alias("DATA_CURTA"),
            pl.col("DESCR_TIPOLOCAL").str.to_uppercase().str.contains("VIA PÚBLICA").cast(pl.Int8).alias("IS_VIA_PUBLICA")
        ]).with_columns([
            pl.col("DIA_SEMANA").is_in([6, 7]).cast(pl.Int8).alias("IS_FIM_SEMANA"),
            pl.col("DATA_CURTA").is_in(self.datas_feriados).cast(pl.Int8).alias("IS_FERIADO")
        ])

        df_engenharia = self.identificar_ciclo_pagamento(df_engenharia)

        fato = (
            df_engenharia.group_by(["H3", "MES", "DIA_SEMANA", "HORA", "IS_FIM_SEMANA", "IS_FERIADO", "FLG_DIA_PAGAMENTO"])
            .agg([
                pl.col("LAT").mean().alias("LAT"), 
                pl.col("LON").mean().alias("LON"), 
                pl.len().alias("INCIDENTES"),
                pl.col("IS_VIA_PUBLICA").sum().alias("QTD_VIA_PUBLICA")
            ])
        )

        self.detectar_anomalias_estatisticas(fato)
        fato = self.aplicar_suavizacao_vizinhanca(fato)

        colunas_modelo = ['LAT', 'LON', 'MES', 'HORA', 'IS_FIM_SEMANA', 'IS_FERIADO', 'FLG_DIA_PAGAMENTO', 'QTD_VIA_PUBLICA', 'RISCO_GEO']
        X_df = fato.select(colunas_modelo).to_pandas()
        X = StandardScaler().fit_transform(X_df)
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        
        modelo_primario = CatBoostRegressor(iterations=150, thread_count=-1, silent=True).fit(X, y)
        modelo_secundario = LGBMRegressor(n_estimators=100, n_jobs=-1, importance_type='gain').fit(X, y)
        
        predicao = (modelo_primario.predict(X) * 0.7) + (modelo_secundario.predict(X) * 0.3)
        margem_acerto = r2_score(y, predicao)
        
        manifesto = {
            "registro_tempo": self.hoje.isoformat(),
            "precisao_modelo": float(margem_acerto),
            "volumetria_bruta": self.registros_brutos,
            "registros_descartados_filtros": self.registros_descartados,
            "total_ocorrencias_validadas": self.registros_validados,
            "anomalias_estatisticas_isoladas": self.anomalias_detectadas,
            "assinaturas_seguranca": self.assinaturas_seguranca,
            "versao_sistema": "6.6.0-seguranca-maxima"
        }
        with open(self.pastas["auditoria"] / "auditoria.json", "w") as f:
            json.dump(manifesto, f, indent=4)

        relatorio_markdown = self.compilar_relatorio_operacional(margem_acerto)

        resultados = np.round(np.expm1(predicao), 2)
        fato = fato.with_columns([pl.Series("PREVISAO_FINAL", resultados)])
        fato.write_parquet(self.pastas["refinado"] / "dashboard_final.parquet")
        
        print("[SINCRONIZACAO_REMOTA] Transmitindo pacotes processados e relatórios para armazenamento.", flush=True)
        balde_nuvem = self.storage_client.bucket(self.bucket_nome)
        
        for f in self.raiz.rglob("datalake/*/*"):
            if f.is_file():
                # Correção de Caminhos: Garante envio seguro (.as_posix) impedindo quebras de diretório em nuvem
                caminho_blob = f.relative_to(self.raiz).as_posix()
                balde_nuvem.blob(caminho_blob).upload_from_filename(str(f))

        webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        if webhook_sucesso:
            print("[TRANSMISSAO_DISCORD] Enviando relatório executivo via webhook.", flush=True)
            carga_util = dict(content=relatorio_markdown[:1950])
            requests.post(webhook_sucesso, json=carga_util)

if __name__ == "__main__":
    try:
        motor = MotorSafeDriverCloud()
        motor.executar_extracao_dados()
        df_processado = motor.executar_limpeza_dados()
        motor.executar_predicao_inteligente(df_processado)
    except Exception:
        erro = traceback.format_exc()
        print("[ALERTA_CRITICO] Exceção não tratada capturada durante a execução da rotina:", flush=True)
        print(erro, flush=True)
        webhook_erro = os.environ.get("DISCORD_ERRO")
        if webhook_erro:
            nl = chr(10)
            msg_erro = dict(content="[FALHA_SISTEMA] Interrupção no ciclo preditivo:" + nl + str(erro)[:1800])
            try: requests.post(webhook_erro, json=msg_erro)
            except: pass
        sys.exit(1)
