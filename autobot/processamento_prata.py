import polars as pl
import boto3
import io
import os
import h3
import logging
from datetime import datetime
from autobot.dicionario_ssp import identificar_coluna_pela_descricao

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self.h3_resolution = 8
        
        self.colunas_lgpd = [
            "NUM_BO", "LOGRADOURO", "NUMERO_LOGRADOURO", 
            "NOME_DELEGACIA", "NOME_DEPARTAMENTO", "NOME_SECCIONAL",
            "NOME_DELEGACIA_CIRCUNSCRICAO", "NOME_DEPARTAMENTO_CIRCUNSCRICAO",
            "NOME_SECCIONAL_CIRCUNSCRICAO", "NOME_MUNICIPIO_CIRCUNSCRICAO"
        ]

        self.mapa_personas = {
            "MOTORISTA": [
                "FURTO DE VEÍCULO", "ROUBO DE VEÍCULO", 
                "HOMICÍDIO CULPOSO POR ACIDENTE DE TRÂNSITO",
                "LESÃO CORPORAL CULPOSA POR ACIDENTE DE TRÂNSITO",
                "HOMICÍDIO DOLOSO POR ACIDENTE DE TRÂNSITO"
            ],
            "PEDESTRE": [
                "FURTO - OUTROS", "ROUBO - OUTROS", "LATROCÍNIO",
                "ESTUPRO", "ESTUPRO DE VULNERÁVEL", "HOMICÍDIO DOLOSO",
                "LESÃO CORPORAL DOLOSA", "TENTATIVA DE HOMICÍDIO"
            ]
        }

    def executar_prata(self, ano):
        path_bronze = f"datalake/bronze/ssp_raw_{ano}.xlsx"
        path_geo = "datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"

        try:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_prata)
                logger.info(f"PRATA: O ano {ano} ja esta consolidado.")
                return True
            except:
                pass

            logger.info(f"PRATA: Iniciando processamento do ano {ano}...")
            resp_ssp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            conteudo_excel = resp_ssp['Body'].read()

            mapa_colunas_dinamico = self._processar_dicionario_capa(conteudo_excel)

            lf_ssp = self._extrair_dados_consolidados(conteudo_excel, mapa_colunas_dinamico)
            
            if lf_ssp is None or lf_ssp.is_empty():
                raise ValueError(f"Nenhum dado real extraido para o ano {ano}")

            lf_ssp = lf_ssp.drop([c for c in self.colunas_lgpd if c in lf_ssp.columns])

            resp_geo = self.s3.get_object(Bucket=self.bucket, Key=path_geo)
            lf_geo = pl.read_parquet(io.BytesIO(resp_geo['Body'].read()))

            lf_ssp = self._geocodificar_h3(lf_ssp)

            lf_crimes = self._agregar_por_persona(lf_ssp)

            lf_final = lf_crimes.join(lf_geo, on="H3_INDEX", how="inner")
            lf_final = self._aplicar_indicadores(lf_final, ano)

            buffer = io.BytesIO()
            lf_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            logger.info(f"PRATA: Dados de {ano} consolidados com sucesso.")
            return True

        except Exception as e:
            logger.error(f"Erro no processamento Prata {ano}: {e}")
            raise e

    def _processar_dicionario_capa(self, conteudo):
        mapa_colunas_dinamico = {} 
        try:
            df_capa = None
            for i in [1, 0, 2]:
                try:
                    temp_df = pl.read_excel(io.BytesIO(conteudo), sheet_id=i)
                    if len(temp_df.columns) <= 3: 
                        df_capa = temp_df
                        break
                except:
                    continue

            if df_capa is not None:
                for row in df_capa.iter_rows():
                    linha = [str(x).strip() for x in row if x is not None]
                    if len(linha) >= 2:
                        nome_excel_ssp = linha[0].upper().strip()
                        descricao_ssp = linha[1]
                        
                        nome_canonico = identificar_coluna_pela_descricao(descricao_ssp)
                        
                        if nome_canonico:
                            mapa_colunas_dinamico[nome_excel_ssp] = nome_canonico
                            
            logger.info(f"PRATA: Mapeamento Semantico gerado via Capa ({len(mapa_colunas_dinamico)} colunas identificadas).")
        except Exception as e:
            logger.warning(f"PRATA: Falha ao ler dicionario da capa: {e}")
            
        return mapa_colunas_dinamico 

    def _extrair_dados_consolidados(self, conteudo, mapa_colunas_dinamico):
        lista_dfs = []
        colunas_canonicamente_esperadas = set(mapa_colunas_dinamico.values()) if mapa_colunas_dinamico else set()
        lixo_coordenadas = ["0", "0.0", "0,0", "NULL", "NA", "", " ", "-", "NaN", "nan"]

        for i in range(1, 15):
            try:
                df = pl.read_excel(io.BytesIO(conteudo), sheet_id=i)
                
                cols_originais = df.columns
                cabecalhos_atuais = {}
                colunas_vistas = set()
                
                for c in cols_originais:
                    limpo = c.upper().strip()
                    traduzido = mapa_colunas_dinamico.get(limpo, limpo)
                    
                    if traduzido in colunas_vistas:
                        traduzido = f"{traduzido}_DUPLICADA_{len(colunas_vistas)}"
                    
                    colunas_vistas.add(traduzido)
                    cabecalhos_atuais[c] = traduzido

                df = df.rename(cabecalhos_atuais)
                
                is_aba_valida = False
                
                if colunas_canonicamente_esperadas:
                    match_colunas = len(set(df.columns).intersection(colunas_canonicamente_esperadas))
                    if match_colunas >= 10:
                        is_aba_valida = True
                else:
                    if all(c in df.columns for c in ["LATITUDE", "LONGITUDE", "NATUREZA_APURADA"]):
                        is_aba_valida = True

                if is_aba_valida:
                    df = df.with_columns(pl.all().cast(pl.Utf8))
                    
                    # Filtro pesado contra hifens, espacos e zeros
                    df = df.filter(
                        (pl.col("LATITUDE").is_not_null()) & 
                        (pl.col("LONGITUDE").is_not_null()) &
                        (~pl.col("LATITUDE").str.strip_chars().is_in(lixo_coordenadas)) &
                        (~pl.col("LONGITUDE").str.strip_chars().is_in(lixo_coordenadas))
                    )
                    
                    lista_dfs.append(df)
                    logger.info(f"PRATA: Aba {i} validada e extraida com sucesso.")
                    
            except Exception as e:
                mensagem_erro = str(e).lower()
                # Mata os Warnings de abas inexistentes
                if "no matching sheet" in mensagem_erro or "out of bounds" in mensagem_erro:
                    logger.info(f"PRATA: Fim do arquivo. Nao existem mais abas apos a {i-1}.")
                    break 
                else:
                    logger.warning(f"PRATA: Aba {i} ignorada devido a erro interno: {e}")
                    continue
        
        if lista_dfs:
            return pl.concat(lista_dfs, how="diagonal")
        return None

    def _geocodificar_h3(self, lf):
        df_pd = lf.to_pandas()
        
        # Funcao blindada para conversao
        def safe_h3(lat, lng):
            try:
                # Troca virgula por ponto caso venha no formato brasileiro
                lat_float = float(str(lat).replace(",", ".").strip())
                lng_float = float(str(lng).replace(",", ".").strip())
                return h3.latlng_to_cell(lat_float, lng_float, self.h3_resolution)
            except Exception:
                return None
                
        df_pd['H3_INDEX'] = df_pd.apply(
            lambda r: safe_h3(r['LATITUDE'], r['LONGITUDE']), axis=1
        )
        
        # Remove eventuais linhas que ainda falharam na conversao
        df_pd = df_pd.dropna(subset=['H3_INDEX'])
        
        return pl.from_pandas(df_pd)

    def _agregar_por_persona(self, lf):
        lf = lf.with_columns([
            pl.when(
                (pl.col("RUBRICA").str.contains("(?i)Moto|Motocicleta")) | 
                (pl.col("DESCR_CONDUTA").str.contains("(?i)Moto|Motocicleta"))
            ).then(pl.lit("MOTOCICLISTA"))
            .when(pl.col("NATUREZA_APURADA").is_in(self.mapa_personas["MOTORISTA"]))
            .then(pl.lit("MOTORISTA"))
            .when(pl.col("NATUREZA_APURADA").is_in(self.mapa_personas["PEDESTRE"]))
            .then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("OUTROS"))
            .alias("CATEGORIA_PERSONA")
        ])

        return lf.group_by("H3_INDEX").agg([
            pl.col("H3_INDEX").filter(pl.col("CATEGORIA_PERSONA") == "MOTORISTA").count().alias("TOTAL_CRIMES_MOTORISTA"),
            pl.col("H3_INDEX").filter(pl.col("CATEGORIA_PERSONA") == "PEDESTRE").count().alias("TOTAL_CRIMES_PEDESTRE"),
            pl.col("H3_INDEX").filter(pl.col("CATEGORIA_PERSONA") == "MOTOCICLISTA").count().alias("TOTAL_CRIMES_MOTOCICLISTA")
        ])

    def _aplicar_indicadores(self, lf, ano):
        return lf.with_columns([
            pl.lit(ano).alias("ANO_REFERENCIA"),
            pl.col("TOTAL_NAO_RESIDENCIAIS_H3").alias("TOTAL_NAO_RES_H3"),
            pl.col("PROPORCAO_RESIDENCIAL_H3").alias("INDICE_RESIDENCIAL"),
            pl.col("DENSIDADE_LOGRADOUROS").alias("DENSIDADE_ENDERECOS")
        ]).fill_null(0)
