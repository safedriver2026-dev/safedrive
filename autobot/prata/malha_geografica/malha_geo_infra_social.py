import os
import boto3
import duckdb
import polars as pl
import h3
import zipfile
import glob
import io
import shutil
import time
from datetime import datetime
from botocore.config import Config

class ProcessadorMalhaPrata:
    def __init__(self):
        self.RES_H3 = 9 
        self.pasta_bronze = "./data_raw"
        self.pasta_prata = "./data_prata"
        self.pasta_temporaria = "./data_raw/extracted_json"
        
        os.makedirs(self.pasta_bronze, exist_ok=True)
        os.makedirs(self.pasta_prata, exist_ok=True)
        os.makedirs(self.pasta_temporaria, exist_ok=True)
        
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )
        
        self.banco = duckdb.connect(database=':memory:')
        self.banco.execute("PRAGMA memory_limit='8GB'; PRAGMA threads=8;")
        self.banco.execute("INSTALL spatial; LOAD spatial;")

    def _limpar_tabela_toda(self, df):
        """Passa o rodo nas colunas de texto: tira acento, bota em maiúsculo e remove caracteres estranhos."""
        cols_texto = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        if not cols_texto: return df
        
        return df.with_columns([
            pl.col(c)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"Ç", "C")
            .str.replace_all(r"[^A-Z0-9\s_]", " ")
            .str.replace_all(r"\s+", " ")
            .fill_null("DESCONHECIDO")
            .alias(c)
            for c in cols_texto
        ])

    def _limpar_codigo_setor(self, coluna):
        return pl.col(coluna).cast(pl.Utf8).str.replace(r"\.0$", "").str.replace_all(r"\D", "").str.slice(0, 15)

    def baixar_arquivos(self):
        print("Baixando os arquivos brutos do armazenamento em nuvem...")
        arquivos_alvo = ["SP_Faces_2022.zip", "SP_bairros_CD2022", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        paginador = self.s3.get_paginator('list_objects_v2')
        for pagina in paginador.paginate(Bucket=self.bucket):
            for arquivo in pagina.get('Contents', []):
                chave = arquivo['Key']
                if any(alvo in chave for alvo in arquivos_alvo):
                    destino = os.path.join(self.pasta_bronze, chave.split('/')[-1])
                    if not os.path.exists(destino):
                        self.s3.download_file(self.bucket, chave, destino)

    def processar(self):
        tempo_inicial = time.time()
        try:
            # 1. CENSO (Leitura otimizada com separador e formatação corretos)
            print("Processando os dados do Censo IBGE...")
            arquivo_csv = glob.glob(f"**/Agregados_por_setores_basico*.csv", recursive=True)[0]
            df_censo = pl.read_csv(arquivo_csv, separator=";", encoding="iso-8859-1", 
                                   infer_schema_length=0, n_threads=8, ignore_errors=True)
            
            df_censo = df_censo.select([
                self._limpar_codigo_setor("CD_SETOR").alias("CD_SETOR"),
                pl.col("NM_MUN").alias("CID_CENSO") if "NM_MUN" in df_censo.columns else pl.lit("DESCONHECIDO").alias("CID_CENSO"),
                pl.col("V0001").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0).alias("CENSO_POPULACAO"),
                pl.col("V0002").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0).alias("CENSO_RENDA")
            ]).pipe(self._limpar_tabela_toda)

            # 2. FACES DE RUA (Extração e leitura em lotes)
            print("Processando as ruas (Extração do JSON)...")
            arquivo_zip = glob.glob(f"**/SP_Faces_2022.zip", recursive=True)[0]
            lista_ruas = []
            with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
                arquivos_json = [f for f in zip_ref.namelist() if f.endswith('.json')]
                for i in range(0, len(arquivos_json), 100):
                    lote = arquivos_json[i:i+100]
                    for f in lote: zip_ref.extract(f, self.pasta_temporaria)
                    
                    sql = f"SELECT CD_SETOR, trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON, TRY_CAST(TOT_RES AS FLOAT) as TOT_RES FROM ST_Read('{self.pasta_temporaria}/*/*.json')"
                    lista_ruas.append(self.banco.execute(sql).pl())
                    
                    for f in lote: shutil.rmtree(os.path.join(self.pasta_temporaria, f.split('/')[0]), ignore_errors=True)

            df_ruas = pl.concat(lista_ruas).with_columns(self._limpar_codigo_setor("CD_SETOR")).pipe(self._limpar_tabela_toda)

            # 3. GERAÇÃO DE H3 E CRUZAMENTO ESPACIAL
            print("Gerando índices hexagonais (H3) e cruzando com o mapa da cidade...")
            df_ruas_h3 = df_ruas.join(df_censo, on="CD_SETOR", how="left").fill_null(0)
            df_ruas_h3 = df_ruas_h3.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.RES_H3) for x in s])).alias("H3_INDEX")
            )

            # Agrupa apenas coordenadas únicas para o cruzamento espacial (muito mais rápido)
            df_h3_geografia = df_ruas_h3.group_by("H3_INDEX").agg([
                pl.col("LAT").first().alias("LAT"), pl.col("LON").first().alias("LON"),
                pl.col("CID_CENSO").first().alias("CID_CENSO")
            ])
            
            self.banco.register("tabela_resumo_h3", df_h3_geografia.to_arrow())
            caminho_shp = glob.glob(f"**/*bairros_CD2022*.shp", recursive=True)[0].replace("\\", "/")
            
            df_bairros_referencia = self.banco.execute(f"""
                SELECT h3.H3_INDEX, 
                COALESCE(ibge.NM_MUN, h3.CID_CENSO) AS CIDADE,
                COALESCE(ibge.NM_BAIRRO, 'DESCONHECIDO') AS BAIRRO
                FROM tabela_resumo_h3 h3 
                LEFT JOIN ST_Read('{caminho_shp}') ibge ON ST_Contains(ibge.geom, ST_Point(h3.LON, h3.LAT))
            """).pl().pipe(self._limpar_tabela_toda)

            # 4. EXPORTAÇÃO FINAL
            print("Finalizando e salvando as tabelas de população e infraestrutura...")
            df_final = df_ruas_h3.drop("CID_CENSO").join(df_bairros_referencia, on="H3_INDEX", how="left")
            
            # Exportando dados sociais
            df_final.group_by("H3_INDEX").agg([
                pl.sum("TOT_RES").alias("MICRO_POPULACAO_FACES"),
                pl.mean("CENSO_POPULACAO").alias("CENSO_MEDIA_V0001"),
                pl.mean("CENSO_RENDA").alias("CENSO_MEDIA_V0002")
            ]).write_parquet(f"{self.pasta_prata}/PRATA_MALHA_SOCIAL_H3.parquet")

            # Exportando empresas e infraestrutura (CNAE)
            arquivos_infra = glob.glob(f"**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            if arquivos_infra:
                df_infra = pl.scan_parquet(arquivos_infra) \
                    .filter(pl.col("lat").is_not_null()) \
                    .with_columns(pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV")) \
                    .collect() \
                    .with_columns(pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.RES_H3) for x in s])).alias("H3_INDEX")) \
                    .group_by(["H3_INDEX", "CNAE_DIV"]).len() \
                    .pivot(values="len", index="H3_INDEX", on="CNAE_DIV").fill_null(0) \
                    .rename({c: f"INFRA_DIV_{c}" for c in df_infra.columns if c != "H3_INDEX"}) \
                    .join(df_bairros_referencia, on="H3_INDEX", how="left") \
                    .pipe(self._limpar_tabela_toda)
                
                df_infra.write_parquet(f"{self.pasta_prata}/PRATA_MALHA_INFRA_AGREGADA.parquet")

            print("Enviando os arquivos processados para a nuvem...")
            for arquivo in glob.glob(f"{self.pasta_prata}/*.parquet"):
                self.s3.upload_file(arquivo, self.bucket, f"datalake/prata/malha_trusted/{os.path.basename(arquivo)}")

            print(f"Sucesso! A Malha Prata foi gerada em {time.time() - tempo_inicial:.2f} segundos.")

        finally:
            self.banco.close()
            if os.path.exists(self.pasta_temporaria): shutil.rmtree(self.pasta_temporaria)

if __name__ == "__main__":
    app = ProcessadorMalhaPrata()
    app.baixar_arquivos()
    app.processar()
