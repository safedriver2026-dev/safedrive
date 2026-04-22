import os
import boto3
import duckdb
import polars as pl
import h3
import unicodedata
import zipfile
import json
import glob
import re
import csv
import io
import shutil
import time
import requests
from datetime import datetime
from botocore.config import Config

class ArquitetoSafeDriverPrata:
    def __init__(self):
        # Configurações de Resolução H3 e Diretórios
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        self.temp_extract_dir = "./data_raw/extracted_json"
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        os.makedirs(self.temp_extract_dir, exist_ok=True)
        
        # Configuração Cloudflare R2 / S3
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
        
        self.webhook_url = os.getenv("DISCORD_SUCESSO")

        # Inicialização do DuckDB
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("PRAGMA memory_limit='5GB';")
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível carregar extensão spatial: {e}")
        
        self.auditoria = {
            "projeto": "SafeDriver - Malha Híbrida Equalizada (V2)",
            "data_execucao": str(datetime.now()),
            "status_pipeline": "PROCESSANDO",
            "telemetria": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except:
                pass

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE", "NAO INFORMADO"]: 
            return "DESCONHECIDO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', texto.upper()).strip()

    def _buscar_arquivo_flexivel(self, padrao):
        arqs = glob.glob(f"**/{padrao}", recursive=True)
        if not arqs:
            raise FileNotFoundError(f"Arquivo ({padrao}) não localizado!")
        return arqs[0]

    def _ler_csv_agnostico(self, filepath):
        encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
        for enc in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    sample = f.read(5000)
                sep = ';' if sample.count(';') > sample.count(',') else ','
                return pl.read_csv(filepath, separator=sep, encoding='utf8' if enc=='utf-8' else 'iso-8859-1', 
                                 infer_schema_length=0, ignore_errors=True)
            except:
                continue
        raise ValueError(f"Não foi possível ler o CSV {filepath}")

    def download_r2(self):
        print("📥 Sincronizando Bronze do R2...", flush=True)
        targets = ["SP_Faces_2022.zip", "SP_bairros_CD2022", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        pag = self.s3.get_paginator('list_objects_v2')
        for p in pag.paginate(Bucket=self.bucket):
            for obj in p.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    dest = os.path.join(self.bronze_dir, key.split('/')[-1])
                    if not os.path.exists(dest):
                        print(f"  -> Baixando {key}...")
                        self.s3.download_file(self.bucket, key, dest)
        print("✅ Bronze sincronizada.")

    def processar(self):
        tempo_inicio = time.time()
        print("🚀 Iniciando Processamento Híbrido...", flush=True)
        try:
            # 1. PREPARAR PONTE RELACIONAL (CENSO CSV)
            print("--- Equalizando Ponta 1: CSV do Censo ---")
            csv_f = self._buscar_arquivo_flexivel("Agregados_por_setores_basico*.csv")
            df_censo_raw = self._ler_csv_agnostico(csv_f)
            
            df_censo = df_censo_raw.select([
                pl.col("CD_SETOR").cast(pl.Utf8).str.extract(r"(\d{15})").alias("CD_SETOR"),
                pl.col("NM_MUN").alias("CID_CENSO"),
                pl.col("NM_BAIRRO").alias("BAI_CENSO"),
                pl.col("v0001").str.replace(",", ".").cast(pl.Float64, strict=False).alias("CENSO_POPULACAO"),
                pl.col("v0002").str.replace(",", ".").cast(pl.Float64, strict=False).alias("CENSO_RENDA")
            ]).filter(pl.col("CD_SETOR").is_not_null())

            # 2. CARREGAR RUAS (JSON) E EQUALIZAR CD_SETOR
            print("--- Equalizando Ponta 2: Faces JSON ---")
            zip_f = self._buscar_arquivo_flexivel("SP_Faces_2022.zip")
            list_vias_df = []
            
            with zipfile.ZipFile(zip_f, 'r') as z:
                json_files = [f for f in z.namelist() if f.endswith('.json')]
                batch_size = 50 
                for i in range(0, len(json_files), batch_size):
                    batch = json_files[i:i + batch_size]
                    sqls = []
                    for f in batch:
                        z.extract(f, self.temp_extract_dir)
                        f_path = os.path.join(self.temp_extract_dir, f).replace("\\","/")
                        # Forçamos CD_SETOR para VARCHAR no DuckDB para garantir match no Polars
                        sqls.append(f"SELECT CAST(CD_SETOR AS VARCHAR) as CD_SETOR, trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON, TRY_CAST(TOT_RES AS FLOAT) as TOT_RES FROM ST_Read('{f_path}')")
                    
                    df_batch = self.con.execute(" UNION ALL ".join(sqls)).pl()
                    list_vias_df.append(df_batch)
                    for f in batch:
                        try: os.remove(os.path.join(self.temp_extract_dir, f))
                        except: pass
                    print(f"    -> Extração de Vias: {min(i + batch_size, len(json_files))}/{len(json_files)}", flush=True)

            df_ruas_raw = pl.concat(list_vias_df)
            df_ruas_raw = df_ruas_raw.with_columns(
                pl.col("CD_SETOR").str.extract(r"(\d{15})").alias("CD_SETOR")
            )

            print("--- Cruzando Ruas e Censo (Join Relacional) ---")
            df_ruas_censo = df_ruas_raw.join(df_censo, on="CD_SETOR", how="left")
            
            print("--- Mapeando para H3 ---")
            # H3 v4 usa latlng_to_cell
            df_ruas_h3 = df_ruas_censo.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX")
            )

            # 3. SPATIAL JOIN E FALLBACK
            print("--- Executando Spatial Join Híbrido ---")
            df_h3_centers = df_ruas_h3.group_by("H3_INDEX").agg([
                pl.col("LAT").mean().alias("LAT"), 
                pl.col("LON").mean().alias("LON"),
                pl.col("CID_CENSO").first().alias("CID_CENSO"), 
                pl.col("BAI_CENSO").first().alias("BAI_CENSO")
            ])
            self.con.register("tabela_h3", df_h3_centers.to_arrow())
            
            arq_poligonos = self._buscar_arquivo_flexivel("SP_bairros_CD2022*.shp").replace("\\", "/")
            
            query_espacial = f"""
                SELECT h3.H3_INDEX, 
                COALESCE(ibge.NM_MUN, h3.CID_CENSO, 'DESCONHECIDO') AS CIDADE,
                COALESCE(ibge.NM_BAIRRO, h3.BAI_CENSO, 'DESCONHECIDO') AS BAIRRO,
                CASE WHEN ibge.NM_BAIRRO IS NOT NULL THEN 1 ELSE 0 END as MATCH_POLIGONAL
                FROM tabela_h3 h3 LEFT JOIN ST_Read('{arq_poligonos}') ibge ON ST_Contains(ibge.geom, ST_Point(h3.LON, h3.LAT))
            """
            df_h3_mapeado = self.con.execute(query_espacial).pl()
            sucesso_poligono = df_h3_mapeado.select(pl.sum("MATCH_POLIGONAL")).item()

            df_h3_mapeado = df_h3_mapeado.with_columns([
                pl.col("CIDADE").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("BAIRRO").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])

            df_vias_completo = df_ruas_h3.drop(["CID_CENSO", "BAI_CENSO"]).join(
                df_h3_mapeado.drop("MATCH_POLIGONAL"), on="H3_INDEX", how="left"
            ).with_columns(pl.col("RUA").map_elements(self._normalizar_texto, return_dtype=pl.Utf8))

            # 4. EXPORTAÇÕES
            print("--- Exportando Tabelas Prata ---")
            df_vias_completo.group_by(["CIDADE", "BAIRRO", "RUA"]).agg(pl.col("H3_INDEX").unique().alias("H3_LIST")) \
                .group_by(["CIDADE", "BAIRRO"]).agg(pl.struct([pl.col("RUA"), pl.col("H3_LIST")]).alias("LOGRADOUROS")) \
                .group_by("CIDADE").agg(pl.struct([pl.col("BAIRRO"), pl.col("LOGRADOUROS")]).alias("BAIRROS")) \
                .write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")

            df_social_h3 = df_vias_completo.group_by("H3_INDEX").agg([
                pl.sum("TOT_RES").alias("MICRO_POPULACAO_FACES"),
                pl.mean("CENSO_POPULACAO").fill_null(0).alias("CENSO_MEDIA_V0001"),
                pl.mean("CENSO_RENDA").fill_null(0).alias("CENSO_MEDIA_V0002")
            ])
            df_social_h3.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL_H3.parquet", compression="zstd")

            # 5. INFRAESTRUTURA (CNAEs)
            pqs_infra = glob.glob(f"**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            if pqs_infra:
                print("--- Processando Infraestrutura ---")
                df_pivot = pl.scan_parquet(pqs_infra).filter(pl.col("lat").is_not_null()) \
                    .with_columns(pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV")) \
                    .with_columns(pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])).alias("H3_INDEX")) \
                    .group_by(["H3_INDEX", "CNAE_DIV"]).len().collect(engine="streaming") \
                    .pivot(values="len", index="H3_INDEX", on="CNAE_DIV").fill_null(0)
                
                df_pivot = df_pivot.rename({c: f"INFRA_DIV_{c}" for c in df_pivot.columns if c != "H3_INDEX"})
                df_pivot.join(df_h3_mapeado.select(["H3_INDEX", "CIDADE", "BAIRRO"]), on="H3_INDEX", how="left") \
                    .write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet", compression="zstd")

            # 6. TELEMETRIA FINAL
            tempo_exec = round(time.time() - tempo_inicio, 2)
            sucesso_total = df_h3_mapeado.filter(pl.col("BAIRRO") != "DESCONHECIDO").height
            taxa_sucesso_geral = round((sucesso_total / max(df_h3_mapeado.height, 1)) * 100, 2)
            
            self.auditoria["status_pipeline"] = "SUCESSO"
            self.auditoria["telemetria"] = {
                "h3_unicos": df_h3_mapeado.height,
                "salvos_pelo_censo_csv": sucesso_total - sucesso_poligono,
                "taxa_cobertura_final_pct": taxa_sucesso_geral
            }
            
            msg = (f"🗺️ **[SafeDriver] Malha Prata Híbrida OK**\n"
                   f"- H3 Unicos: {df_h3_mapeado.height}\n"
                   f"- Salvos pelo Censo: {sucesso_total - sucesso_poligono}\n"
                   f"- Cobertura: {taxa_sucesso_geral}%\n"
                   f"- Tempo: {tempo_exec}s")
            
            self._notificar_discord(msg)
            print(f"✅ Processamento Finalizado: {taxa_sucesso_geral}% de cobertura.")

        except Exception as e:
            err_msg = f"❌ **[SafeDriver] Erro Fatal:** {str(e)}"
            print(err_msg)
            self._notificar_discord(err_msg)
            raise e

if __name__ == "__main__":
    arquiteto = ArquitetoSafeDriverPrata()
    # arquiteto.download_r2() # Descomente se precisar baixar os arquivos no runner
    arquiteto.processar()
