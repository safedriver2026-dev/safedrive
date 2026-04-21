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
import hashlib
import csv
from datetime import datetime
from botocore.config import Config
from pathlib import Path

class ArquitetoSafeDriverPrata:
    def __init__(self):
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        self.temp_dir = "./temp_duckdb"
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
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

        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"PRAGMA memory_limit='4GB';")
        self.con.execute(f"PRAGMA temp_directory='{self.temp_dir}';")
        
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except: pass 
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now())}

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE"]: 
            return "NAO INFORMADO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _buscar_arquivo_seguro(self, padrao, nome):
        arqs = glob.glob(f"{self.bronze_dir}/**/{padrao}", recursive=True)
        if not arqs: raise FileNotFoundError(f"{nome} ({padrao}) ausente!")
        return arqs[0]

    def _analisar_csv_dinamicamente(self, path):
        encs = ['utf-8', 'windows-1252', 'latin1']
        with open(path, 'rb') as f: sample = f.read(100000)
        enc_final = 'utf-8'
        for e in encs:
            try:
                sample.decode(e); enc_final = e; break
            except: continue
        sep = ";"
        try:
            sep = csv.Sniffer().sniff(sample.decode(enc_final), delimiters=";,|").delimiter
        except: pass
        return enc_final, sep

    def _garantir_osmconf(self):
        path = os.path.join(self.bronze_dir, "osmconf.ini")
        if not os.path.exists(path):
            with open(path, "w") as f: 
                f.write("[lines]\nosm_id=yes\nattributes=name,highway\n")
        return path.replace("\\", "/")

    def download_r2(self):
        print("Sincronizando Bronze...", flush=True)
        targets = ["SP_Faces_2022.zip", "sp-latest.osm.pbf", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        pag = self.s3.get_paginator('list_objects_v2')
        for p in pag.paginate(Bucket=self.bucket):
            for obj in p.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    dest = os.path.join(self.bronze_dir, key.split('/')[-1])
                    if not os.path.exists(dest):
                        self.s3.download_file(self.bucket, key, dest)
        print("Bronze carregada.", flush=True)

    def processar(self):
        print("Processando Malha Geografica Hierarquica e Micro-Populacao...", flush=True)
        try:
            zip_f = self._buscar_arquivo_seguro("SP_Faces_2022.zip", "IBGE Faces")
            if not glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True):
                with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            
            json_files = glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True)
            osm_pbf = self._buscar_arquivo_seguro("sp-latest.osm.pbf", "OSM PBF")
            conf = self._garantir_osmconf()

            # 1. Captura de dados mantendo a hierarquia Cidade > Bairro > Rua
            sqls = []
            for f in json_files:
                f_path = str(f).replace("\\", "/")
                # No IBGE Faces, buscamos a hierarquia completa
                sqls.append(f"""
                    SELECT 
                        COALESCE(NM_MUN, 'NAO INFORMADO') as CIDADE,
                        COALESCE(NM_BAIRRO, 'NAO INFORMADO') as BAIRRO,
                        trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, 
                        CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, 
                        CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON, 
                        TRY_CAST(TOT_RES AS FLOAT) as TOT_RES 
                    FROM ST_Read('{f_path}')
                """)
            
            # Query consolidada (OSM entra como 'NAO INFORMADO' para Bairro/Cidade pois exige join espacial)
            query = f"""
                SELECT 'NAO INFORMADO' as CIDADE, 'NAO INFORMADO' as BAIRRO, name as RUA, 
                       CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, 
                       CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON, 0.0 as TOT_RES
                FROM ST_Read('{osm_pbf}', layer='lines', open_options=['CONFIG_FILE={conf}', 'INTERLEAVED_READING=YES']) 
                WHERE highway IS NOT NULL AND name IS NOT NULL
                UNION ALL {" UNION ALL ".join(sqls)}
            """
            df = self.con.execute(query).pl()
            
            # Normalização de texto para as chaves da hierarquia
            df = df.with_columns([
                pl.col("CIDADE").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("BAIRRO").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("RUA").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])
            
            # Geração dos índices H3
            df = df.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(
                    lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])
                ).alias("H3_INDEX")
            )
            
            # --- CONSTRUÇÃO DA MALHA HIERÁRQUICA ---
            # Agrupamos H3s únicos por Rua, depois aninhamos Ruas em Bairros e Bairros em Cidades
            df_vias_hierarquico = (
                df.group_by(["CIDADE", "BAIRRO", "RUA"])
                .agg(pl.col("H3_INDEX").unique().alias("H3_LIST"))
                .group_by(["CIDADE", "BAIRRO"])
                .agg(
                    pl.struct([
                        pl.col("RUA"),
                        pl.col("H3_LIST")
                    ]).alias("LOGRADOUROS")
                )
                .group_by("CIDADE")
                .agg(
                    pl.struct([
                        pl.col("BAIRRO"),
                        pl.col("LOGRADOUROS")
                    ]).alias("BAIRROS")
                )
            )

            # Salvando a malha mestre hierárquica
            df_vias_hierarquico.write_parquet(
                f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", 
                compression="zstd"
            )

            # Salvando Micro-População em formato flat (essencial para join direto via H3 na Ouro)
            df_micro_pop = df.group_by("H3_INDEX").agg(pl.sum("TOT_RES").fill_null(0).alias("MICRO_POPULACAO_H3"))
            df_micro_pop.write_parquet(f"{self.prata_dir}/PRATA_MALHA_MICRO_POPULACAO.parquet", compression="zstd")

            self.auditoria["MALHA_GEOGRAFICA"] = "OK - HIERARQUICA"
        except Exception as e: 
            self.auditoria["MALHA_GEOGRAFICA"] = str(e)
            print(f"Erro no processamento da malha: {e}")

        # --- PROCESSAMENTO SOCIAL E INFRA (Mantido conforme original) ---
        print("Processando Malha Social Macro...", flush=True)
        try:
            csv_f = self._buscar_arquivo_seguro("Agregados_por_setores_basico*.csv", "Censo")
            enc, sep = self._analisar_csv_dinamicamente(csv_f)
            df_social = pl.read_csv(csv_f, separator=sep, encoding=enc, null_values=["."], infer_schema_length=10000).filter(pl.col("CD_SETOR").cast(pl.Utf8).str.starts_with("35"))
            df_social = df_social.with_columns([
                pl.col("NM_MUN").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("NM_BAIRRO").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])
            df_social = df_social.select([c for c in ["CD_SETOR", "NM_MUN", "NM_BAIRRO", "v0001", "v0002"] if c in df_social.columns])
            df_social.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")
            self.auditoria["SOCIAL_MACRO"] = "OK"
        except Exception as e: self.auditoria["SOCIAL_MACRO"] = str(e)

        print("Processando Malha Infra...", flush=True)
        try:
            pqs = glob.glob(f"{self.bronze_dir}/**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            lf = pl.scan_parquet(pqs)
            lf = lf.select([c for c in ["lat", "lon", "cnae_fiscal_principal"] if c in lf.collect_schema().names()]).filter(pl.col("lat").is_not_null())
            lf = lf.with_columns(pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV"))
            df_reduced = lf.group_by(["lat", "lon", "CNAE_DIV"]).len().collect(engine="streaming")
            
            unique_coords = df_reduced.select(["lat", "lon"]).unique().with_columns(
                pl.struct(["lat", "lon"]).map_batches(
                    lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])
                ).alias("H3_INDEX")
            )
            df_pivot = df_reduced.join(unique_coords, on=["lat", "lon"]).group_by(["H3_INDEX", "CNAE_DIV"]).agg(pl.sum("len").alias("TOTAL")).pivot(values="TOTAL", index="H3_INDEX", on="CNAE_DIV").fill_null(0)
            df_pivot = df_pivot.rename({c: f"INFRA_DIV_{c}" for c in df_pivot.columns if c != "H3_INDEX"})
            df_pivot.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet", compression="zstd")
            self.auditoria["INFRA"] = f"OK - {df_pivot.height} hexagonos"
        except Exception as e: self.auditoria["INFRA"] = str(e)

    def finalizar(self):
        r2_dest_path = "datalake/prata/malha_trusted"
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f: 
            json.dump(self.auditoria, f, indent=4)
        
        print(f"Exportando Malha Trusted para R2...", flush=True)
        for f in os.listdir(self.prata_dir):
            if f.endswith((".parquet", ".json")):
                self.s3.upload_file(os.path.join(self.prata_dir, f), self.bucket, f"{r2_dest_path}/{f}")
        print("Pipeline Prata Finalizado!", flush=True)

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
