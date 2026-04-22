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
from datetime import datetime
from botocore.config import Config

class ArquitetoSafeDriverPrata:
    def __init__(self):
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        self.temp_extract_dir = "./data_raw/extracted_json"
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        os.makedirs(self.temp_extract_dir, exist_ok=True)
        
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

        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"PRAGMA memory_limit='5GB';")
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except:
            pass 
        
        self.auditoria = {
            "projeto": "SafeDriver - Malha Geográfica",
            "data_execucao": str(datetime.now()),
            "status_pipeline": "PROCESSANDO",
            "telemetria": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE"]: 
            return "NAO INFORMADO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _buscar_arquivo_flexivel(self, padrao):
        print(f"📂 Buscando por: {padrao}...", flush=True)
        arqs = glob.glob(f"**/{padrao}", recursive=True)
        if not arqs:
            raise FileNotFoundError(f"Arquivo ({padrao}) não localizado no disco!")
        return arqs[0]

    def _ler_csv_agnostico(self, filepath):
        """Leitor cego: Descobre encoding e delimitador sozinho antes de ler."""
        print(f"🔍 Auto-detetando formato do CSV: {filepath}", flush=True)
        
        # 1. Descobrir Encoding
        encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        used_enc = 'utf-8'
        sample_text = ""
        for enc in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    sample_text = f.read(10000)
                used_enc = enc
                break
            except Exception:
                continue
                
        # 2. Descobrir Delimitador via Sniffer Estatístico
        try:
            sep = csv.Sniffer().sniff(sample_text).delimiter
        except Exception:
            # Fallback seguro caso o sniffer falhe
            sep = ';' if sample_text.count(';') > sample_text.count(',') else ','
            
        print(f"✅ Identificado -> Encoding: {used_enc} | Separador: '{sep}'", flush=True)
        
        # O Polars usa internamente 'utf8' ou 'iso-8859-1'
        pl_encoding = 'utf8' if used_enc == 'utf-8' else 'iso-8859-1'
        
        # infer_schema_length=0 FORÇA que tudo seja lido como String (blindando o CD_SETOR)
        return pl.read_csv(filepath, separator=sep, encoding=pl_encoding, infer_schema_length=0, ignore_errors=True)

    def download_r2(self):
        print("📥 Sincronizando Bronze do R2...", flush=True)
        targets = ["SP_Faces_2022.zip", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        pag = self.s3.get_paginator('list_objects_v2')
        for p in pag.paginate(Bucket=self.bucket):
            for obj in p.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    dest = os.path.join(self.bronze_dir, key.split('/')[-1])
                    if not os.path.exists(dest):
                        print(f"   -> Baixando: {key}")
                        self.s3.download_file(self.bucket, key, dest)
        print("✅ Bronze carregada localmente.")

    def processar(self):
        tempo_inicio = time.time()
        print("🚀 Iniciando Processamento da Malha Híbrida Hierárquica...", flush=True)
        try:
            # 1. REFERÊNCIA SOCIAL (Agora 100% Agnóstico)
            print("--- Lendo Censo (Referência de Nomes) ---")
            csv_f = self._buscar_arquivo_flexivel("Agregados_por_setores_basico*.csv")
            
            df_social_raw = self._ler_csv_agnostico(csv_f)
            
            df_ref_nomes = df_social_raw.select([
                pl.col("CD_SETOR").str.strip_chars(),
                pl.col("NM_MUN").alias("CIDADE"),
                pl.col("NM_BAIRRO").alias("BAIRRO")
            ]).with_columns([
                pl.col("CIDADE").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("BAIRRO").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])
            total_censo = df_social_raw.height

            # 2. GEOMETRIAS (IBGE Faces)
            print("--- Extraindo e Processando Geometrias (Batch Mode) ---")
            zip_f = self._buscar_arquivo_flexivel("SP_Faces_2022.zip")
            
            list_vias_df = []
            with zipfile.ZipFile(zip_f, 'r') as z:
                json_files = [f for f in z.namelist() if f.endswith('.json')]
                batch_size = 50 
                for i in range(0, len(json_files), batch_size):
                    batch = json_files[i:i + batch_size]
                    sqls = []
                    for f_json in batch:
                        z.extract(f_json, self.temp_extract_dir)
                        f_path_sql = os.path.join(self.temp_extract_dir, f_json).replace("\\", "/")
                        sqls.append(f"""
                            SELECT 
                                TRIM(CAST(CD_SETOR AS VARCHAR)) as CD_SETOR,
                                trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, 
                                ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON, 
                                TRY_CAST(TOT_RES AS FLOAT) as TOT_RES 
                            FROM ST_Read('{f_path_sql}')
                        """)
                    
                    df_batch = self.con.execute(" UNION ALL ".join(sqls)).pl()
                    list_vias_df.append(df_batch)
                    
                    for f_json in batch:
                        try: os.remove(os.path.join(self.temp_extract_dir, f_json))
                        except: pass
                    print(f"   -> Progresso: {min(i + batch_size, len(json_files))}/{len(json_files)} arquivos", flush=True)

            df_faces = pl.concat(list_vias_df)
            total_faces = df_faces.height
            
            print("--- Consolidando Hierarquia e H3 ---")
            df_vias_completo = df_faces.join(df_ref_nomes, on="CD_SETOR", how="left").with_columns([
                pl.col("CIDADE").fill_null("NAO INFORMADO"),
                pl.col("BAIRRO").fill_null("NAO INFORMADO"),
                pl.col("RUA").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX")
            ])

            # Métricas de Qualidade
            faces_sem_cidade = df_vias_completo.filter(pl.col("CIDADE") == "NAO INFORMADO").height
            faces_sem_bairro = df_vias_completo.filter(pl.col("BAIRRO") == "NAO INFORMADO").height
            h3_unicos = df_vias_completo.select("H3_INDEX").n_unique()

            # Exportação 1: Hierarquia
            df_hierarquico = (
                df_vias_completo.group_by(["CIDADE", "BAIRRO", "RUA"]).agg(pl.col("H3_INDEX").unique().alias("H3_LIST"))
                .group_by(["CIDADE", "BAIRRO"]).agg(pl.struct([pl.col("RUA"), pl.col("H3_LIST")]).alias("LOGRADOUROS"))
                .group_by("CIDADE").agg(pl.struct([pl.col("BAIRRO"), pl.col("LOGRADOUROS")]).alias("BAIRROS"))
            )
            df_hierarquico.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")

            # Exportação 2: Micro-População
            df_micro = df_vias_completo.group_by("H3_INDEX").agg(pl.sum("TOT_RES").alias("MICRO_POPULACAO_H3"))
            df_micro.write_parquet(f"{self.prata_dir}/PRATA_MALHA_MICRO_POPULACAO.parquet")

            # Exportação 3: Social
            df_social_raw.select([pl.col("CD_SETOR").cast(pl.Utf8), "NM_MUN", "NM_BAIRRO", "v0001", "v0002"]) \
                .write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet")

            # 4. INFRAESTRUTURA
            print("--- Processando Infraestrutura (Streaming Mode) ---")
            pqs_infra = glob.glob(f"**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            total_cnpjs = 0
            df_pivot_height = 0
            
            if pqs_infra:
                df_infra = pl.scan_parquet(pqs_infra).filter(pl.col("lat").is_not_null()).with_columns([
                    pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV")
                ]).group_by(["lat", "lon", "CNAE_DIV"]).len().collect(engine="streaming")
                
                total_cnpjs = df_infra.select(pl.sum("len")).item()

                df_infra_h3 = df_infra.with_columns(pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])).alias("H3_INDEX"))
                df_pivot = df_infra_h3.group_by(["H3_INDEX", "CNAE_DIV"]).agg(pl.sum("len").alias("TOTAL")).pivot(values="TOTAL", index="H3_INDEX", on="CNAE_DIV").fill_null(0)
                df_pivot = df_pivot.rename({c: f"INFRA_DIV_{c}" for c in df_pivot.columns if c != "H3_INDEX"})
                df_pivot.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet", compression="zstd")
                df_pivot_height = df_pivot.height

            # --- POPULANDO A AUDITORIA ---
            tempo_exec = round(time.time() - tempo_inicio, 2)
            self.auditoria["status_pipeline"] = "SUCESSO"
            self.auditoria["telemetria"] = {
                "processamento": {
                    "tempo_total_segundos": tempo_exec,
                    "resolucao_h3_utilizada": self.H3_RES
                },
                "censo_ibge": {
                    "setores_lidos_csv": total_censo
                },
                "geometrias_faces": {
                    "poligonos_extraidos": total_faces,
                    "h3_unicos_gerados": h3_unicos
                },
                "saude_do_join_espacial": {
                    "faces_com_cidade_invalida": faces_sem_cidade,
                    "faces_com_bairro_invalido": faces_sem_bairro,
                    "taxa_sucesso_relacionamento_pct": round(((total_faces - faces_sem_bairro) / total_faces) * 100, 2) if total_faces > 0 else 0
                },
                "infraestrutura_urbana": {
                    "cnpjs_agregados": total_cnpjs,
                    "h3_com_atividade_comercial": df_pivot_height
                },
                "tabelas_exportadas_linhas": {
                    "PRATA_MALHA_GEOGRAFICA_VIAS": df_hierarquico.height,
                    "PRATA_MALHA_MICRO_POPULACAO": df_micro.height,
                    "PRATA_MALHA_INFRA_AGREGADA": df_pivot_height
                }
            }

            print("✨ Pipeline Prata concluído com sucesso!")
            
            # Notificação no Discord
            msg = "🗺️ **[SafeDriver] Malha Geográfica Prata Atualizada**\n```ml\n"
            msg += f"• H3 Únicos Gerados: {h3_unicos}\n"
            msg += f"• Taxa Sucesso IBGE : {self.auditoria['telemetria']['saude_do_join_espacial']['taxa_sucesso_relacionamento_pct']}%\n"
            msg += f"• CNPJs Mapeados   : {total_cnpjs}\n"
            msg += f"• Tempo Execução   : {tempo_exec}s\n```"
            self._notificar_discord(msg)
            
        except Exception as e:
            self.auditoria["status_pipeline"] = f"ERRO: {str(e)}"
            print(f"❌ Erro Crítico: {e}")
        finally:
            shutil.rmtree(self.temp_extract_dir, ignore_errors=True)

    def finalizar(self):
        r2_dest_path = "datalake/prata/malha_trusted"
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f: 
            json.dump(self.auditoria, f, indent=4)
        print(f"📤 Exportando Malha para R2...")
        for f in os.listdir(self.prata_dir):
            if f.endswith((".parquet", ".json")):
                self.s3.upload_file(os.path.join(self.prata_dir, f), self.bucket, f"{r2_dest_path}/{f}")

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
