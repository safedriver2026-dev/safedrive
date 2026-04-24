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
            "projeto": "SafeDriver - Malha Híbrida Equalizada",
            "data_execucao": str(datetime.now()),
            "status_pipeline": "PROCESSANDO",
            "telemetria": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE", "NAO INFORMADO"]: 
            return "DESCONHECIDO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _buscar_arquivo_flexivel(self, padrao):
        print(f"📂 Buscando por: {padrao}...", flush=True)
        arqs = glob.glob(f"**/{padrao}", recursive=True)
        if not arqs:
            raise FileNotFoundError(f"Arquivo ({padrao}) não localizado no disco!")
        return arqs[0]

    def _ler_csv_agnostico(self, filepath):
        print(f"🔍 Auto-detetando formato do CSV: {filepath}", flush=True)
        encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        used_enc = 'utf-8'
        sample_text = ""
        for enc in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    sample_text = f.read(10000)
                used_enc = enc
                break
            except:
                continue
        try:
            sep = csv.Sniffer().sniff(sample_text).delimiter
        except:
            sep = ';' if sample_text.count(';') > sample_text.count(',') else ','
        pl_encoding = 'utf8' if used_enc == 'utf-8' else 'iso-8859-1'
        return pl.read_csv(filepath, separator=sep, encoding=pl_encoding, infer_schema_length=0, ignore_errors=True)

    def _limpar_cd_setor(self, coluna):
        """Força o CD_SETOR a ser uma string limpa de 15 dígitos, matando floats e sufixos"""
        return pl.col(coluna).cast(pl.Utf8) \
                 .str.replace(r"\.0$", "") \
                 .str.replace_all(r"\D", "") \
                 .str.slice(0, 15)

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
                        print(f"   -> Baixando: {key}")
                        self.s3.download_file(self.bucket, key, dest)
        print("✅ Bronze carregada localmente.")

    def upload_r2(self):
        """Faz o upload de todos os arquivos da pasta Prata para o bucket R2 na nova hierarquia"""
        print("📤 Enviando arquivos da Malha Prata para o R2...", flush=True)
        arquivos_prata = glob.glob(f"{self.prata_dir}/*")
        
        if not arquivos_prata:
            print("⚠️ Nenhum arquivo encontrado na pasta Prata para upload.")
            return

        for filepath in arquivos_prata:
            filename = os.path.basename(filepath)
            # PADRONIZAÇÃO DO DATA LAKE APLICADA AQUI:
            s3_key = f"datalake/prata/malha_trusted/{filename}"
            print(f"   -> Subindo: {filename} para {s3_key}...", flush=True)
            try:
                self.s3.upload_file(filepath, self.bucket, s3_key)
            except Exception as e:
                print(f"❌ Erro ao subir {filename}: {str(e)}")
                raise e
        print("✅ Upload da camada Prata concluído com sucesso!")

    def processar(self):
        tempo_inicio = time.time()
        print("🚀 Iniciando Processamento Equalizado (JSON + CSV + SHP)...", flush=True)
        try:
            # =================================================================
            # 1. PREPARAR PONTE RELACIONAL (CENSO CSV)
            # =================================================================
            print("--- Equalizando Ponta 1: CSV do Censo ---")
            csv_f = self._buscar_arquivo_flexivel("Agregados_por_setores_basico*.csv")
            df_censo_raw = self._ler_csv_agnostico(csv_f)
            
            df_censo_raw = df_censo_raw.rename({c: c.strip().upper() for c in df_censo_raw.columns})
            cols = df_censo_raw.columns

            colunas_localidade = [c for c in ["NM_BAIRRO", "NM_SUBDIST", "NM_DISTR"] if c in cols]
            
            if colunas_localidade:
                bairro_expr = pl.coalesce([
                    pl.when(pl.col(c).cast(pl.Utf8).str.strip_chars() == "").then(None).otherwise(pl.col(c)) 
                    for c in colunas_localidade
                ]).fill_null("DESCONHECIDO")
            else:
                bairro_expr = pl.lit("DESCONHECIDO")

            df_censo = df_censo_raw.select([
                self._limpar_cd_setor("CD_SETOR").alias("CD_SETOR"),
                pl.col("NM_MUN").alias("CID_CENSO") if "NM_MUN" in cols else pl.lit("DESCONHECIDO").alias("CID_CENSO"),
                bairro_expr.alias("BAI_CENSO"),
                pl.col("V0001").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).alias("CENSO_POPULACAO") if "V0001" in cols else pl.lit(0.0).alias("CENSO_POPULACAO"),
                pl.col("V0002").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).alias("CENSO_RENDA") if "V0002" in cols else pl.lit(0.0).alias("CENSO_RENDA")
            ]).filter(pl.col("CD_SETOR").str.len_chars() == 15)

            # =================================================================
            # 2. CARREGAR RUAS (JSON) E EQUALIZAR CD_SETOR
            # =================================================================
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
                        path_json = os.path.join(self.temp_extract_dir, f).replace('\\','/')
                        sqls.append(f"SELECT CD_SETOR, trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON, TRY_CAST(TOT_RES AS FLOAT) as TOT_RES FROM ST_Read('{path_json}')")
                    
                    df_batch = self.con.execute(" UNION ALL ".join(sqls)).pl()
                    list_vias_df.append(df_batch)
                    for f in batch: shutil.rmtree(os.path.join(self.temp_extract_dir, f.split('/')[0]), ignore_errors=True)
                    print(f"   -> Extração de Vias: {min(i + batch_size, len(json_files))}/{len(json_files)}", flush=True)

            df_ruas_raw = pl.concat(list_vias_df)
            df_ruas_raw = df_ruas_raw.with_columns(self._limpar_cd_setor("CD_SETOR").alias("CD_SETOR"))

            print("--- Cruzando Ruas e Censo (Join Relacional) ---")
            df_ruas_censo = df_ruas_raw.join(df_censo, on="CD_SETOR", how="left")
            total_faces = df_ruas_censo.height
            faces_com_censo = df_ruas_censo.filter(pl.col("CENSO_POPULACAO").is_not_null()).height

            print("--- Mapeando Ruas para Hexágonos H3 ---")
            df_ruas_h3 = df_ruas_censo.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX")
            )

            # =================================================================
            # 3. SPATIAL JOIN E COALESCE FINAL
            # =================================================================
            print("--- Executando Spatial Join e Fallback ---")
            df_h3_centers = df_ruas_h3.group_by("H3_INDEX").agg([
                pl.col("LAT").first().alias("LAT"), pl.col("LON").first().alias("LON"),
                pl.col("CID_CENSO").first().alias("CID_CENSO"), pl.col("BAI_CENSO").first().alias("BAI_CENSO")
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

            # =================================================================
            # 4. EXPORTAÇÕES (Ouro/Prata)
            # =================================================================
            print("--- Exportando Tabelas para a Camada Prata ---")
            df_hierarquico = df_vias_completo.group_by(["CIDADE", "BAIRRO", "RUA"]).agg(pl.col("H3_INDEX").unique().alias("H3_LIST")) \
                .group_by(["CIDADE", "BAIRRO"]).agg(pl.struct([pl.col("RUA"), pl.col("H3_LIST")]).alias("LOGRADOUROS")) \
                .group_by("CIDADE").agg(pl.struct([pl.col("BAIRRO"), pl.col("LOGRADOUROS")]).alias("BAIRROS"))
            df_hierarquico.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")

            df_social_h3 = df_vias_completo.group_by("H3_INDEX").agg([
                pl.sum("TOT_RES").alias("MICRO_POPULACAO_FACES"),
                pl.mean("CENSO_POPULACAO").fill_null(0).alias("CENSO_MEDIA_V0001"),
                pl.mean("CENSO_RENDA").fill_null(0).alias("CENSO_MEDIA_V0002")
            ])
            df_social_h3.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL_H3.parquet", compression="zstd")

            # =================================================================
            # 5. INFRAESTRUTURA URBANA (CNAEs)
            # =================================================================
            pqs_infra = glob.glob(f"**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            if pqs_infra:
                print("--- Processando Infraestrutura Urbana ---")
                df_infra = pl.scan_parquet(pqs_infra).filter(pl.col("lat").is_not_null()) \
                    .with_columns(pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV")) \
                    .group_by(["lat", "lon", "CNAE_DIV"]).len().collect(engine="streaming")
                
                df_infra_h3 = df_infra.with_columns(pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])).alias("H3_INDEX"))
                df_pivot = df_infra_h3.group_by(["H3_INDEX", "CNAE_DIV"]).agg(pl.sum("len").alias("TOTAL")).pivot(values="TOTAL", index="H3_INDEX", on="CNAE_DIV").fill_null(0)
                df_pivot = df_pivot.rename({c: f"INFRA_DIV_{c}" for c in df_pivot.columns if c != "H3_INDEX"})
                
                df_infra_hierarquica = df_pivot.join(df_h3_mapeado.select(["H3_INDEX", "CIDADE", "BAIRRO"]), on="H3_INDEX", how="left")
                df_infra_hierarquica.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet", compression="zstd")

            # =================================================================
            # 6. TELEMETRIA, UPLOAD R2 E FINALIZAÇÃO
            # =================================================================
            
            # Contagens Exatas para a Auditoria
            qtd_cidades = df_vias_completo.select("CIDADE").unique().height
            qtd_bairros = df_vias_completo.select(["CIDADE", "BAIRRO"]).unique().height
            qtd_ruas = df_vias_completo.select(["CIDADE", "BAIRRO", "RUA"]).unique().height

            tempo_exec = round(time.time() - tempo_inicio, 2)
            sucesso_total = df_h3_mapeado.filter(pl.col("BAIRRO") != "DESCONHECIDO").height
            total_h3 = df_h3_mapeado.height
            
            taxa_sucesso_geral = round((sucesso_total / total_h3) * 100, 2) if total_h3 > 0 else 0
            taxa_join_censo = round((faces_com_censo / total_faces) * 100, 2) if total_faces > 0 else 0
            
            self.auditoria["status_pipeline"] = "SUCESSO"
            self.auditoria["telemetria"] = {
                "tempo_seg": tempo_exec,
                "h3_unicos": total_h3,
                "cidades_processadas": qtd_cidades,
                "bairros_processados": qtd_bairros,
                "ruas_processadas": qtd_ruas,
                "faces_processadas": total_faces,
                "faces_com_dados_censo": faces_com_censo,
                "taxa_enriquecimento_censo_pct": taxa_join_censo,
                "taxa_cobertura_nome_localidade_pct": taxa_sucesso_geral
            }
            
            # Salva auditoria localmente
            with open(f"{self.prata_dir}/AUDITORIA_PRATA_MALHA.json", "w") as f:
                json.dump(self.auditoria, f, indent=4)

            # ✨ Upload de tudo para o R2 com a nova hierarquia
            self.upload_r2()

            msg = (
                f"🗺️ **[SafeDriver] Malha Prata Híbrida OK**\n"
                f"- Tempo de Processamento: {tempo_exec}s\n"
                f"--------------------------------------\n"
                f"📍 **Volumetria Espacial:**\n"
                f"  • Cidades Mapeadas: {qtd_cidades}\n"
                f"  • Bairros Mapeados: {qtd_bairros}\n"
                f"  • Logradouros Únicos: {qtd_ruas}\n"
                f"  • Hexágonos (H3): {total_h3}\n"
                f"--------------------------------------\n"
                f"📊 **Qualidade dos Dados:**\n"
                f"  • Cobertura de Bairros: {taxa_sucesso_geral}%\n"
                f"  • Enriquecimento IBGE: {taxa_join_censo}%\n"
                f"☁️ Arquivos sincronizados em `datalake/prata/malha_trusted/`"
            )
            print(f"\n✨ {msg}")
            self._notificar_discord(msg)

        except Exception as e:
            err_msg = f"❌ **Erro na Pipeline Prata**: {str(e)}"
            print(err_msg)
            self._notificar_discord(err_msg)
            raise e
        finally:
            if os.path.exists(self.temp_extract_dir):
                shutil.rmtree(self.temp_extract_dir, ignore_errors=True)
            self.con.close()

if __name__ == "__main__":
    arquiteto = ArquitetoSafeDriverPrata()
    arquiteto.download_r2()
    arquiteto.processar()
