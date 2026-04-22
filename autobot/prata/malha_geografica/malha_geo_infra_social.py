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

        # Configuração do Motor Matemático
        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"PRAGMA memory_limit='5GB';")
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except:
            pass 
        
        self.auditoria = {
            "projeto": "SafeDriver - Malha Geográfica (Spatial Join + Social H3)",
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
        """Leitor cego para os dados Sociais do Censo"""
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
            except Exception:
                continue
                
        try:
            sep = csv.Sniffer().sniff(sample_text).delimiter
        except Exception:
            sep = ';' if sample_text.count(';') > sample_text.count(',') else ','
            
        print(f"✅ Identificado -> Encoding: {used_enc} | Separador: '{sep}'", flush=True)
        pl_encoding = 'utf8' if used_enc == 'utf-8' else 'iso-8859-1'
        return pl.read_csv(filepath, separator=sep, encoding=pl_encoding, infer_schema_length=0, ignore_errors=True)

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

    def processar(self):
        tempo_inicio = time.time()
        print("🚀 Iniciando Processamento da Malha com Spatial Join...", flush=True)
        try:
            # =================================================================
            # 1. CARREGAR RUAS E CALCULAR H3 (Incluindo a ponte CD_SETOR)
            # =================================================================
            print("--- Extraindo Vias, Setores e Coordenadas (Faces) ---")
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
                    print(f"   -> Extração de Vias: {min(i + batch_size, len(json_files))}/{len(json_files)}", flush=True)

            df_ruas_raw = pl.concat(list_vias_df)
            total_faces = df_ruas_raw.height
            
            print("--- Mapeando Ruas para Hexágonos H3 ---")
            df_ruas = df_ruas_raw.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX")
            )

            # =================================================================
            # 2. O SPATIAL JOIN (Motor Matemático de Bairros)
            # =================================================================
            print("--- Executando Spatial Join (H3 vs Polígonos de Bairros) ---")
            
            df_h3_centers = df_ruas.group_by("H3_INDEX").agg([
                pl.col("LAT").first().alias("LAT"),
                pl.col("LON").first().alias("LON")
            ])
            self.con.register("tabela_h3", df_h3_centers.to_arrow())
            
            print("--- Localizando Shapefile de Bairros ---")
            arq_poligonos = self._buscar_arquivo_flexivel("SP_bairros_CD2022*.shp").replace("\\", "/")
            
            query_espacial = f"""
                SELECT 
                    h3.H3_INDEX,
                    COALESCE(ibge.NM_MUN, 'DESCONHECIDO') AS CIDADE,
                    COALESCE(ibge.NM_BAIRRO, 'DESCONHECIDO') AS BAIRRO
                FROM tabela_h3 h3
                LEFT JOIN ST_Read('{arq_poligonos}') ibge
                ON ST_Contains(ibge.geom, ST_Point(h3.LON, h3.LAT))
            """
            df_h3_mapeado = self.con.execute(query_espacial).pl()

            df_h3_mapeado = df_h3_mapeado.with_columns([
                pl.col("CIDADE").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("BAIRRO").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])

            df_vias_completo = df_ruas.join(df_h3_mapeado, on="H3_INDEX", how="left").with_columns(
                pl.col("RUA").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            )

            # =================================================================
            # 3. EXPORTAÇÕES DA MALHA (Hierarquia e Micro-população)
            # =================================================================
            print("--- Consolidando e Exportando a Hierarquia Geográfica ---")
            df_hierarquico = (
                df_vias_completo.group_by(["CIDADE", "BAIRRO", "RUA"]).agg(pl.col("H3_INDEX").unique().alias("H3_LIST"))
                .group_by(["CIDADE", "BAIRRO"]).agg(pl.struct([pl.col("RUA"), pl.col("H3_LIST")]).alias("LOGRADOUROS"))
                .group_by("CIDADE").agg(pl.struct([pl.col("BAIRRO"), pl.col("LOGRADOUROS")]).alias("BAIRROS"))
            )
            df_hierarquico.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")

            df_micro = df_vias_completo.group_by("H3_INDEX").agg(pl.sum("TOT_RES").alias("MICRO_POPULACAO_H3"))
            df_micro.write_parquet(f"{self.prata_dir}/PRATA_MALHA_MICRO_POPULACAO.parquet")

            # =================================================================
            # 4. REFERÊNCIA SOCIAL MACRO (Criando o H3 Social para a Camada Ouro)
            # =================================================================
            print("--- Extraindo Variáveis Sociais (Censo) e Agrupando por H3 ---")
            linhas_social_exportadas = 0
            try:
                csv_f = self._buscar_arquivo_flexivel("Agregados_por_setores_basico*.csv")
                df_social_raw = self._ler_csv_agnostico(csv_f)
                
                # Tratamento numérico para População (v0001) e Renda (v0002)
                df_social_limpo = df_social_raw.select([
                    pl.col("CD_SETOR").str.strip_chars(),
                    pl.col("v0001").str.replace(",", ".").cast(pl.Float64, strict=False).alias("CENSO_POPULACAO"),
                    pl.col("v0002").str.replace(",", ".").cast(pl.Float64, strict=False).alias("CENSO_RENDA")
                ])
                
                # O Join com df_vias_completo cria a ponte entre Setor Censitário e H3
                df_social_h3 = df_vias_completo.select(["CD_SETOR", "H3_INDEX", "TOT_RES"]).join(
                    df_social_limpo, on="CD_SETOR", how="left"
                ).group_by("H3_INDEX").agg([
                    pl.sum("TOT_RES").fill_null(0).alias("MICRO_POPULACAO_FACES"),
                    pl.mean("CENSO_POPULACAO").fill_null(0).alias("CENSO_MEDIA_V0001"),
                    pl.mean("CENSO_RENDA").fill_null(0).alias("CENSO_MEDIA_V0002")
                ])

                df_social_h3.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL_H3.parquet", compression="zstd")
                linhas_social_exportadas = df_social_h3.height
                print(f"✅ Arquivo PRATA_MALHA_SOCIAL_H3.parquet gerado com {linhas_social_exportadas} registros.")
            except Exception as e:
                print(f"❌ Erro Crítico no processamento Social: {e}")

            # =================================================================
            # 5. INFRAESTRUTURA URBANA (CNAEs)
            # =================================================================
            print("--- Processando Infraestrutura Urbana (CNAEs) ---")
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
                
                mapa_oficial = df_h3_mapeado.select(["H3_INDEX", "CIDADE", "BAIRRO"]).unique(subset=["H3_INDEX"])
                df_infra_hierarquica = df_pivot.join(mapa_oficial, on="H3_INDEX", how="left").with_columns([
                    pl.col("CIDADE").fill_null("DESCONHECIDO"), pl.col("BAIRRO").fill_null("DESCONHECIDO")
                ])
                
                cols_ordem = ["CIDADE", "BAIRRO", "H3_INDEX"] + [c for c in df_infra_hierarquica.columns if c.startswith("INFRA_")]
                df_infra_hierarquica = df_infra_hierarquica.select(cols_ordem)
                df_infra_hierarquica.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet", compression="zstd")
                df_pivot_height = df_infra_hierarquica.height

            # =================================================================
            # 6. TELEMETRIA E AUDITORIA DE PRECISÃO
            # =================================================================
            tempo_exec = round(time.time() - tempo_inicio, 2)
            sucesso_bairros = df_h3_mapeado.filter(pl.col("BAIRRO") != "DESCONHECIDO").height
            taxa_sucesso = round((sucesso_bairros / df_h3_mapeado.height) * 100, 2) if df_h3_mapeado.height > 0 else 0

            self.auditoria["status_pipeline"] = "SUCESSO"
            self.auditoria["telemetria"] = {
                "processamento": {"tempo_total_segundos": tempo_exec},
                "geometrias_faces": {"poligonos_ruas_extraidos": total_faces},
                "saude_espacial_spatial_join": {
                    "h3_unicos_processados": df_h3_mapeado.height,
                    "h3_com_bairro_matematicamente_confirmado": sucesso_bairros,
                    "taxa_precisao_poligonal_pct": taxa_sucesso
                },
                "infraestrutura_urbana": {
                    "cnpjs_agregados": total_cnpjs,
                    "h3_com_atividade_comercial": df_pivot_height
                },
                "tabelas_exportadas_linhas": {
                    "PRATA_MALHA_GEOGRAFICA_VIAS": df_hierarquico.height,
                    "PRATA_MALHA_SOCIAL_H3": linhas_social_exportadas
                }
            }

            print("✨ Pipeline Prata da Malha concluído com sucesso!")
            
            msg = "🗺️ **[SafeDriver] Malha Prata Atualizada (Ponte Social OK)**\n```ml\n"
            msg += f"• H3 Únicos Mapeados : {df_h3_mapeado.height}\n"
            msg += f"• Precisão do IBGE   : {taxa_sucesso}%\n"
            msg += f"• Hexágonos Sociais  : {linhas_social_exportadas}\n"
            msg += f"• Tempo Execução     : {tempo_exec}s\n```"
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
