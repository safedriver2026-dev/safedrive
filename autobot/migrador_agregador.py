import os
import io
import boto3
import logging
import gc
import polars as pl
from botocore.config import Config

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class MigradorAgregador:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        raw_endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        
        if not self.bucket:
            logger.error("❌ A variável R2_BUCKET_NAME está vazia no GitHub Secrets!")
            raise ValueError("Bucket name não configurado.")

        # 🔥 BLINDAGEM: Remove o nome do bucket do endpoint caso o usuário tenha colocado por engano
        if raw_endpoint.endswith(f"/{self.bucket}"):
            raw_endpoint = raw_endpoint[: -len(f"/{self.bucket}")]
            logger.info("🔧 Endpoint URL corrigido automaticamente pelo sistema.")

        self.s3 = boto3.client(
            's3', 
            endpoint_url=raw_endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        self.prefixo_antigo = "datalake/bronze/malha_raw/comercio/"
        self.prefixo_novo = "datalake/bronze/malha_raw/"
        
        # Lote de 100 arquivos de 4MB (Gera 1 arquivo de ~400MB na RAM)
        self.TAMANHO_LOTE = 100 

    def espremer_e_agregar(self):
        logger.info("🚀 Iniciando varredura para AGREGAR e COMPRIMIR arquivos...")
        
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            paginas = paginator.paginate(Bucket=self.bucket, Prefix=self.prefixo_antigo)
        except Exception as e:
            logger.error(f"❌ Erro fatal ao tentar ler o Bucket. Verifique suas chaves. Erro: {e}")
            return

        lote_atual_dfs = []
        lote_atual_keys = []
        lote_id = 1
        total_arquivos = 0

        for pagina in paginas:
            if 'Contents' not in pagina: 
                continue
                
            for obj in pagina['Contents']:
                old_key = obj['Key']
                
                if not old_key.endswith('.parquet'): 
                    continue
                
                try:
                    arquivo_bytes = self.s3.get_object(Bucket=self.bucket, Key=old_key)['Body'].read()
                    df_temp = pl.read_parquet(io.BytesIO(arquivo_bytes))
                    lote_atual_dfs.append(df_temp)
                    lote_atual_keys.append(old_key)
                    total_arquivos += 1
                except Exception as e:
                    logger.error(f"⚠️ Falha isolada ao ler {old_key}: {e}")
                    continue

                if len(lote_atual_dfs) >= self.TAMANHO_LOTE:
                    self._processar_blocao(lote_atual_dfs, lote_atual_keys, lote_id)
                    lote_atual_dfs.clear()
                    lote_atual_keys.clear()
                    lote_id += 1
                    gc.collect()

        if lote_atual_dfs:
            self._processar_blocao(lote_atual_dfs, lote_atual_keys, lote_id)

        if total_arquivos == 0:
            logger.info("🎉 FILA ZERADA! Não há mais arquivos na pasta antiga.")
        else:
            logger.info(f"✅ SUCESSO! {total_arquivos} arquivos transformados em {lote_id} blocões.")

    def _processar_blocao(self, lista_dfs, lista_keys, lote_id):
        try:
            logger.info(f"📦 Esmagando Lote {lote_id} ({len(lista_dfs)} arquivos)...")
            
            df_blocao = pl.concat(lista_dfs, how="diagonal")
            
            # --- 1. TRATAMENTO NUMÉRICO E DATAS (Evita o erro de tipagem) ---
            colunas_tratadas = []
            
            # Trata Latitude e Longitude
            for col in ["lat", "lon"]:
                if col in df_blocao.columns:
                    if df_blocao.schema[col] in [pl.Utf8, pl.String]:
                        colunas_tratadas.append(
                            pl.col(col).str.strip_chars().str.replace(",", ".").cast(pl.Float32, strict=False)
                        )
                    else:
                        colunas_tratadas.append(pl.col(col).cast(pl.Float32, strict=False))
            
            # Trata a Data
            if "data_inicio_atividade" in df_blocao.columns:
                colunas_tratadas.append(pl.col("data_inicio_atividade").str.to_date("%Y-%m-%d", strict=False))

            if colunas_tratadas:
                df_blocao = df_blocao.with_columns(colunas_tratadas)

            # --- 2. LIMPEZA ---
            # Se não tem coordenada, não serve pro mapa H3
            if "lat" in df_blocao.columns and "lon" in df_blocao.columns:
                df_blocao = df_blocao.drop_nulls(subset=["lat", "lon"])
            
            # --- 3. COMPRESSÃO DE TEXTO RESTANTE ---
            colunas_texto = [col for col, dtype in zip(df_blocao.columns, df_blocao.dtypes) if dtype in (pl.Utf8, pl.String)]
            if colunas_texto:
                df_blocao = df_blocao.with_columns([pl.col(c).cast(pl.Categorical) for c in colunas_texto])

            # --- 4. UPLOAD E LIMPEZA (ZSTD) ---
            out_buffer = io.BytesIO()
            df_blocao.write_parquet(out_buffer, compression="zstd", compression_level=9, use_pyarrow=True)
            
            new_key = f"{self.prefixo_novo}CNPJ_SP_HISTORICO_LOTE_{str(lote_id).zfill(3)}.parquet"
            self.s3.put_object(Bucket=self.bucket, Key=new_key, Body=out_buffer.getvalue())
            
            tamanho_mb = out_buffer.getbuffer().nbytes / 1024 / 1024
            logger.info(f"✨ Lote {lote_id} concluído! Tamanho Otimizado: {tamanho_mb:.2f} MB")
            
            for key in lista_keys:
                self.s3.delete_object(Bucket=self.bucket, Key=key)
            
            del df_blocao

        except Exception as e:
            logger.error(f"❌ Erro crítico ao processar Lote {lote_id}: {e}")

if __name__ == "__main__":
    migrador = MigradorAgregador()
    migrador.espremer_e_agregar()
