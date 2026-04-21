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
        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.s3 = boto3.client(
            's3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY").strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        self.prefixo_antigo = "datalake/bronze/malha_raw/comercio/"
        self.prefixo_novo = "datalake/bronze/malha_raw/"
        
        # Junta 100 arquivos de 4MB = 400MB na RAM, resultando em 1 arquivo Parquet otimizado
        self.TAMANHO_LOTE = 100 

    def espremer_e_agregar(self):
        logger.info("🚀 Iniciando varredura para AGREGAR e COMPRIMIR arquivos...")
        
        paginator = self.s3.get_paginator('list_objects_v2')
        paginas = paginator.paginate(Bucket=self.bucket, Prefix=self.prefixo_antigo)

        lote_atual_dfs = []       # Vai guardar os dados
        lote_atual_keys = []      # Vai guardar os nomes para apagar depois
        lote_id = 1
        total_arquivos = 0

        for pagina in paginas:
            if 'Contents' not in pagina: continue
                
            for obj in pagina['Contents']:
                old_key = obj['Key']
                if not old_key.endswith('.parquet'): continue
                
                # 1. Puxa do R2 e lê com Polars
                try:
                    arquivo_bytes = self.s3.get_object(Bucket=self.bucket, Key=old_key)['Body'].read()
                    df_temp = pl.read_parquet(io.BytesIO(arquivo_bytes))
                    lote_atual_dfs.append(df_temp)
                    lote_atual_keys.append(old_key)
                    total_arquivos += 1
                except Exception as e:
                    logger.error(f"❌ Erro ao ler {old_key}: {e}")
                    continue

                # 2. Se o lote encheu, hora de juntar e salvar!
                if len(lote_atual_dfs) >= self.TAMANHO_LOTE:
                    self._processar_blocao(lote_atual_dfs, lote_atual_keys, lote_id)
                    
                    # Limpa a RAM para o próximo lote
                    lote_atual_dfs.clear()
                    lote_atual_keys.clear()
                    lote_id += 1
                    gc.collect()

        # 3. Processa a "sobra" (os últimos arquivos que não completaram um lote de 100)
        if lote_atual_dfs:
            self._processar_blocao(lote_atual_dfs, lote_atual_keys, lote_id)

        if total_arquivos == 0:
            logger.info("🎉 FILA ZERADA! Não há mais arquivos na pasta antiga.")
        else:
            logger.info(f"✅ SUCESSO! {total_arquivos} arquivinhos transformados em {lote_id} blocões super comprimidos.")

    def _processar_blocao(self, lista_dfs, lista_keys, lote_id):
        try:
            logger.info(f"📦 Esmagando e juntando o Lote {lote_id} ({len(lista_dfs)} arquivos)...")
            
            # Concatena os 100 DataFrames em um só gigante
            df_blocao = pl.concat(lista_dfs, how="diagonal")
            
            # --- COMPRESSÃO MÁXIMA ---
            colunas_texto = [col for col, dtype in zip(df_blocao.columns, df_blocao.dtypes) if dtype in (pl.Utf8, pl.String)]
            if colunas_texto:
                df_blocao = df_blocao.with_columns([pl.col(c).cast(pl.Categorical) for c in colunas_texto])
            
            if "lat" in df_blocao.columns and "lon" in df_blocao.columns:
                df_blocao = df_blocao.with_columns([
                    pl.col("lat").cast(pl.Float32, strict=False),
                    pl.col("lon").cast(pl.Float32, strict=False)
                ])

            # Salva o arquivo GIGANTE em buffer usando ZSTD
            out_buffer = io.BytesIO()
            df_blocao.write_parquet(out_buffer, compression="zstd", compression_level=9, use_pyarrow=True)
            
            # --- UPLOAD DO NOVO ---
            # Nomeia como CNPJ_SP_HISTORICO_LOTE_001.parquet
            new_key = f"{self.prefixo_novo}CNPJ_SP_HISTORICO_LOTE_{str(lote_id).zfill(3)}.parquet"
            self.s3.put_object(Bucket=self.bucket, Key=new_key, Body=out_buffer.getvalue())
            
            tamanho_mb = out_buffer.getbuffer().nbytes / 1024 / 1024
            logger.info(f"✨ Lote {lote_id} salvo com SUCESSO! Tamanho Final: {tamanho_mb:.2f} MB")
            
            # --- LIMPEZA DOS ANTIGOS ---
            # Deleta os 100 arquivos arquivinhos antigos do R2
            for key in lista_keys:
                self.s3.delete_object(Bucket=self.bucket, Key=key)
            
            del df_blocao

        except Exception as e:
            logger.error(f"❌ Erro crítico ao processar Lote {lote_id}: {e}")

if __name__ == "__main__":
    migrador = MigradorAgregador()
    migrador.espremer_e_agregar()
