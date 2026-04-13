import pandas as pd
import json
import boto3
import os
import io

class IngestaoBronze:
    def __init__(self):
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")

    def processar_arquivo_ssp(self, arquivo_bytes, ano_ref):
        print(f" Iniciando Ingestão Bronze para o ano {ano_ref}...")
        arquivo_io = io.BytesIO(arquivo_bytes)

        print(" Lendo metadados (Capa)...")
  
        df_temp = pd.read_excel(arquivo_io, sheet_name=0, nrows=20, header=None)
        
        linha_cabecalho = 0
        for index, row in df_temp.iterrows():
            linha_texto = " ".join(str(val).lower() for val in row.values)
            if 'campo' in linha_texto and 'descri' in linha_texto:
                linha_cabecalho = index
                break
                
        df_capa = pd.read_excel(arquivo_io, sheet_name=0, header=linha_cabecalho)
        col_campo = [col for col in df_capa.columns if 'campo' in col.lower()][0]
        col_desc = [col for col in df_capa.columns if 'descri' in col.lower()][0]
        
        df_capa = df_capa.dropna(subset=[col_campo, col_desc])
        mapa_ano = dict(zip(df_capa[col_campo].astype(str).str.strip(), df_capa[col_desc].astype(str).str.strip()))
        

        self._salvar_dicionario(ano_ref, mapa_ano)


        print("📥 Extraindo base de dados principal...")
        
        df_dados = pd.read_excel(
            arquivo_io, 
            sheet_name='Dados', 
            dtype=str 
        )

        df_dados = df_dados.fillna("").apply(lambda x: x.str.strip())

        print("☁️ Salvando RAW Data (Parquet) no Cloudflare R2...")
        buffer_parquet = io.BytesIO()
        
      
        df_dados.to_parquet(buffer_parquet, index=False)
        
        caminho_raw = f"datalake/bronze/ssp_raw_{ano_ref}.parquet"
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=caminho_raw, 
            Body=buffer_parquet.getvalue()
        )
        
        print(f" Ingestão Bronze {ano_ref} concluída! {len(df_dados)} registros salvos.")
        return caminho_raw

    def _salvar_dicionario(self, ano, mapa_novo):
        pass
