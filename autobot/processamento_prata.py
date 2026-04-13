import polars as pl
import boto3
import json
import os
import io
import h3

class ProcessamentoPrata:
    def __init__(self):
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")
        
       
        self.colunas_canonicas = [
            'NUM_BO', 'DATA_OCORRENCIA', 'LATITUDE', 'LONGITUDE', 
            'NATUREZA_APURADA', 'DESCRICAO_LOCAL'
        ]

    def _converter_para_h3(self, lat, lon):
        try:
            return h3.latlng_to_cell(float(lat), float(lon), 8)
        except:
            return None

    def executar_prata(self, ano_ref):
        print(f" Iniciando Refinaria Prata para {ano_ref}...")
        resp_dic = self.s3.get_object(Bucket=self.bucket, Key="datalake/bronze/mapa_capas_ssp.json")
        dicionario_mestre = json.loads(resp_dic['Body'].read().decode('utf-8'))
        mapa_ano = dicionario_mestre.get(str(ano_ref), {})

        de_para_colunas = self._gerar_mapeamento_canonico(mapa_ano)

        resp_raw = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/bronze/ssp_raw_{ano_ref}.parquet")
        df = pl.read_parquet(io.BytesIO(resp_raw['Body'].read()))

        print(" Aplicando tradução de colunas...")
        df = df.rename(de_para_colunas)
        colunas_presentes = [c for c in self.colunas_canonicas if c in df.columns]
        df = df.select(colunas_presentes)

        print(" Limpando dados e filtrando Estado de SP...")
        df = df.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
            pl.col("NATUREZA_APURADA").str.to_uppercase()
        ])

        df = df.filter(
            pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null() &
            pl.col("LATITUDE").is_between(-25.31, -19.77) &
            pl.col("LONGITUDE").is_between(-53.11, -44.16)
        )

        print(" Classificando Personas e gerando Malha H3...")
        
        # Classificação da Persona baseada na Natureza do Crime
        df = df.with_columns(
            pl.when(pl.col("NATUREZA_APURADA").str.contains("VEICULO|CARGA"))
            .then(pl.lit("MOTORISTA"))
            .when(pl.col("NATUREZA_APURADA").str.contains("MOTOCICLETA"))
            .then(pl.lit("MOTOCICLISTA"))
            .when(pl.col("NATUREZA_APURADA").str.contains("CELULAR|TRANSEUNTE"))
            .then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("GERAL"))
            .alias("PERFIL_PERSONA")
        )
        df = df.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"])
            .map_elements(lambda x: self._converter_para_h3(x["LATITUDE"], x["LONGITUDE"]), return_dtype=pl.Utf8)
            .alias("H3_INDEX")
        )
        print("☁️ Salvando tabela Consolidada na Prata...")
        buffer_parquet = io.BytesIO()
        df.write_parquet(buffer_parquet)
        
        caminho_prata = f"datalake/prata/ssp_consolidada_{ano_ref}.parquet"
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=caminho_prata, 
            Body=buffer_parquet.getvalue()
        )
        
        print(f"✅ Prata {ano_ref} concluída! {df.height} registros válidos e mapeados.")
        return caminho_prata

    def _gerar_mapeamento_canonico(self, mapa_ano):
        de_para = {}
        for coluna_ssp, descricao in mapa_ano.items():
            desc_upper = str(descricao).upper()
            if "LATITUDE" in desc_upper: de_para[coluna_ssp] = "LATITUDE"
            elif "LONGITUDE" in desc_upper: de_para[coluna_ssp] = "LONGITUDE"
            elif "NÚMERO DO BOLETIM" in desc_upper: de_para[coluna_ssp] = "NUM_BO"
            elif "DATA DA OCORRÊNCIA" in desc_upper: de_para[coluna_ssp] = "DATA_OCORRENCIA"
            elif "NATUREZA" in desc_upper: de_para[coluna_ssp] = "NATUREZA_APURADA"
            # ... mapear as demais essenciais
        return de_para
