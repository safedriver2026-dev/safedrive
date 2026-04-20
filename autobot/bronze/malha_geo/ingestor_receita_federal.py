import os
import polars as pl
import boto3
import hashlib
import time

class IngestorReceitaBronze:
    def __init__(self):
        self.salt = os.getenv('LGPD_SALT', 'default_salt')
        self.pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        
        self.s3_client = boto3.client('s3',
            region_name='auto',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME', '').strip()
        self.r2_path = "datalake/bronze/malha_raw/CNPJ_SP_HISTORICO.parquet"

    def gerar_hash_lgpd(self, cnpj_base, cnpj_ordem, cnpj_dv):
        cnpj_completo = f"{cnpj_base}{cnpj_ordem}{cnpj_dv}"
        base_string = f"{cnpj_completo}{self.salt}{self.pepper}"
        return hashlib.sha256(base_string.encode()).hexdigest()

    def processar_dados_rfb(self, input_path):
        # Layout oficial da Receita Federal (Estabelecimentos)
        colunas = [
            "cnpj_base", "cnpj_ordem", "cnpj_dv", "matriz_filial", "nome_fantasia",
            "situacao_cadastral", "data_situacao", "motivo_situacao", "cidade_exterior",
            "pais", "data_inicio_atividade", "cnae_principal", "cnae_secundario",
            "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
            "cep", "uf", "municipio"
        ]

        print(f"🚀 Processando arquivo da Receita: {input_path}")
        
        # Usamos Lazy para performance extrema
        query = (
            pl.scan_csv(input_path, separator=";", has_header=False, 
                        new_columns=colunas, infer_schema_length=0, 
                        encoding="iso-8859-1", ignore_errors=True)
            .filter(pl.col("uf") == "SP") # Filtro geográfico na origem
            .select([
                "cnpj_base", "cnpj_ordem", "cnpj_dv", "situacao_cadastral", 
                "data_situacao", "data_inicio_atividade", "cnae_principal", "cep"
            ])
        )

        df = query.collect()

        # Aplicar Proteção LGPD e Limpeza
        print("🛡️ Aplicando LGPD e formatando datas...")
        df_protegido = df.with_columns([
            pl.struct(["cnpj_base", "cnpj_ordem", "cnpj_dv"]).map_elements(
                lambda x: self.gerar_hash_lgpd(x["cnpj_base"], x["cnpj_ordem"], x["cnpj_dv"]),
                return_dtype=pl.Utf8
            ).alias("ID_PROTEGIDO"),
            pl.col("data_inicio_atividade").str.strptime(pl.Date, format="%Y%m%d", strict=False),
            pl.col("data_situacao").str.strptime(pl.Date, format="%Y%m%d", strict=False)
        ]).drop(["cnpj_base", "cnpj_ordem", "cnpj_dv"])

        # Salva em Parquet (muito mais leve que CSV)
        temp_file = "temp_historico.parquet"
        df_protegido.write_parquet(temp_file)
        
        # Sobe para o R2
        self.s3_client.upload_file(temp_file, self.bucket_name, self.r2_path)
        print(f"✅ Camada Bronze Histórica salva no R2: {self.r2_path}")
        os.remove(temp_file)

if __name__ == "__main__":
    # O arquivo .csv é extraído do .zip da Receita pelo YAML
    ingestor = IngestorReceitaBronze()
    # Processamos a partição 0 (exemplo)
    if os.path.exists("K330.L312.V.E026.ESTABELE0.csv"):
        ingestor.processar_dados_rfb("K330.L312.V.E026.ESTABELE0.csv")
