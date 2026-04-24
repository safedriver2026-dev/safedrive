import os
import boto3
import polars as pl
import io
import time
import requests
import json
from botocore.config import Config
from datetime import datetime

class ArquitetoSafeDriverOuro:
    """
    Componente central da Camada Ouro (Refined). Responsavel pela forja da 
    Analytical Base Table (ABT). Realiza a fusao de dados censitarios, 
    infraestrutura urbana e registros criminais, aplicando tecnicas de 
    Feature Engineering e Dosimetria Penal para preparacao de modelos de IA.
    """
    def __init__(self):
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
        self.prata_crimes = "datalake/prata/crimes_trusted"
        self.prata_malha = "datalake/prata/malha_trusted"
        self.ouro_dir = "datalake/ouro"
        
        self.auditoria = {
            "projeto": "SafeDriver - Camada Ouro",
            "fase": "Consolidacao de Analytical Base Table (ABT)",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        """Dispara telemetria operacional via Webhook."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        """Leitura de arquivos colunares Parquet otimizada para nuvem."""
        try:
            print(f"   -> Lendo artefato: {key}", flush=True)
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except: return None

    def _normalizar_texto_pl(self, coluna):
        """
        Padronizacao morfologica de atributos textuais via Polars.
        Remove ruidos diacriticos, carateres especiais e impoe Caixa Alta.
        Garante a integridade dos Joins geoespaciais e filtragem em BI.
        """
        return (
            pl.col(coluna)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[^A-Z0-9\s_]", "")
        )

    def _aplicar_normalizacao_global(self, df):
        """Aplica rotina de normalizacao a todas as colunas do tipo String."""
        colunas_string = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        if colunas_string:
            df = df.with_columns([self._normalizar_texto_pl(col).alias(col) for col in colunas_string])
        return df

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("Iniciando a reconstrucao da Camada Ouro (ABT Master)...", flush=True)
        
        # 1. CARREGAMENTO E FUSAO DA MALHA ESTATICA
        print("Consolidando bases de infraestrutura e demografia IBGE...", flush=True)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        
        if df_infra is None: 
            raise FileNotFoundError("Base de Infraestrutura ausente no repositorio S3.")
            
        if df_social is None:
            df_social = pl.DataFrame({"H3_INDEX": df_infra["H3_INDEX"].unique()})

        df_infra = self._aplicar_normalizacao_global(df_infra)
        df_social = self._aplicar_normalizacao_global(df_social)
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. CARREGAMENTO E NORMALIZACAO DE EVENTOS CRIMINAIS
        print("Concatenando historico de Boletins de Ocorrencia da SSP-SP...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        lista_crimes = [df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None]
        df_crimes = pl.concat(lista_crimes, how="diagonal").filter(pl.col("H3_INDEX").is_not_null())
        df_crimes = self._aplicar_normalizacao_global(df_crimes)

        # 3. ACOPLAMENTO DE CALENDARIO (FERIADOS)
        print("Injetando matriz de feriados e pontos facultativos...", flush=True)
        try:
            obj_feriados = self.s3.get_object(Bucket=self.bucket, Key="datalake/prata/referencias/feriados_sp_2022_2026.json")
            json_feriados = json.loads(obj_feriados['Body'].read().decode('utf-8'))
            df_feriados = pl.DataFrame(json_feriados)
            
            df_feriados = df_feriados.select([
                pl.col("data").str.to_date("%Y-%m-%d", strict=False).alias("DATA_CHAVE"),
                pl.when(pl.col("feriado_tipo").is_not_null()).then(pl.lit("FERIADO"))
                  .when(pl.col("is_ponto_facultativo") == True).then(pl.lit("PONTO_FACULTATIVO"))
                  .otherwise(pl.lit("DIA_UTIL")).alias("CLASSIFICACAO_CALENDARIO")
            ])
            
            df_crimes = df_crimes.with_columns(pl.col("DATAOCORRENCIA").cast(pl.Date, strict=False))
            df_crimes = df_crimes.join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_CHAVE", how="left")
            df_crimes = df_crimes.with_columns(pl.col("CLASSIFICACAO_CALENDARIO").fill_null("DIA_UTIL"))
        except Exception as e:
            print(f"Aviso: Falha na integracao de feriados ({e}).")
            df_crimes = df_crimes.with_columns(pl.lit("DIA_UTIL").alias("CLASSIFICACAO_CALENDARIO"))

        # 4. ENGENHARIA DE ATRIBUTOS TEMPORAIS (SAZONALIDADE)
        print("Executando extracao robusta de horários e turnos...", flush=True)
        
        # Logica de seguranca para tratar formatos: "9:30", "09:30", "14:00:00"
        df_crimes = df_crimes.with_columns([
            pl.col("HORAOCORRENCIA")
            .cast(pl.Utf8)
            .str.split(":")
            .list.first()
            .str.strip_chars()
            .cast(pl.Int8, strict=False)
            .alias("HORA_INT")
        ])

        df_crimes = df_crimes.with_columns([
            pl.col("RUBRICA").fill_null("").alias("RUBRICA_UPPER"),
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA"),
            pl.col("DATAOCORRENCIA").dt.weekday().fill_null(0).alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().fill_null(0).alias("FEAT_MES"),
            
            pl.when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
            .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
            .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
            .when((pl.col("HORA_INT") >= 0) & (pl.col("HORA_INT") < 6)).then(pl.lit("MADRUGADA"))
            .otherwise(pl.lit("INCERTO")).alias("SAZON_PERIODO")
        ])

        # Achatamento (Flattening): Unificacao de caracteristicas de calendario
        df_crimes = df_crimes.with_columns([
            pl.when(pl.col("CLASSIFICACAO_CALENDARIO") == "FERIADO").then(pl.lit("FERIADO"))
            .when(pl.col("FEAT_DIA_SEMANA").is_in([6, 7])).then(pl.lit("FIM_DE_SEMANA"))
            .when(pl.col("CLASSIFICACAO_CALENDARIO") == "PONTO_FACULTATIVO").then(pl.lit("PONTO_FACULTATIVO"))
            .otherwise(pl.lit("DIA_UTIL")).alias("FEAT_TIPO_DIA")
        ]).drop(["HORA_INT", "CLASSIFICACAO_CALENDARIO"])

        # 5. DOSIMETRIA PENAL E CONTEXTO CRITICO
        df_gold = df_crimes.with_columns([
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"VEICULO|CARGA")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"TRANSEUNTE|CELULAR|PESSOA")).then(pl.lit("PEDESTRE"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"RESIDENCIA|ESTABELECIMENTO|BANCO|COMERCIO")).then(pl.lit("PATRIMONIO_FIXO"))
            .otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),

            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*121|LATROC")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*157|ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*155|FURTO")).then(pl.lit(2.0))
            .otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ])

        df_gold = df_gold.with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        ])

        # 6. FEATURE STORE (HISTORICO) E JOIN MULTIDIMENSIONAL
        print("Construindo variaveis historicas e fundindo bases...", flush=True)
        df_fs_hex = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT")
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left") \
                          .join(df_fs_hex.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left")
        
        # Limpeza de nulos e Reducao de Dimensionalidade CNAE
        cols_fill = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "FS_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_fill])

        cnae_macros = {
            "MACRO_FINANCEIRO": ["INFRA_DIV_64", "INFRA_DIV_65", "INFRA_DIV_66"],
            "MACRO_LAZER_NOTURNO": ["INFRA_DIV_56", "INFRA_DIV_90", "INFRA_DIV_93"],
            "MACRO_VAREJO": ["INFRA_DIV_45", "INFRA_DIV_47"]
        }
        for macro_name, div_list in cnae_macros.items():
            existentes = [c for c in div_list if c in df_final.columns]
            df_final = df_final.with_columns(pl.sum_horizontal(existentes).alias(macro_name)) if existentes else df_final.with_columns(pl.lit(0).alias(macro_name))

        # Expurgo de colunas temporarias
        cols_cnae_brutos = [c for c in df_final.columns if c.startswith("INFRA_DIV_")]
        df_final = df_final.drop(cols_cnae_brutos + ["RUBRICA_UPPER", "ANO_OCORRENCIA"])

        # 7. EXPORTACAO E AUDITORIA DETALHADA
        print("Sincronizando ABT Master com o Repositorio de Dados...", flush=True)
        buf_abt = io.BytesIO()
        df_final.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf_abt.getvalue())

        duracao = round(time.time() - inicio_timer, 2)
        dist_sazon = df_final.group_by("SAZON_PERIODO").len().to_dict(as_series=False)
        sazon_dict = {row[0]: row[1] for row in zip(dist_sazon['SAZON_PERIODO'], dist_sazon['len'])}

        report = (
            f"Relatorio de Consolidacao - Camada Ouro Finalizada\n"
            f"================================================\n"
            f"- Registros Processados    : {df_final.height}\n"
            f"- Features Estruturadas    : {len(df_final.columns)}\n"
            f"- Tempo de Execucao (s)    : {duracao}s\n"
            f"------------------------------------------------\n"
            f"AUDITORIA DE PERIODOS (SAZONALIDADE):\n"
            f"  > MANHA      : {sazon_dict.get('MANHA', 0)}\n"
            f"  > TARDE      : {sazon_dict.get('TARDE', 0)}\n"
            f"  > NOITE      : {sazon_dict.get('NOITE', 0)}\n"
            f"  > MADRUGADA  : {sazon_dict.get('MADRUGADA', 0)}\n"
            f"  > INCERTO    : {sazon_dict.get('INCERTO', 0)}\n"
            f"================================================"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    ArquitetoSafeDriverOuro().construir_abt_final()
