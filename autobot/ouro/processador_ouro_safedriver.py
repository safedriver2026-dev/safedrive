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
    Componente responsavel pela consolidacao da Analytical Base Table (ABT).
    Integra a malha de infraestrutura urbana, eventos criminais historicos e 
    calendarios de feriados. Implementa dosimetria penal, achatamento de features 
    e a geracao da variavel de Contexto Critico.
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
            "fase": "ABT Master + Holiday Integration + Feature Store",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        """Dispara notificacao de telemetria operacional."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        """Leitura de arquivos colunares em nuvem utilizando stream de memoria."""
        try:
            print(f"   -> Lendo artefato: {key}", flush=True)
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except: return None

    def _normalizar_texto_pl(self, coluna):
        """
        Rotina de normalizacao padronizada para atributos textuais.
        Remove acentuacoes, espacos invisiveis e impoe tipagem em caixa alta.
        Garante a integridade referencial entre bases governamentais distintas.
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
        """
        Aplica a rotina de normalizacao a todas as colunas identificadas 
        como tipo string no DataFrame de entrada.
        """
        colunas_string = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        if colunas_string:
            df = df.with_columns([self._normalizar_texto_pl(col).alias(col) for col in colunas_string])
        return df

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("[OURO] Iniciando Consolidacao da Tabela Master...", flush=True)
        
        # 1. CARREGAMENTO DA MALHA FÍSICA E SOCIAL
        print("Carregando bases de infraestrutura e dados socioeconomicos...", flush=True)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        
        if df_infra is None: 
            raise FileNotFoundError("Base de Infraestrutura ausente no repositorio de nuvem.")
            
        if df_social is None:
            df_social = pl.DataFrame({"H3_INDEX": df_infra["H3_INDEX"].unique()})

        df_infra = self._aplicar_normalizacao_global(df_infra)
        df_social = self._aplicar_normalizacao_global(df_social)

        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. CARREGAMENTO E CONSOLIDAÇÃO DOS EVENTOS CRIMINAIS
        print("Consolidando base historica de boletins de ocorrencia...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        lista_crimes = [df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None]
        df_crimes = pl.concat(lista_crimes, how="diagonal").filter(pl.col("H3_INDEX").is_not_null())
        
        print("Iniciando normalizacao textual global no conjunto de ocorrencias...", flush=True)
        df_crimes = self._aplicar_normalizacao_global(df_crimes)

        # 3. INTEGRAÇÃO SEGURA DE CALENDARIO (FERIADOS E DIAS UTEIS)
        print("Acoplando matriz temporal de feriados...", flush=True)
        try:
            obj_feriados = self.s3.get_object(Bucket=self.bucket, Key="datalake/prata/referencias/feriados_sp_2022_2026.json")
            json_feriados = json.loads(obj_feriados['Body'].read().decode('utf-8'))
            
            df_feriados = pl.DataFrame(json_feriados)
            
            # Captura o tipo de feriado ou ponto facultativo como uma classificacao pre-processada
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
            print(f"Falha ao acoplar base de feriados: {e}. Contingencia: Atribuindo dias uteis globais.", flush=True)
            df_crimes = df_crimes.with_columns(pl.lit("DIA_UTIL").alias("CLASSIFICACAO_CALENDARIO"))

        # 4. ENGENHARIA DE ATRIBUTOS (ACHATAMENTO E DOSIMETRIA PENAL)
        print("Executando achatamento de variaveis espaco-temporais e dosimetria penal...", flush=True)
        
        # Extracao estruturada da hora
        df_crimes = df_crimes.with_columns([
            pl.col("HORAOCORRENCIA").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int8, strict=False).alias("HORA_INT")
        ])

        df_crimes = df_crimes.with_columns([
            pl.col("RUBRICA").fill_null("").alias("RUBRICA_UPPER"),
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA"),
            pl.col("DATAOCORRENCIA").dt.weekday().fill_null(0).alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().fill_null(0).alias("FEAT_MES"),
            
            # Categorizacao Sociologica do Tempo (Sazonalidade)
            pl.when(pl.col("HORA_INT") >= 18).then(pl.lit("NOITE"))
            .when(pl.col("HORA_INT") >= 12).then(pl.lit("TARDE"))
            .when(pl.col("HORA_INT") >= 6).then(pl.lit("MANHA"))
            .when(pl.col("HORA_INT") >= 0).then(pl.lit("MADRUGADA"))
            .otherwise(pl.lit("INCERTO")).alias("SAZON_PERIODO")
        ])

        # Achatamento (Flattening): Consolida final de semana e feriados em uma unica coluna categorica
        df_crimes = df_crimes.with_columns([
            pl.when(pl.col("CLASSIFICACAO_CALENDARIO") == "FERIADO").then(pl.lit("FERIADO"))
            .when(pl.col("FEAT_DIA_SEMANA").is_in([6, 7])).then(pl.lit("FIM_DE_SEMANA"))
            .when(pl.col("CLASSIFICACAO_CALENDARIO") == "PONTO_FACULTATIVO").then(pl.lit("PONTO_FACULTATIVO"))
            .otherwise(pl.lit("DIA_UTIL")).alias("FEAT_TIPO_DIA")
        ]).drop(["HORA_INT", "CLASSIFICACAO_CALENDARIO"])

        df_gold = df_crimes.with_columns([
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"VEICULO|CARGA")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"TRANSEUNTE|CELULAR|PESSOA")).then(pl.lit("PEDESTRE"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"RESIDENCIA|ESTABELECIMENTO|BANCO|COMERCIO")).then(pl.lit("PATRIMONIO_FIXO"))
            .otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),

            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*121|LATROC")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*15[89]|EXTORSAO|SEQUESTRO")).then(pl.lit(8.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*33|TRAFICO")).then(pl.lit(7.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*157|ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*129|LESAO")).then(pl.lit(4.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*155|FURTO")).then(pl.lit(2.0))
            .otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ])

        # Criacao Direta da Variavel de Contexto Critico
        print("Sintetizando a variavel de Contexto Critico (Sazonalidade + Perfil Alvo)...", flush=True)
        df_gold = df_gold.with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        ])

        # 5. CONSTRUÇÃO DA FEATURE STORE (HISTÓRICO ESPAÇO-TEMPORAL)
        print("Processando variaveis de defasagem temporal (Feature Store)...", flush=True)
        
        df_fs_hex = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").sum().alias("FS_RISCO_TOTAL_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT"),
            (pl.col("LABEL_PESO_RISCO") >= 5).sum().alias("FS_CRIMES_GRAVES_ANO_ANT")
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        df_fs_periodo = df_gold.group_by(["H3_INDEX", "SAZON_PERIODO", "ANO_OCORRENCIA"]).agg([
            pl.col("LABEL_PESO_RISCO").sum().alias("FS_RISCO_PERIODO_ANO_ANT")
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        for name, df_fs in [("macro_hex", df_fs_hex), ("micro_periodo", df_fs_periodo)]:
            buf = io.BytesIO()
            df_fs.drop("ANO_JOIN").write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/feature_store/fs_{name}_historico.parquet", Body=buf.getvalue())

        # 6. ENRIQUECIMENTO FINAL DA BASE ANALÍTICA (JOIN MULTIDIMENSIONAL)
        print("Executando fusao de bases operacionais, infraestrutura e historicas...", flush=True)
        
        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left") \
                          .join(df_fs_hex.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left") \
                          .join(df_fs_periodo.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "SAZON_PERIODO", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "SAZON_PERIODO", "ANO_JOIN"], how="left")
        
        cols_fill = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "FS_", "MICRO_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_fill])

        # 6.1 REDUÇÃO DE DIMENSIONALIDADE: MACRO CLASSES DE CNAE
        print("Agrupando subsetores economicos em Macro Classes de Risco Urbano...", flush=True)
        
        cnae_macros = {
            "MACRO_FINANCEIRO": ["INFRA_DIV_64", "INFRA_DIV_65", "INFRA_DIV_66"],
            "MACRO_LAZER_NOTURNO": ["INFRA_DIV_56", "INFRA_DIV_90", "INFRA_DIV_93"],
            "MACRO_VAREJO": ["INFRA_DIV_45", "INFRA_DIV_47"],
            "MACRO_LOGISTICA_INDUSTRIA": ["INFRA_DIV_49", "INFRA_DIV_52", "INFRA_DIV_53"] + [f"INFRA_DIV_{i}" for i in range(10, 34)],
            "MACRO_SERVICOS_BASE": ["INFRA_DIV_84", "INFRA_DIV_85", "INFRA_DIV_86"]
        }

        for macro_name, div_list in cnae_macros.items():
            cols_existentes = [c for c in div_list if c in df_final.columns]
            if cols_existentes:
                df_final = df_final.with_columns(pl.sum_horizontal(cols_existentes).alias(macro_name))
            else:
                df_final = df_final.with_columns(pl.lit(0).alias(macro_name))

        cols_cnae_brutos = [c for c in df_final.columns if c.startswith("INFRA_DIV_")]
        df_final = df_final.drop(cols_cnae_brutos + ["RUBRICA_UPPER", "ANO_OCORRENCIA"])

        # 7. EXPORTAÇÃO E AUDITORIA DO PIPELINE
        print("Gravando ABT Final no Repositorio em Nuvem...", flush=True)
        buf_abt = io.BytesIO()
        df_final.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf_abt.getvalue())

        duracao = round(time.time() - inicio_timer, 2)
        fs_hit_rate = (df_final.filter(pl.col("FS_VOL_CRIMES_ANO_ANT") > 0).height / df_final.height) * 100
        
        report = (
            f"Relatorio Operacional - Camada Ouro Concluida\n"
            f"=============================================\n"
            f"- Total de Registros (ABT): {df_final.height}\n"
            f"- Quantidade de Atributos : {len(df_final.columns)}\n"
            f"- Taxa Cobertura Histórica: {fs_hit_rate:.1f}%\n"
            f"- Tempo de Execucao (s)   : {duracao}s\n"
            f"============================================="
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_abt_final()
