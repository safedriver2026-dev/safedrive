import os
import boto3
import polars as pl
import io
import requests
import time
import json
from botocore.config import Config
from datetime import datetime

class ArquitetoSafeDriverOuro:
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
            "fase": "ABT Master + Feature Store Materializada",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        try:
            print(f"   -> Procurando: {key}", flush=True)
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            print(f"      [OK] {df.height} linhas carregadas.")
            return df
        except Exception as e:
            print(f"      [AVISO] Falha ao carregar {key}. Detalhe: {e}")
            return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("🚀 [OURO] Iniciando Consolidação da ABT Final + Feature Store...", flush=True)
        
        # =================================================================
        # 1. CARREGAMENTO DOS COMPONENTES DA MALHA (O Palco)
        # =================================================================
        print("📥 Carregando Infraestrutura e Dados Sociais...", flush=True)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")

        if df_infra is None:
            raise FileNotFoundError("Base de Infraestrutura não encontrada. Abortando pipeline Gold.")
        if df_social is None:
            print("⚠️ Ficheiro Social ausente. Gerando esqueleto H3 para não quebrar o Join.")
            df_social = pl.DataFrame({"H3_INDEX": df_infra["H3_INDEX"].unique()})

        # Base Estática (Onde as coisas acontecem)
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # =================================================================
        # 2. CARREGAMENTO DOS CRIMES DA PRATA
        # =================================================================
        print("📥 Consolidando crimes da Prata...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        if not crime_files:
            raise FileNotFoundError("Nenhum arquivo de crime encontrado em datalake/prata/crimes_trusted/")

        lista_crimes = []
        for f in crime_files:
            df_ano = self._ler_parquet_r2(f)
            if df_ano is not None:
                lista_crimes.append(df_ano)
        
        df_crimes = pl.concat(lista_crimes, how="diagonal")
        df_crimes = df_crimes.filter(pl.col("H3_INDEX").is_not_null())

        # =================================================================
        # 3. FEATURE ENGINEERING & DIREITO PENAL (O Advogado)
        # =================================================================
        print("⚖️ Tipificação Penal baseada no Código Penal Brasileiro...", flush=True)
        
        df_crimes = df_crimes.with_columns(pl.col("DATAOCORRENCIA").cast(pl.Date, strict=False))
        df_crimes = df_crimes.with_columns([
            pl.col("RUBRICA").fill_null("").str.to_uppercase().alias("RUBRICA_UPPER"),
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA")
        ])
        
        df_gold = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").dt.weekday().fill_null(0).alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().fill_null(0).alias("FEAT_MES"),
            
            # Classificação do Perfil da Vítima
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"VEICULO|VEÍCULO|CARGA")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"TRANSEUNTE|CELULAR|PESSOA")).then(pl.lit("PEDESTRE"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"RESIDENCIA|RESIDÊNCIA|ESTABELECIMENTO|BANCO|COMERCIO|COMÉRCIO")).then(pl.lit("PATRIMONIO_FIXO"))
            .otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),

            # ⚖️ DOSIMETRIA DO RISCO (Do Art. 121 ao Art. 155)
            # Risco 10: Crimes contra a Vida e Dignidade Sexual
            pl.when(
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*121") | 
                pl.col("RUBRICA_UPPER").str.contains(r"LATROCINIO|LATROCÍNIO") |
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*21[37]") | 
                pl.col("RUBRICA_UPPER").str.contains(r"\bESTUPRO\b")
            ).then(pl.lit(10.0))
            
            # Risco 8: Extorsão e Sequestro (Golpe do PIX / Sequestro Relâmpago)
            .when(
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*15[89]") | 
                pl.col("RUBRICA_UPPER").str.contains(r"EXTORSAO|EXTORSÃO|SEQUESTRO")
            ).then(pl.lit(8.0))
            
            # Risco 7: Tráfico de Drogas (Zonas de criminalidade estruturada)
            .when(
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*33") | 
                pl.col("RUBRICA_UPPER").str.contains(r"TRAFICO|TRÁFICO|ENTORPECENTE")
            ).then(pl.lit(7.0))
            
            # Risco 5: Roubo Comum (Com grave ameaça)
            .when(
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*157") | 
                pl.col("RUBRICA_UPPER").str.contains(r"\bROUBO\b")
            ).then(pl.lit(5.0))
            
            # Risco 4: Lesão Corporal (Violência em vias públicas/bares)
            .when(
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*129") | 
                pl.col("RUBRICA_UPPER").str.contains(r"LESAO|LESÃO")
            ).then(pl.lit(4.0))

            # Risco 2: Furto (Subtração sem violência)
            .when(
                pl.col("RUBRICA_UPPER").str.contains(r"ART(?:IGO)?\s*155") | 
                pl.col("RUBRICA_UPPER").str.contains(r"\bFURTO\b")
            ).then(pl.lit(2.0))
            
            # Risco 1: Fatos atípicos ou acidentes
            .otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ]).drop("RUBRICA_UPPER")

        # =================================================================
        # 3.5 🏪 FEATURE STORE: Histórico (O Cientista de Dados)
        # =================================================================
        print("🏪 Construindo a Feature Store Histórica (Teoria das Janelas Quebradas)...", flush=True)
        
        # Agregação 1: Visão Macro do Hexágono
        df_fs_hex = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").sum().alias("FS_RISCO_TOTAL_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT"),
            (pl.col("LABEL_PESO_RISCO") >= 5).sum().alias("FS_CRIMES_GRAVES_ANO_ANT")
        ]).with_columns(
            (pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN")
        )

        # Agregação 2: Visão Micro (Hexágono + Período)
        df_fs_periodo = df_gold.group_by(["H3_INDEX", "SAZON_PERIODO", "ANO_OCORRENCIA"]).agg([
            pl.col("LABEL_PESO_RISCO").sum().alias("FS_RISCO_PERIODO_ANO_ANT")
        ]).with_columns(
            (pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN")
        )

        # =================================================================
        # 3.6 💾 MATERIALIZANDO A FEATURE STORE NO R2 (Para Produção)
        # =================================================================
        print("💾 Materializando a Feature Store offline no R2...", flush=True)
        fs_dir = "datalake/ouro/feature_store"
        
        buf_fs_hex = io.BytesIO()
        df_fs_hex.drop("ANO_JOIN").write_parquet(buf_fs_hex, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{fs_dir}/fs_macro_hex_historico.parquet", Body=buf_fs_hex.getvalue())

        buf_fs_periodo = io.BytesIO()
        df_fs_periodo.drop("ANO_JOIN").write_parquet(buf_fs_periodo, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{fs_dir}/fs_micro_periodo_historico.parquet", Body=buf_fs_periodo.getvalue())
        print("   [OK] Feature Store gravada com sucesso.")

        # =================================================================
        # 4. O GRANDE JOIN FINAL: EVENTO + INFRA + FEATURE STORE
        # =================================================================
        print("🏙️ Enriquecendo Base de Treino com Espaço e Histórico...", flush=True)
        
        df_fs_hex_join = df_fs_hex.drop("ANO_OCORRENCIA")
        df_fs_periodo_join = df_fs_periodo.drop("ANO_OCORRENCIA")

        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left")
        
        df_final = df_final.join(
            df_fs_hex_join,
            left_on=["H3_INDEX", "ANO_OCORRENCIA"],
            right_on=["H3_INDEX", "ANO_JOIN"],
            how="left"
        )
        
        df_final = df_final.join(
            df_fs_periodo_join,
            left_on=["H3_INDEX", "SAZON_PERIODO", "ANO_OCORRENCIA"],
            right_on=["H3_INDEX", "SAZON_PERIODO", "ANO_JOIN"],
            how="left"
        )

        # Limpeza de Nulos: Se não achou na infra ou no histórico, é 0.
        cols_contexto = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "MICRO_", "FS_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_contexto])
        
        # Remoção do Ano de Ocorrência (Prevenção de Data Leakage)
        df_final = df_final.drop("ANO_OCORRENCIA")

        # =================================================================
        # 5. SALVAMENTO DA ABT NO R2
        # =================================================================
        print("📦 Gravando ABT Master no Data Lake...", flush=True)
        buf_parquet = io.BytesIO()
        df_final.write_parquet(buf_parquet, compression="zstd")
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", 
            Body=buf_parquet.getvalue()
        )

        # =================================================================
        # 6. GERAÇÃO DO DOSSIÊ DE AUDITORIA DE QUALIDADE
        # =================================================================
        duracao = round(time.time() - inicio_timer, 2)
        total_linhas = df_final.height
        mem_mb = round(df_final.estimated_size() / (1024 * 1024), 2)
        fs_cols_count = len([c for c in df_final.columns if c.startswith("FS_")])
        
        # A. Distribuição da Variável Alvo
        df_target = df_final.group_by("LABEL_PESO_RISCO").len().sort("LABEL_PESO_RISCO").to_dicts()
        dist_str = "\n".join([f"   • Risco {str(d['LABEL_PESO_RISCO']).ljust(4)}: {d['len']} registros ({(d['len']/total_linhas)*100:.2f}%)" for d in df_target])

        # B. Saúde da Feature Store
        fs_hits = df_final.filter(pl.col("FS_VOL_CRIMES_ANO_ANT") > 0).height
        fs_hit_rate = (fs_hits / total_linhas) * 100 if total_linhas > 0 else 0
        max_crimes_h3 = df_final["FS_VOL_CRIMES_ANO_ANT"].max() if total_linhas > 0 else 0

        # C. Distribuição de Perfis
        df_perfil = df_final.group_by("FEAT_PERFIL_VITIMA").len().sort("len", descending=True).to_dicts()
        perfil_str = "\n".join([f"   • {d['FEAT_PERFIL_VITIMA'].ljust(15)}: {d['len']} ({(d['len']/total_linhas)*100:.1f}%)" for d in df_perfil])

        # Salvar JSON de Auditoria
        self.auditoria["metricas"] = {
            "linhas_processadas": total_linhas,
            "colunas_totais": len(df_final.columns),
            "feature_store_hit_rate_pct": round(fs_hit_rate, 2),
            "tempo_execucao_segundos": duracao
        }
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/auditoria/AUDITORIA_OURO_FINAL.json", Body=json.dumps(self.auditoria, indent=4).encode())

        # Log Detalhado (Dossiê)
        report = (
            f"==============================================================\n"
            f" 🛡️ DOSSIÊ DE QUALIDADE DE DADOS (CAMADA OURO PRO) 🛡️\n"
            f"==============================================================\n"
            f"📊 1. VOLUMETRIA E ARQUITETURA\n"
            f"   • Total de Registros (ABT) : {total_linhas}\n"
            f"   • Total de Features        : {len(df_final.columns)}\n"
            f"   • Tamanho em Memória       : {mem_mb} MB\n"
            f"   • Tempo de Construção      : {duracao} segundos\n\n"
            f"⚖️ 2. DOSIMETRIA PENAL (VARIÁVEL ALVO)\n"
            f"{dist_str}\n\n"
            f"🏪 3. SAÚDE DA FEATURE STORE (JANELAS QUEBRADAS)\n"
            f"   • Hit Rate (Possui Histórico) : {fs_hit_rate:.1f}%\n"
            f"   • Features Temporais Geradas  : {fs_cols_count}\n"
            f"   • Máx. Crimes/Ano em um Hex   : {max_crimes_h3}\n\n"
            f"👥 4. CONTEXTO DA VÍTIMA\n"
            f"{perfil_str}\n"
            f"==============================================================\n"
            f"Status: ABT MASTER E FEATURE STORE SALVAS COM SUCESSO.\n"
            f"=============================================================="
        )
        
        print(f"\n{report}\n")
        self._notificar_discord(f"```ml\n{report}\n```")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_abt_final()
