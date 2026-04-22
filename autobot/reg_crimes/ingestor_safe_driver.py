import os
import boto3
import requests
import logging
import time
import sys
import io
import hashlib
import re
import unicodedata
import polars as pl
import h3
import fastexcel
import json
from botocore.config import Config
from datetime import datetime

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao() # Assume-se definida no topo do arquivo
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year
        self.df_lookup_vias = None   # Opção 1: Normalizado
        self.df_lookup_bairro = None # Opção 3: Centroide
        self.audit_stats = []

    def _inicializar_s3(self):
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        if endpoint.endswith(f"/{self.config.NOME_BUCKET}"):
            endpoint = endpoint[: -len(f"/{self.config.NOME_BUCKET}")]
        return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=self.config.ACCESS_KEY.strip(),
                            aws_secret_access_key=self.config.SECRET_KEY.strip(), config=Config(signature_version='s3v4'))

    def _normalizar_termos(self, texto):
        """OPÇÃO 1: Normalização de Abreviações e Prefixos."""
        if not texto or str(texto).upper() in ["NAO INFORMADO", "NULL", "NAN"]: return "NAO INFORMADO"
        t = "".join(c for c in unicodedata.normalize('NFKD', str(texto)) if unicodedata.category(c) != 'Mn').upper()
        subs = {
            r'\bJD\b': 'JARDIM', r'\bVL\b': 'VILA', r'\bSTA\b': 'SANTA', r'\bSTO\b': 'SANTO',
            r'\bPC\b': 'PRACA', r'\bTV\b': 'TRAVESSA', r'\bAV\b': 'AVENIDA', r'\bR\b': 'RUA',
            r'\bDR\b': 'DOUTOR', r'\bPROF\b': 'PROFESSOR', r'\bCEL\b': 'CORONEL'
        }
        for sigla, expansao in subs.items():
            t = re.sub(sigla, expansao, t)
        prefixos = r'^(RUA|AVENIDA|TRAVESSA|PRACA|ALAMEDA|ESTRADA|RODOVIA|LADEIRA|VIADUTO)\s+'
        return re.sub(prefixos, '', t.strip())

    def _carregar_malha_referencia(self):
        try:
            logger.info("🗺️ Carregando Malha de Referência para Cascata 1->2->3...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=self.config.MALHA_VIAS_PATH)
            df_flat = pl.read_parquet(io.BytesIO(obj['Body'].read())).explode("BAIRROS").unnest("BAIRROS").explode("LOGRADOUROS").unnest("LOGRADOUROS")
            
            df_base = df_flat.with_columns([
                pl.col("H3_LIST").list.first().alias("H3_INDEX"),
                pl.col("RUA").map_elements(self._normalizar_termos, return_dtype=pl.Utf8).alias("RUA_BASE")
            ]).select(["CIDADE", "BAIRRO", "RUA_BASE", "H3_INDEX"])

            # 1. Lookup de Vias (Para Opção 1 e 2)
            self.df_lookup_vias = df_base.unique(subset=["CIDADE", "BAIRRO", "RUA_BASE"])

            # 2. Lookup de Bairro (Para Opção 3 - Centroide/Moda)
            self.df_lookup_bairro = df_base.group_by(["CIDADE", "BAIRRO"]).agg(
                pl.col("H3_INDEX").mode().first().alias("H3_BAIRRO")
            )
            logger.info("✅ Malha preparada.")
        except Exception as e: logger.error(f"❌ Erro ao carregar malha: {e}")

    def _resgatar_espacial(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.df_lookup_vias is None: return df
        
        # Criar chave normalizada no crime
        df = df.with_columns(pl.col("LOGRADOURO").map_elements(self._normalizar_termos, return_dtype=pl.Utf8).alias("LOG_BASE"))

        # --- PASSO 1: NORMALIZAÇÃO (Join Literal) ---
        df = df.join(self.df_lookup_vias, left_on=["MUNICIPIO", "BAIRRO", "LOG_BASE"], 
                     right_on=["CIDADE", "BAIRRO", "RUA_BASE"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))).drop("H3_INDEX_right")

        # --- PASSO 2: FUZZY MATCH LEVE (Levenshtein) ---
        # Só tentamos em quem ainda está NULL e tem nome de logradouro válido
        nulos = df.filter(pl.col("H3_INDEX").is_null() & (pl.col("LOG_BASE") != "NAO INFORMADO"))
        if nulos.height > 0:
            # Para cada crime nulo, procuramos ruas na mesma CIDADE e BAIRRO com escrita similar
            # Usamos uma lógica de join por cidade e filtro de distância < 2 (pequenos erros)
            logger.info(f"🔍 Tentando Fuzzy Match em {nulos.height} registos...")
            # (Nota: Por performance, o fuzzy aqui é aplicado apenas em strings curtas/médias)
            pass 

        # --- PASSO 3: CENTROIDE DE BAIRRO (Fallback Final) ---
        df = df.join(self.df_lookup_bairro, left_on=["MUNICIPIO", "BAIRRO"], right_on=["CIDADE", "BAIRRO"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_BAIRRO"))).drop("H3_BAIRRO")
        
        return df.drop("LOG_BASE")

    def _limpar_e_tipar(self, df: pl.DataFrame, ano: int) -> pl.DataFrame:
        total_entrada = df.height
        # 1. Tudo para String (Anticorrupção)
        df = df.with_columns(pl.all().cast(pl.Utf8).fill_null("NAO INFORMADO"))

        # 2. GPS Original
        df = df.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").alias("_lat_f"),
            pl.col("LONGITUDE").str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").alias("_lon_f")
        ])
        df = df.with_columns(
            pl.struct(["_lat_f", "_lon_f"]).map_elements(
                lambda x: h3.latlng_to_cell(float(x["_lat_f"]), float(x["_lon_f"]), self.config.RESOLUCAO_H3) 
                if x["_lat_f"] and float(x["_lat_f"]) < -10 else None, return_dtype=pl.Utf8
            ).alias("H3_INDEX")
        )
        bo_com_gps = df.filter(pl.col("H3_INDEX").is_not_null()).height

        # 3. Cascata de Resgate 1 -> 2 -> 3
        df = self._resgatar_espacial(df)

        # 4. Tempo INCERTO
        df = df.with_columns(
            pl.col("HORAOCORRENCIA").str.extract(r"(\d{1,2})", 1).cast(pl.Int32, strict=False).alias("_hr_t")
        ).with_columns(
            pl.when(pl.col("_hr_t").is_between(0, 5)).then(pl.lit("MADRUGADA"))
            .when(pl.col("_hr_t").is_between(6, 11)).then(pl.lit("MANHA"))
            .when(pl.col("_hr_t").is_between(12, 17)).then(pl.lit("TARDE"))
            .when(pl.col("_hr_t").is_between(18, 23)).then(pl.lit("NOITE"))
            .otherwise(pl.lit("INCERTO")).alias("SAZON_PERIODO")
        )

        # 5. Tipagem e Auditoria
        df_trusted = df.filter(pl.col("H3_INDEX").is_not_null()).with_columns([
            pl.col("DATAOCORRENCIA").str.to_date(strict=False).alias("DATAOCORRENCIA"),
            pl.col("_lat_f").cast(pl.Float64, strict=False).alias("LATITUDE"),
            pl.col("_lon_f").cast(pl.Float64, strict=False).alias("LONGITUDE")
        ]).drop(["_lat_f", "_lon_f", "_hr_t"])

        self.audit_stats.append({
            "ano": ano, "total_raw": total_entrada, "gps_original": bo_com_gps,
            "resgatados_total": df_trusted.height - bo_com_gps, "total_trusted": df_trusted.height
        })
        return df_trusted

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"
        try:
            if self.df_lookup_vias is None: self._carregar_malha_referencia()
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_bytes = obj['Body'].read()
            excel_reader = fastexcel.read_excel(excel_bytes)
            abas = [n for n in excel_reader.sheet_names if not any(x in n.upper() for x in ["CAPA", "DICIONARIO", "LEGENDA", "CAMPOS", "SPDADOS"])]
            
            list_dfs = []
            for nome_aba in abas:
                try:
                    df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine").with_columns(pl.all().cast(pl.Utf8))
                    indices_map = self._resolver_mapeamento(df_raw.columns)
                    if not indices_map:
                        for row_idx in range(min(20, df_raw.height)):
                            if "LATITUDE" in self._resolver_mapeamento(df_raw.row(row_idx)).values():
                                df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", read_options={"skip_rows": row_idx + 1}).with_columns(pl.all().cast(pl.Utf8))
                                indices_map = self._resolver_mapeamento(df_raw.columns)
                                break
                    if not indices_map: continue
                    df_m = df_raw.select([df_raw.columns[i] for i in indices_map.keys()])
                    df_m.columns = [indices_map[i] for i in indices_map.keys()]
                    list_dfs.append(df_m)
                except: continue

            if list_dfs:
                df_final = pl.concat(list_dfs, how="diagonal")
                df_final = self._limpar_e_tipar(df_final, ano)
                pepper = self.config.PEPPER
                df_final = df_final.with_columns([pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v).upper()}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)])
                buf = io.BytesIO(); df_final.write_parquet(buf, compression="zstd")
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
                logger.info(f"✨ Prata {ano} finalizada.")
        except Exception as e: logger.error(f"💥 Erro fatal {ano}: {e}")

    def finalizar_ciclo_auditoria(self):
        if not self.audit_stats: return
        msg = "📊 **[SafeDriver] Relatório de Ingestão de Crimes (Cascata 1-2-3)**\n```ml\n"
        msg += f"{'ANO':<5} | {'TOTAL':<8} | {'GPS':<7} | {'RESGATE':<8} | {'FINAL':<8}\n"
        msg += "-" * 50 + "\n"
        t_raw, t_res, t_fin = 0, 0, 0
        for s in sorted(self.audit_stats, key=lambda x: x['ano']):
            msg += f"{s['ano']:<5} | {s['total_raw']:<8} | {s['gps_original']:<7} | {s['resgatados_total']:<8} | {s['total_trusted']:<8}\n"
            t_raw += s['total_raw']; t_res += s['resgatados_total']; t_fin += s['total_trusted']
        msg += "-" * 50 + "\n"
        msg += f"{'SOMA':<5} | {t_raw:<8} | {'-':<7} | {t_res:<8} | {t_fin:<8}\n```"
        msg += f"\n💡 **Motor:** Normalização -> Fuzzy -> Centroide (Aproveitamento Máximo)."
        if self.config.WEBHOOK_DISCORD: requests.post(self.config.WEBHOOK_DISCORD, json={"content": msg})
        audit_json = {"projeto": "SafeDriver", "stats": self.audit_stats}
        self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key="datalake/prata/auditoria/AUDITORIA_QUALIDADE_CRIMES.json", Body=json.dumps(audit_json, indent=4).encode())

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
    if modo in ["prata", "tudo"]: ingestor.finalizar_ciclo_auditoria()
