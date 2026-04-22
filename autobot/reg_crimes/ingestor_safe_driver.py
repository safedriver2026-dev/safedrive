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

class ConfiguracaoIngestao:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME")
    ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
    ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
    SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    PEPPER = os.getenv("LGPD_PEPPER", "safedriver_secret_2026")
    WEBHOOK_DISCORD = os.getenv("DISCORD_SUCESSO")

    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    MALHA_VIAS_PATH = "datalake/prata/malha_trusted/PRATA_MALHA_GEOGRAFICA_VIAS.parquet"
    RESOLUCAO_H3 = 9
    
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN", r"NOME_MUN"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"NM_LOG", r"ENDERECO"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR", r"DATA_OCORRENCIA"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR", r"HORA_OCORRENCIA"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$", r"COORDENADA_X", r"LATITUD"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"COORDENADA_Y", r"LONGITUD"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL", r"LOCAL"],
        "PERIODO_NATIVO": [r"DESC.*PERIODO", r"PERIODO", r"DS_PERIODO", r"DESC_PERIODO_OCORRENCIA"]
    }

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year
        self.df_lookup_vias = None
        # Acumulador para o relatório final
        self.audit_stats = []

    def _notificar_discord(self, msg):
        if self.config.WEBHOOK_DISCORD:
            try: requests.post(self.config.WEBHOOK_DISCORD, json={"content": msg}, timeout=10)
            except: pass

    def _inicializar_s3(self):
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        if endpoint.endswith(f"/{self.config.NOME_BUCKET}"):
            endpoint = endpoint[: -len(f"/{self.config.NOME_BUCKET}")]
        return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=self.config.ACCESS_KEY.strip(),
                            aws_secret_access_key=self.config.SECRET_KEY.strip(), config=Config(signature_version='s3v4'))

    def _carregar_malha_referencia(self):
        try:
            logger.info("🗺️ Baixando Malha Geográfica para resgate...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=self.config.MALHA_VIAS_PATH)
            df_hierarquico = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            self.df_lookup_vias = (
                df_hierarquico
                .explode("BAIRROS").unnest("BAIRROS")
                .explode("LOGRADOUROS").unnest("LOGRADOUROS")
                .with_columns(pl.col("H3_LIST").list.first().alias("H3_INDEX"))
                .select(["CIDADE", "BAIRRO", "RUA", "H3_INDEX"])
                .unique()
            )
            logger.info(f"✅ Malha preparada: {self.df_lookup_vias.height} logradouros mapeados.")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar malha: {e}. Resgate espacial desativado.")

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE"]: 
            return "NAO INFORMADO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _resolver_mapeamento(self, lista_colunas):
        mapeamento = {}
        for i, nome_col in enumerate(lista_colunas):
            if nome_col is None: continue
            col_limpa = "".join(c for c in str(nome_col).upper() if c.isalnum() or c == '_')
            for alvo, padroes in self.config.MAPA_COLUNAS.items():
                if alvo in mapeamento.values(): continue
                for p in padroes:
                    if re.search(p, col_limpa, re.IGNORECASE):
                        mapeamento[i] = alvo
                        break
        return mapeamento

    def _resgatar_espacial(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.df_lookup_vias is None: return df
        # Resgate Nível 1: Precisão Total
        df = df.join(self.df_lookup_vias, left_on=["MUNICIPIO", "BAIRRO", "LOGRADOURO"], 
                     right_on=["CIDADE", "BAIRRO", "RUA"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))).drop("H3_INDEX_right")
        # Resgate Nível 2: Fallback (Cidade + Rua)
        df_fb = self.df_lookup_vias.unique(subset=["CIDADE", "RUA"]).select(["CIDADE", "RUA", "H3_INDEX"])
        df = df.join(df_fb, left_on=["MUNICIPIO", "LOGRADOURO"], right_on=["CIDADE", "RUA"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))).drop("H3_INDEX_right")
        return df

    def _limpar_e_tipar(self, df: pl.DataFrame, ano: int) -> pl.DataFrame:
        total_entrada = df.height
        
        # 1. Normalização de Texto
        cols_ignorar = ["LATITUDE", "LONGITUDE", "NUM_BO", "HORAOCORRENCIA", "PERIODO_NATIVO"]
        for col in [c for c in df.columns if c not in cols_ignorar]:
            df = df.with_columns(pl.col(col).map_elements(self._normalizar_texto, return_dtype=pl.Utf8))

        # 2. Hierarquia de Período
        df = df.with_columns(pl.col("HORAOCORRENCIA").str.to_time(format="%H:%M:%S", strict=False).dt.hour().alias("_hr")) \
               .with_columns(
                   pl.when(pl.col("_hr").is_between(0, 5)).then(pl.lit("MADRUGADA"))
                   .when(pl.col("_hr").is_between(6, 11)).then(pl.lit("MANHA"))
                   .when(pl.col("_hr").is_between(12, 17)).then(pl.lit("TARDE"))
                   .when(pl.col("_hr").is_between(18, 23)).then(pl.lit("NOITE"))
                   .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
                   .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("MANHA|MANHÃ")).then(pl.lit("MANHA"))
                   .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("TARDE")).then(pl.lit("TARDE"))
                   .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("NOITE")).then(pl.lit("NOITE"))
                   .otherwise(pl.lit(None)).alias("SAZON_PERIODO")
               ).drop(["_hr", "PERIODO_NATIVO"])

        # 3. GPS Original (H3)
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ]).with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], self.config.RESOLUCAO_H3) if x["LATITUDE"] and x["LATITUDE"] != 0 else None,
                return_dtype=pl.Utf8
            ).alias("H3_INDEX")
        )

        bo_com_gps = df.filter(pl.col("H3_INDEX").is_not_null()).height
        
        # 4. Resgate via Malha
        df = self._resgatar_espacial(df)
        
        # 5. Exclusão Rígida (Aproveitamento Final)
        df_limpo = df.filter(pl.col("H3_INDEX").is_not_null() & pl.col("SAZON_PERIODO").is_not_null())
        bo_resgatados = df_limpo.height - bo_com_gps

        # Acumula estatísticas para o relatório consolidado
        self.audit_stats.append({
            "ano": ano,
            "total_raw": total_entrada,
            "com_gps": bo_com_gps,
            "resgatados": bo_resgatados,
            "descartados": total_entrada - df_limpo.height,
            "final_trusted": df_limpo.height
        })
        
        return df_limpo

    def finalizar_ciclo_auditoria(self):
        if not self.audit_stats: return

        # 1. Montagem do Relatório Discord
        msg = "📊 **[SafeDriver] Relatório Consolidado de Ingestão**\n```ml\n"
        msg += f"{'ANO':<5} | {'TOTAL':<7} | {'GPS':<6} | {'RESGATE':<7} | {'FINAL':<7}\n"
        msg += "-" * 45 + "\n"
        
        t_raw, t_gps, t_res, t_fin = 0, 0, 0, 0
        for s in sorted(self.audit_stats, key=lambda x: x['ano']):
            msg += f"{s['ano']:<5} | {s['total_raw']:<7} | {s['com_gps']:<6} | {s['resgatados']:<7} | {s['final_trusted']:<7}\n"
            t_raw += s['total_raw']; t_gps += s['com_gps']; t_res += s['resgatados']; t_fin += s['final_trusted']

        msg += "-" * 45 + "\n"
        msg += f"{'SOMA':<5} | {t_raw:<7} | {t_gps:<6} | {t_res:<7} | {t_fin:<7}\n```"
        
        # Indicadores de Eficiência
        taxa_resgate = (t_res / (t_raw - t_gps)) * 100 if (t_raw - t_gps) > 0 else 0
        taxa_aproveitamento = (t_fin / t_raw) * 100 if t_raw > 0 else 0
        
        msg += f"\n💡 **Eficiência da Malha:** {taxa_resgate:.2f}% dos BOs sem GPS foram recuperados."
        msg += f"\n🎯 **Aproveitamento Global:** {taxa_aproveitamento:.2f}% da base bruta virou Trusted."
        
        self._notificar_discord(msg)

        # 2. Geração e Upload do JSON de Auditoria
        audit_file = {
            "projeto": "SafeDriver",
            "data_execucao": str(datetime.now()),
            "stats_por_ano": self.audit_stats,
            "consolidado": {"total_bruto": t_raw, "total_gps": t_gps, "total_resgate": t_res, "total_trusted": t_fin}
        }
        buf = io.BytesIO(json.dumps(audit_file, indent=4).encode())
        self.s3.put_object(Bucket=self.config.NOME_BUCKET, 
                           Key="datalake/prata/auditoria/AUDITORIA_QUALIDADE_CRIMES.json", 
                           Body=buf.getvalue())
        logger.info("📁 Arquivo de auditoria salvo no R2.")

    def extrair_bronze(self, ano: int):
        path = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            res = requests.get(url, timeout=600)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path, Body=res.content)
                logger.info(f"✅ Bronze {ano} baixada.")
        except Exception as e: logger.error(f"❌ Erro Download {ano}: {e}")

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
                    df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine")
                    indices_map = self._resolver_mapeamento(df_raw.columns)
                    if "LATITUDE" not in indices_map.values():
                        for row_idx in range(min(20, df_raw.height)):
                            indices_map = self._resolver_mapeamento(df_raw.row(row_idx))
                            if "LATITUDE" in indices_map.values():
                                df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", read_options={"skip_rows": row_idx + 1})
                                indices_map = self._resolver_mapeamento(df_raw.columns)
                                break
                    df_mes = df_raw.select([df_raw.columns[i] for i in indices_map.keys()])
                    df_mes.columns = [indices_map[i] for i in indices_map.keys()]
                    list_dfs.append(df_mes.with_columns(pl.all().cast(pl.Utf8)))
                except: continue

            df_final = pl.concat(list_dfs, how="diagonal")
            df_final = self._limpar_e_tipar(df_final, ano)
            
            pepper = self.config.PEPPER
            df_final = df_final.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v).upper()}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
            ])
            buf = io.BytesIO(); df_final.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} finalizada.")

        except Exception as e: logger.error(f"💥 Erro fatal {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
    
    # Ao final de todos os anos, executa a auditoria consolidada
    if modo in ["prata", "tudo"]:
        ingestor.finalizar_ciclo_auditoria()
