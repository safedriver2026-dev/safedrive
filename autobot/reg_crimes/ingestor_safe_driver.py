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

    def _extrair_nome_base(self, texto):
        """Padroniza nomes de ruas removendo prefixos (AV, RUA, etc) para resgate flexível."""
        if not texto or str(texto).upper() in ["NAO INFORMADO", "NULL", "NAN"]: return "NAO INFORMADO"
        # Remove acentos e bota em caixa alta
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(texto)) if unicodedata.category(c) != 'Mn').upper()
        # Remove prefixos comuns no início da string
        prefixos = r'^(RUA|R|AVENIDA|AV|TRAVESSA|TV|PRACA|PC|ALAMEDA|AL|ESTRADA|EST|RODOVIA|ROD|LADEIRA|LD|VIADUTO|VD|BOULEVARD|BLV)\s+'
        return re.sub(prefixos, '', texto.strip())

    def _carregar_malha_referencia(self):
        try:
            logger.info("🗺️ Baixando Malha Geográfica para resgate...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=self.config.MALHA_VIAS_PATH)
            df_h = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            
            self.df_lookup_vias = (
                df_h.explode("BAIRROS").unnest("BAIRROS")
                .explode("LOGRADOUROS").unnest("LOGRADOUROS")
                .with_columns(pl.col("H3_LIST").list.first().alias("H3_INDEX"))
                .select(["CIDADE", "BAIRRO", "RUA", "H3_INDEX"])
                .with_columns(pl.col("RUA").map_elements(self._extrair_nome_base, return_dtype=pl.Utf8).alias("RUA_BASE"))
                .unique()
            )
            logger.info(f"✅ Malha preparada: {self.df_lookup_vias.height} endereços mapeados.")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar malha: {e}. Resgate espacial limitado.")

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
        
        # Cria a chave de busca simplificada no dataframe de crimes
        df = df.with_columns(pl.col("LOGRADOURO").map_elements(self._extrair_nome_base, return_dtype=pl.Utf8).alias("LOG_BASE"))

        # 1. Join por Nome Base + Bairro + Cidade
        df = df.join(self.df_lookup_vias.select(["CIDADE", "BAIRRO", "RUA_BASE", "H3_INDEX"]).unique(),
                     left_on=["MUNICIPIO", "BAIRRO", "LOG_BASE"],
                     right_on=["CIDADE", "BAIRRO", "RUA_BASE"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))) \
               .drop("H3_INDEX_right")

        # 2. Join Fallback (Cidade + Nome Base) - Se o bairro estiver divergente entre SSP e IBGE
        df_fb = self.df_lookup_vias.unique(subset=["CIDADE", "RUA_BASE"]).select(["CIDADE", "RUA_BASE", "H3_INDEX"])
        df = df.join(df_fb, left_on=["MUNICIPIO", "LOG_BASE"], right_on=["CIDADE", "RUA_BASE"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))) \
               .drop("H3_INDEX_right")
        
        return df.drop("LOG_BASE")

    def _limpar_e_tipar(self, df: pl.DataFrame, ano: int) -> pl.DataFrame:
        total_entrada = df.height
        
        # --- ETAPA 1: TUDO PARA STRING (Proteção contra inferência do Excel) ---
        df = df.with_columns(pl.all().cast(pl.Utf8).fill_null("NAO INFORMADO"))

        # --- ETAPA 2: TRATAMENTO GEOGRÁFICO ---
        # Extração resiliente de coordenadas
        df = df.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").alias("_lat_f"),
            pl.col("LONGITUDE").str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").alias("_lon_f")
        ])

        # Geração de H3 via GPS Original (onde lat/lon são válidos)
        df = df.with_columns(
            pl.struct(["_lat_f", "_lon_f"]).map_elements(
                lambda x: h3.latlng_to_cell(float(x["_lat_f"]), float(x["_lon_f"]), self.config.RESOLUCAO_H3) 
                if x["_lat_f"] and float(x["_lat_f"]) < -10 else None,
                return_dtype=pl.Utf8
            ).alias("H3_INDEX")
        )

        bo_com_gps = df.filter(pl.col("H3_INDEX").is_not_null()).height

        # Resgate via Malha de Referência (Nome Base)
        df = self._resgatar_espacial(df)

        # --- ETAPA 3: TRATAMENTO TEMPORAL (Hierarquia de Recuperação) ---
        # Tenta pegar a Hora. Se não der, tenta o período nativo. Se não der, "INCERTO".
        df = df.with_columns(
            pl.col("HORAOCORRENCIA").str.extract(r"(\d{1,2})", 1).cast(pl.Int32, strict=False).alias("_hr_temp")
        ).with_columns(
            pl.when(pl.col("_hr_temp").is_between(0, 5)).then(pl.lit("MADRUGADA"))
            .when(pl.col("_hr_temp").is_between(6, 11)).then(pl.lit("MANHA"))
            .when(pl.col("_hr_temp").is_between(12, 17)).then(pl.lit("TARDE"))
            .when(pl.col("_hr_temp").is_between(18, 23)).then(pl.lit("NOITE"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("MANHA|MANHÃ")).then(pl.lit("MANHA"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("TARDE")).then(pl.lit("TARDE"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("NOITE")).then(pl.lit("NOITE"))
            .otherwise(pl.lit("INCERTO")).alias("SAZON_PERIODO")
        )

        # --- ETAPA 4: CONVERSÃO DE TIPOS FINAIS ---
        # Só mantemos o que tem localização. O tempo aceitamos "INCERTO".
        df_trusted = df.filter(pl.col("H3_INDEX").is_not_null()).with_columns([
            pl.col("DATAOCORRENCIA").str.to_date(strict=False).alias("DATAOCORRENCIA"),
            pl.col("_lat_f").cast(pl.Float64, strict=False).alias("LATITUDE"),
            pl.col("_lon_f").cast(pl.Float64, strict=False).alias("LONGITUDE")
        ]).drop(["_lat_f", "_lon_f", "_hr_temp"])

        # Estatísticas para Auditoria
        self.audit_stats.append({
            "ano": ano,
            "total_bruto": total_entrada,
            "com_gps_original": bo_com_gps,
            "resgatados_via_malha": df_trusted.height - bo_com_gps,
            "total_trusted": df_trusted.height,
            "taxa_aproveitamento": round((df_trusted.height / total_entrada) * 100, 2)
        })
        
        return df_trusted

    def finalizar_ciclo_auditoria(self):
        if not self.audit_stats: return

        # 1. Relatório Discord (Tabela)
        msg = "📊 **[SafeDriver] Relatório de Ingestão de Crimes**\n```ml\n"
        msg += f"{'ANO':<5} | {'TOTAL':<7} | {'GPS':<6} | {'RESGATE':<7} | {'FINAL':<7}\n"
        msg += "-" * 45 + "\n"
        t_raw, t_res, t_fin = 0, 0, 0
        for s in sorted(self.audit_stats, key=lambda x: x['ano']):
            msg += f"{s['ano']:<5} | {s['total_bruto']:<7} | {s['com_gps_original']:<6} | {s['resgatados_via_malha']:<7} | {s['total_trusted']:<7}\n"
            t_raw += s['total_bruto']; t_res += s['resgatados_via_malha']; t_fin += s['total_trusted']
        msg += "-" * 45 + "\n"
        msg += f"{'SOMA':<5} | {t_raw:<7} | {'-':<6} | {t_res:<7} | {t_fin:<7}\n```"
        
        taxa_recup = (t_res / (t_raw - (t_fin-t_res)) * 100) if t_raw > 0 else 0
        msg += f"\n💡 **Inteligência:** {taxa_recup:.2f}% dos registros sem GPS foram salvos pela Malha Base."
        self._notificar_discord(msg)

        # 2. Upload do JSON de Auditoria
        audit_json = {"projeto": "SafeDriver", "execucao": str(datetime.now()), "stats": self.audit_stats}
        self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key="datalake/prata/auditoria/AUDITORIA_QUALIDADE_CRIMES.json", 
                           Body=json.dumps(audit_json, indent=4).encode())

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"
        try:
            if self.df_lookup_vias is None: self._carregar_malha_referencia()
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_bytes = obj['Body'].read()
            
            # Carrega abas úteis
            excel_reader = fastexcel.read_excel(excel_bytes)
            abas = [n for n in excel_reader.sheet_names if not any(x in n.upper() for x in ["CAPA", "DICIONARIO", "LEGENDA", "CAMPOS", "SPDADOS"])]
            
            list_dfs = []
            for nome_aba in abas:
                try:
                    df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine")
                    # FORÇAR STRING IMEDIATO
                    df_raw = df_raw.with_columns(pl.all().cast(pl.Utf8))
                    
                    indices_map = self._resolver_mapeamento(df_raw.columns)
                    if not indices_map:
                        for row_idx in range(min(20, df_raw.height)):
                            if "LATITUDE" in self._resolver_mapeamento(df_raw.row(row_idx)).values():
                                df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", read_options={"skip_rows": row_idx + 1})
                                df_raw = df_raw.with_columns(pl.all().cast(pl.Utf8))
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
                
                # Anonimização e Upload
                pepper = self.config.PEPPER
                df_final = df_final.with_columns([
                    pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v).upper()}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
                ])
                buf = io.BytesIO(); df_final.write_parquet(buf, compression="zstd")
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
                logger.info(f"✨ Prata {ano} finalizada com sucesso.")
        except Exception as e: logger.error(f"💥 Erro fatal {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
    if modo in ["prata", "tudo"]:
        ingestor.finalizar_ciclo_auditoria()
