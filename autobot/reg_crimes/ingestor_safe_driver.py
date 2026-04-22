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
        "NUM_BO": [r"NUM.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN", r"NOME_MUNICIPIO$"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"ENDERECO"],
        "DATAOCORRENCIA": [r"DATA_OCORRENCIA_BO", r"DT_OCORR", r"DATA_OCORRENCIA"],
        "DATAREGISTRO": [r"DATA_REGISTRO", r"DT_REGISTRO"],
        "HORAOCORRENCIA": [r"HORA_OCORRENCIA_BO", r"HR_OCORR", r"HORA_OCORRENCIA"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"COORDENADA_X", r"LATITUD"],
        "LONGITUDE": [r"LONGITUDE", r"COORDENADA_Y", r"LONGITUD"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL"],
        "PERIODO_NATIVO": [r"DESC_PERIODO", r"PERIODO", r"DS_PERIODO", r"DESC_PERIODO_OCORRENCIA"]
    }

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year
        self.df_lookup_vias = None
        self.df_lookup_prefix = None
        self.df_lookup_bairro = None
        self.audit_stats = []

    def _inicializar_s3(self):
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        if endpoint.endswith(f"/{self.config.NOME_BUCKET}"):
            endpoint = endpoint[: -len(f"/{self.config.NOME_BUCKET}")]
        return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=self.config.ACCESS_KEY.strip(),
                            aws_secret_access_key=self.config.SECRET_KEY.strip(), config=Config(signature_version='s3v4'))

    def _limpeza_extrema(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", "0", ".", "NAO INFORMADO"]:
            return "DESCONHECIDO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        texto = re.sub(r'[^A-Z0-9\s]', ' ', texto.upper())
        texto = " ".join(texto.split())
        subs_cidades = {
            r'\bS PAULO\b': 'SAO PAULO', r'\bS\.PAULO\b': 'SAO PAULO',
            r'\bS BERNARDO\b': 'SAO BERNARDO DO CAMPO',
            r'\bS CAETANO\b': 'SAO CAETANO DO SUL', r'\bS ANDRE\b': 'SANTO ANDRE'
        }
        for erro, correto in subs_cidades.items():
            texto = re.sub(erro, correto, texto)
        return texto.strip() if texto.strip() != "" else "DESCONHECIDO"

    def _normalizar_logradouro(self, texto):
        t = self._limpeza_extrema(texto)
        if t == "DESCONHECIDO" or any(x in t for x in ["VEDACAO", "VIA PUBLICA", "NAO INFORMADO"]):
            return "DESCONHECIDO"
        subs = {
            r'\bJD\b': 'JARDIM', r'\bVL\b': 'VILA', r'\bSTA\b': 'SANTA', r'\bSTO\b': 'SANTO',
            r'\bPC\b': 'PRACA', r'\bTV\b': 'TRAVESSA', r'\bAV\b': 'AVENIDA', r'\bR\b': 'RUA'
        }
        for sigla, expansao in subs.items():
            t = re.sub(sigla, expansao, t)
        prefixos = r'^(RUA|AVENIDA|TRAVESSA|PRACA|ALAMEDA|ESTRADA|RODOVIA|LADEIRA|VIADUTO|MARGINAL)\s+'
        return re.sub(prefixos, '', t).strip()

    def _carregar_malha_referencia(self):
        try:
            logger.info("🗺️ Carregando Malha com Normalização Universal...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=self.config.MALHA_VIAS_PATH)
            df_flat = pl.read_parquet(io.BytesIO(obj['Body'].read())).explode("BAIRROS").unnest("BAIRROS").explode("LOGRADOUROS").unnest("LOGRADOUROS")
            df_base = df_flat.with_columns([
                pl.col("H3_LIST").list.first().alias("H3_INDEX"),
                pl.col("CIDADE").map_elements(self._limpeza_extrema, return_dtype=pl.Utf8).alias("CID_NORM"),
                pl.col("BAIRRO").map_elements(self._limpeza_extrema, return_dtype=pl.Utf8).alias("BAI_NORM"),
                pl.col("RUA").map_elements(self._normalizar_logradouro, return_dtype=pl.Utf8).alias("RUA_BASE")
            ]).select(["CID_NORM", "BAI_NORM", "RUA_BASE", "H3_INDEX"])
            self.df_lookup_vias = df_base.unique(subset=["CID_NORM", "BAI_NORM", "RUA_BASE"])
            df_prefix = df_base.with_columns(pl.col("RUA_BASE").str.replace_all(" ", "").str.slice(0, 10).alias("RUA_PREFIX"))
            valid_p = df_prefix.group_by(["CID_NORM", "RUA_PREFIX"]).agg(pl.col("H3_INDEX").n_unique().alias("n")).filter(pl.col("n") == 1)
            self.df_lookup_prefix = df_prefix.join(valid_p.select(["CID_NORM", "RUA_PREFIX"]), on=["CID_NORM", "RUA_PREFIX"], how="inner").unique(subset=["CID_NORM", "RUA_PREFIX"])
            self.df_lookup_bairro = df_base.group_by(["CID_NORM", "BAI_NORM"]).agg(pl.col("H3_INDEX").mode().first().alias("H3_BAIRRO"))
            logger.info(f"✅ Malha pronta. Vias: {self.df_lookup_vias.height} | Bairros: {self.df_lookup_bairro.height}")
        except Exception as e: logger.error(f"❌ Erro malha: {e}")

    def _resgatar_espacial(self, df: pl.DataFrame):
        if self.df_lookup_vias is None: 
            return df, {"p1_exato": 0, "p2_prefixo": 0, "p3_bairro": 0}
        df = df.with_columns([
            pl.col("MUNICIPIO").map_elements(self._limpeza_extrema, return_dtype=pl.Utf8).alias("MUN_NORM"),
            pl.col("BAIRRO").map_elements(self._limpeza_extrema, return_dtype=pl.Utf8).alias("BAI_NORM"),
            pl.col("LOGRADOURO").map_elements(self._normalizar_logradouro, return_dtype=pl.Utf8).alias("LOG_BASE")
        ])
        df = df.with_columns(pl.col("LOG_BASE").str.replace_all(" ", "").str.slice(0, 10).alias("LOG_PREFIX"))
        count_init = df.filter(pl.col("H3_INDEX").is_not_null()).height
        
        df = df.join(self.df_lookup_vias, left_on=["MUN_NORM", "BAI_NORM", "LOG_BASE"], right_on=["CID_NORM", "BAI_NORM", "RUA_BASE"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))).drop("H3_INDEX_right")
        count_p1 = df.filter(pl.col("H3_INDEX").is_not_null()).height
        resgatados_p1 = count_p1 - count_init
        
        df = df.join(self.df_lookup_prefix.select(["CID_NORM", "RUA_PREFIX", "H3_INDEX"]), left_on=["MUN_NORM", "LOG_PREFIX"], right_on=["CID_NORM", "RUA_PREFIX"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))).drop("H3_INDEX_right")
        count_p2 = df.filter(pl.col("H3_INDEX").is_not_null()).height
        resgatados_p2 = count_p2 - count_p1
        
        df = df.join(self.df_lookup_bairro, left_on=["MUN_NORM", "BAI_NORM"], right_on=["CID_NORM", "BAI_NORM"], how="left") \
               .with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_BAIRRO"))).drop("H3_BAIRRO")
        count_p3 = df.filter(pl.col("H3_INDEX").is_not_null()).height
        resgatados_p3 = count_p3 - count_p2
        
        return df.drop(["MUN_NORM", "BAI_NORM", "LOG_BASE", "LOG_PREFIX"]), {"p1_exato": resgatados_p1, "p2_prefixo": resgatados_p2, "p3_bairro": resgatados_p3}

    def _limpar_e_tipar(self, df: pl.DataFrame, ano: int, tempo_inicio_ano: float) -> pl.DataFrame:
        total_entrada = df.height
        df = df.with_columns(pl.all().cast(pl.Utf8).fill_null("NAO INFORMADO"))

        # ✨ NORMALIZAÇÃO TEXTUAL FORTE (Preparando o terreno para a Ouro)
        if "RUBRICA" in df.columns:
            df = df.with_columns(
                pl.col("RUBRICA").map_elements(self._limpeza_extrema, return_dtype=pl.Utf8)
            )

        # GAP DATA: Lógica de Fallback Dinâmica
        if "DATAREGISTRO" in df.columns:
            df = df.with_columns([
                pl.col("DATAOCORRENCIA").str.to_date(strict=False).alias("_dt_oc"),
                pl.col("DATAREGISTRO").str.to_date(strict=False).alias("_dt_re")
            ]).with_columns(
                pl.coalesce(["_dt_oc", "_dt_re"]).alias("DATAOCORRENCIA")
            ).drop(["_dt_oc", "_dt_re"])
        else:
            df = df.with_columns(pl.col("DATAOCORRENCIA").str.to_date(strict=False))

        # GPS Original
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

        # Resgate Espacial 1-2-3
        df, stats_resgate = self._resgatar_espacial(df)

        # GAP PERÍODO: Lógica Híbrida (Hora + Texto)
        df = df.with_columns([
            pl.col("HORAOCORRENCIA").str.extract(r"(\d{1,2})", 1).cast(pl.Int32, strict=False).alias("_hr_t"),
            pl.col("PERIODO_NATIVO").str.to_uppercase().alias("_per_txt")
        ])

        df = df.with_columns(
            pl.when(pl.col("_hr_t").is_between(0, 5) | pl.col("_per_txt").str.contains("MADRUGADA"))
            .then(pl.lit("MADRUGADA"))
            .when(pl.col("_hr_t").is_between(6, 11) | pl.col("_per_txt").str.contains("MANH"))
            .then(pl.lit("MANHA"))
            .when(pl.col("_hr_t").is_between(12, 17) | pl.col("_per_txt").str.contains("TARDE"))
            .then(pl.lit("TARDE"))
            .when(pl.col("_hr_t").is_between(18, 23) | pl.col("_per_txt").str.contains("NOITE"))
            .then(pl.lit("NOITE"))
            .otherwise(pl.lit("INCERTO")).alias("SAZON_PERIODO")
        )

        # Safe Drop: Remove apenas o que existe
        cols_remover = ["_lat_f", "_lon_f", "_hr_t", "_per_txt", "DATAREGISTRO", "PERIODO_NATIVO"]
        existing_cols_to_drop = [c for c in cols_remover if c in df.columns]

        df_trusted = df.filter(pl.col("H3_INDEX").is_not_null()).with_columns([
            pl.col("_lat_f").cast(pl.Float64, strict=False).alias("LATITUDE"),
            pl.col("_lon_f").cast(pl.Float64, strict=False).alias("LONGITUDE")
        ]).drop(existing_cols_to_drop)

        # Auditoria por Período
        cp = df_trusted.group_by("SAZON_PERIODO").len().to_dicts()
        dict_periodos = {d["SAZON_PERIODO"]: d["len"] for d in cp}

        tempo_exec = round(time.time() - tempo_inicio_ano, 2)
        self.audit_stats.append({
            "ano_referencia": int(ano),
            "telemetria_funil": {
                "1_total_bruto_ssp": total_entrada,
                "2_com_coordenadas_validas": bo_com_gps,
                "3_resgatados_passo1_rua_exata": stats_resgate["p1_exato"],
                "4_resgatados_passo2_rua_prefixo": stats_resgate["p2_prefixo"],
                "5_resgatados_passo3_centroide_bairro": stats_resgate["p3_bairro"],
                "6_total_salvo_pela_malha": sum(stats_resgate.values()),
                "8_total_final_trusted": df_trusted.height
            },
            "contagem_por_periodo": dict_periodos,
            "performance_segundos": tempo_exec
        })
        return df_trusted

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
        tempo_inicio_ano = time.time()
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
                df_final = self._limpar_e_tipar(df_final, ano, tempo_inicio_ano)
                pepper = self.config.PEPPER
                df_final = df_final.with_columns([pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v).upper()}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)])
                buf = io.BytesIO(); df_final.write_parquet(buf, compression="zstd")
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
                logger.info(f"✨ Prata {ano} finalizada.")
        except Exception as e: logger.error(f"💥 Erro fatal {ano}: {e}")

    def finalizar_ciclo_auditoria(self):
        if not self.audit_stats: return
        t_raw, t_gps, t_salvos, t_fin = 0, 0, 0, 0
        p_mad, p_man, p_tar, p_noi, p_inc = 0, 0, 0, 0, 0
        for s in self.audit_stats:
            f = s["telemetria_funil"]
            t_raw += f["1_total_bruto_ssp"]; t_gps += f["2_com_coordenadas_validas"]
            t_salvos += f["6_total_salvo_pela_malha"]; t_fin += f["8_total_final_trusted"]
            cp = s["contagem_por_periodo"]
            p_mad += cp.get("MADRUGADA", 0); p_man += cp.get("MANHA", 0)
            p_tar += cp.get("TARDE", 0); p_noi += cp.get("NOITE", 0); p_inc += cp.get("INCERTO", 0)

        msg = "📊 **[SafeDriver] Relatório Consolidado de Crimes**\n```ml\n"
        msg += f"• Total SSP Bruto: {t_raw}\n• GPS Original   : {t_gps}\n• Salvos Malha   : {t_salvos}\n"
        msg += f"• Total TRUSTED  : {t_fin} ({(t_fin/t_raw*100) if t_raw > 0 else 0:.1f}%)\n"
        msg += "-----------------------------------\n"
        msg += "BREAKDOWN POR PERÍODO (Trusted):\n"
        msg += f"• MADRUGADA : {p_mad}\n• MANHA     : {p_man}\n• TARDE     : {p_tar}\n• NOITE     : {p_noi}\n• INCERTO   : {p_inc}\n```"
        if self.config.WEBHOOK_DISCORD: 
            try: requests.post(self.config.WEBHOOK_DISCORD, json={"content": msg})
            except: pass
        audit_json = {"projeto": "SafeDriver", "data_processamento": str(datetime.now()), "stats_anuais": self.audit_stats}
        self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key="datalake/prata/auditoria/AUDITORIA_QUALIDADE_CRIMES_PERIODOS.json", Body=json.dumps(audit_json, indent=4).encode())

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
    if modo in ["prata", "tudo"]: ingestor.finalizar_ciclo_auditoria()
