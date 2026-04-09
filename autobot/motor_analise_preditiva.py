import sys, os, requests, traceback, hashlib, gc, warnings, re, time, json
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3, polars as pl, pandas as pd, numpy as np
import h3, holidays, boto3, shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

class Telemetria:
    def __init__(self):
        self.sucesso = os.environ.get("DISCORD_SUCESSO", "").strip(' "\'')
        self.erro = os.environ.get("DISCORD_ERRO", "").strip(' "\'')

    def _enviar_webhook(self, url, payload):
        if not url or not url.startswith("https://discord"):
            print("⚠️ AVISO: URL do Webhook ausente ou mal formatada.", file=sys.stderr)
            return
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code >= 400:
                print(f"❌ REJEIÇÃO DO DISCORD: Status {resp.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"❌ FALHA CONEXÃO DISCORD: {e}", file=sys.stderr)

    def notificar_sucesso(self, titulo, tempo_execucao, registros, media_risco, status_s3):
        payload = {
            "embeds": [{
                "title": f"🟢 {titulo}",
                "description": "**Relatório Executivo SafeDriver**\nO motor preditivo sincronizou os dados da SSP e atualizou o modelo com sucesso.",
                "color": 3066993, 
                "fields": [
                    {"name": "📊 Volumetria (Camada Prata)", "value": f"{registros:,} ocorrências", "inline": True},
                    {"name": "⚠️ Risco Médio", "value": f"{media_risco:.2f} pontos", "inline": True},
                    {"name": "⏱️ Tempo de Processamento", "value": f"{tempo_execucao:.1f} segundos", "inline": True},
                    {"name": "☁️ Backup Cloudflare R2", "value": status_s3, "inline": False}
                ],
                "footer": {"text": f"SafeDriver AI • Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        self._enviar_webhook(self.sucesso, payload)

    def notificar_info(self, titulo, corpo):
        payload = {
            "embeds": [{
                "title": f"🔵 {titulo}",
                "description": corpo,
                "color": 3447003,
                "footer": {"text": f"SafeDriver AI • Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        self._enviar_webhook(self.sucesso, payload)

    def notificar_erro(self, titulo, erro_msg):
        # Solução para evitar falhas de formatação Markdown que quebram o código
        bloco_inicio = "```python\n"
        bloco_fim = "\n```"
        resumo_erro = erro_msg[:1000]
        valor_stacktrace = f"{bloco_inicio}{resumo_erro}{bloco_fim}"

        payload = {
            "embeds": [{
                "title": f"🔴 {titulo}",
                "description": "**Falha Crítica no Pipeline MLOps**",
                "color": 15158332,
                "fields": [
                    {"name": "Detalhes Técnicos do Erro", "value": valor_stacktrace, "inline": False}
                ],
                "footer": {"text": f"SafeDriver AI Alerts • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        self._enviar_webhook(self.erro, payload)

class SafeDriver:
    def __init__(self):
        self.t_inicio = time.time()
        self.discord = Telemetria()
        self.pastas = {p: Path(f"datalake/{p}") for p in ["raw", "prata", "ouro"]}
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        cfg = {k: os.environ.get(k, "").strip() for k in ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]}
        if all(cfg.values()):
            self.s3 = boto3.client('s3', endpoint_url=cfg["R2_ENDPOINT_URL"], aws_access_key_id=cfg["R2_ACCESS_KEY_ID"], aws_secret_access_key=cfg["R2_SECRET_ACCESS_KEY"], region_name="auto")
            self.bucket = cfg["R2_BUCKET_NAME"]
        else:
            self.s3 = None
        self.feriados = list(holidays.Brazil(subdiv='SP', years=range(2022, 2027)).keys())
        self.meta = self.pastas["raw"] / "meta.json"

    def cdc_check(self, ano, url, sessao):
        try:
            # Trocando HEAD por GET com stream=True para driblar firewalls que bloqueiam HEAD
            r = sessao.get(url, timeout=30, verify=False, stream=True)
            if r.status_code != 200:
                print(f"⚠️ [CDC] Servidor recusou a conexão para {ano}. Status Code: {r.status_code}", file=sys.stderr)
                return True, 0 # Força tentar baixar de novo
            
            size = int(r.headers.get('Content-Length', 0))
            r.close() # Fecha a conexão rápido, pois só queriamos o tamanho
            
            if self.meta.exists():
                with open(self.meta, 'r') as f:
                    if json.load(f).get(str(ano)) == size: 
                        return False, size
            return True, size
        except Exception as e: 
            print(f"⚠️ [CDC] Erro ao checar tamanho de {ano}: {e}", file=sys.stderr)
            return True, 0

    def wkt(self, h3_id):
        try:
            b = h3.h3_to_geo_boundary(h3_id, geo_json=True)
            pts = ", ".join([f"{ln} {lt}" for ln, lt in b])
            return f"POLYGON(({pts}, {b[0][0]} {b[0][1]}))"
        except: return None

    def processar(self):
        print("Iniciando Verificação de Integridade (Self-Healing)...", file=sys.stdout)
        for f in self.pastas["raw"].glob("*.parquet"):
            try:
                cols = pl.scan_parquet(f).columns
                if "D" not in cols:
                    f.unlink()
                    print(f"🗑️ Cache corrompido removido: {f.name}")
            except:
                try: f.unlink()
                except: pass

        s = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        
        # 🛡️ SUPER HEADERS: Simulando um navegador real, vindo do próprio site da SSP, em Português
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.ssp.sp.gov.br/estatistica",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })
        
        novo, anos = False, []

        for ano in range(2022, datetime.now().year + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            path = self.pastas["raw"] / f"ssp_{ano}.parquet"
            
            run, sz = self.cdc_check(ano, url, s)
            if not run and path.exists(): 
                anos.append(ano)
                continue

            try:
                print(f"📥 Baixando SSP {ano}...", file=sys.stdout)
                r = s.get(url, timeout=(15, 300), verify=False, stream=True)
                
                if r.status_code == 200:
                    novo = True
                    tmp = self.pastas["raw"] / "tmp.xlsx"
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192 * 1024): 
                            if chunk: f.write(chunk)
                                
                    import fastexcel
                    ex = fastexcel.read_excel(str(tmp))
                    abas = []
                    
                    mapping = {
                        'LAT': ['LATITUDE', 'LAT', 'Y'], 'LON': ['LONGITUDE', 'LON', 'X'],
                        'D': ['DATAOCORRENCIA', 'DATA_FATO', 'DATA_REF', 'DATA'],
                        'H': ['HORAOCORRENCIA', 'HORA_FATO', 'HORA_REF', 'HORA'],
                        'N': ['RUBRICA', 'NATUREZA', 'NATUREZA_APURADA']
                    }
                    
                    for n in ex.sheet_names:
                        df = pl.read_excel(str(tmp), sheet_name=n, engine="calamine")
                        if len(df.columns) > 5:
                            df.columns = [c.upper().strip() for c in df.columns]
                            f_cols = {}
                            for target, aliases in mapping.items():
                                for col in df.columns:
                                    if col in aliases:
                                        f_cols[col] = target
                                        break
                            if all(k in f_cols.values() for k in ['LAT', 'LON', 'D', 'H', 'N']):
                                df_clean = df.rename(f_cols).select(['LAT', 'LON', 'D', 'H', 'N']).with_columns(pl.all().cast(pl.String))
                                abas.append(df_clean)
                                
                    if abas:
                        pl.concat(abas, how="diagonal").write_parquet(path)
                        anos.append(ano)
                        m_data = json.load(open(self.meta)) if self.meta.exists() else {}
                        m_data[str(ano)] = sz
                        with open(self.meta, 'w') as f: json.dump(m_data, f)
                    if os.path.exists(tmp): os.remove(tmp)
                else:
                    print(f"❌ [ERRO CRÍTICO] A SSP recusou a conexão para {ano}. Status retornado: {r.status_code}", file=sys.stderr)
            except Exception as e: 
                print(f"❌ [FALHA DE REDE] Erro ao conectar na SSP para {ano}: {e}", file=sys.stderr)
                continue

        arquivos_limpos = list(self.pastas["raw"].glob("*.parquet"))
        if not arquivos_limpos:
            self.discord.notificar_erro("SafeDriver Sync", "A Secretaria de Segurança (SSP) está indisponível ou bloqueando o acesso e não há base histórica local.")
            return

        if not novo and (self.pastas["ouro"] / "dashboard_final.parquet").exists():
            print("Nenhuma atualização pendente.")
            self.discord.notificar_info("SafeDriver Informação", "O pipeline rodou, mas os dados da SSP já estão atualizados na versão mais recente.")
            return

        print("⚙️ Processando Camada Prata (Traduzindo variáveis)...", file=sys.stdout)
        lf = pl.concat([pl.scan_parquet(f) for f in arquivos_limpos], how="diagonal")
        prata = lf.with_columns([
            pl.col("D").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_FATO"),
            pl.col("LAT").str.replace(",",".").cast(pl.Float32, strict=False),
            pl.col("LON").str.replace(",",".").cast(pl.Float32, strict=False),
            pl.col("H").str.slice(0,2).cast(pl.Int8, strict=False).alias("HORA_CRIME")
        ]).filter(pl.col("LAT").is_between(-25.5, -19.5)).with_columns([
            pl.when(pl.col("N").str.to_uppercase().str.contains("ROUBO|FURTO")).then(pl.lit("PATRIMONIO")).otherwise(pl.lit("PESSOA")).alias("TIPO_CRIME"),
            pl.when(pl.col("HORA_CRIME").is_between(6,18)).then(pl.lit("DIA")).otherwise(pl.lit("NOITE")).alias("PERIODO_DIA"),
            pl.col("DATA_FATO").dt.date().is_in(self.feriados).cast(pl.Int8).alias("EH_FERIADO"),
            ((pl.col("DATA_FATO").dt.day().is_between(28,31))|(pl.col("DATA_FATO").dt.day().is_between(1,7))).cast(pl.Int8).alias("SEMANA_PAGAMENTO")
        ]).collect()

        prata.with_columns(pl.col("LAT").hash().alias("ID_ANONIMO")).drop(["DATA_FATO","H","HORA_CRIME","N"]).write_parquet(self.pastas["prata"] / "camada_prata.parquet")
        
        print("🧠 Treinando IA Preditiva...", file=sys.stdout)
        c = prata.select(["LAT","LON"]).unique().to_pandas()
        c['CODIGO_H3'] = c.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        fato = prata.join(pl.from_pandas(c), on=["LAT","LON"]).group_by(["CODIGO_H3","PERIODO_DIA","TIPO_CRIME","EH_FERIADO","SEMANA_PAGAMENTO"]).agg([
            pl.len().alias("TOTAL_OCORRENCIAS"), 
            pl.col("LAT").mean().alias("LATITUDE_MEDIA"), 
            pl.col("LON").mean().alias("LONGITUDE_MEDIA")
        ])
        
        X = fato.select(["LATITUDE_MEDIA","LONGITUDE_MEDIA","EH_FERIADO","SEMANA_PAGAMENTO"]).to_pandas()
        y = np.log1p(fato.select("TOTAL_OCORRENCIAS").to_numpy().ravel())
        
        ens = VotingRegressor([('c', CatBoostRegressor(iterations=100, silent=True)), ('l', LGBMRegressor(n_estimators=100, verbose=-1))]).fit(X, y)
        preds = ens.predict(X)
        risco_avg = np.mean(np.expm1(preds))

        fato.with_columns([
            pl.Series("ESCORE_RISCO", np.round(np.expm1(preds), 2)),
            pl.col("CODIGO_H3").map_elements(self.wkt, return_dtype=pl.String).alias("GEOMETRIA_WKT")
        ]).write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")

        sd = pd.DataFrame(shap.TreeExplainer(ens.estimators_[0]).shap_values(X), columns=X.columns).abs().mean().to_frame("GRAU_IMPORTANCIA").reset_index()
        sd.columns = ["VARIAVEL", "GRAU_IMPORTANCIA"]
        pl.from_pandas(sd).write_parquet(self.pastas["ouro"] / "shap_audit.parquet")

        status_cloud = "❌ Desconectado"
        if self.s3:
            try:
                for f in self.pastas["ouro"].glob("*.parquet"):
                    self.s3.upload_file(str(f), self.bucket, f"ouro/{f.name}")
                status_cloud = "✅ Upload Realizado"
            except: status_cloud = "⚠️ Falha no Backup R2"
        
        tempo_total = time.time() - self.t_inicio
        self.discord.notificar_sucesso("Execução Concluída", tempo_total, prata.height, risco_avg, status_cloud)

if __name__ == "__main__":
    app = SafeDriver()
    try:
        app.processar()
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        app.discord.notificar_erro("Falha Sistêmica", err)
        sys.exit(1)
