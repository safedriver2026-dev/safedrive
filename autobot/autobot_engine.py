import pandas as pd
import numpy as np
import os, io, requests, json, hashlib, unicodedata
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import h3
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class MotorSeguranca:
    def __init__(self, persistencia=True):
        self.identificador = "MOTOR-RECONSTRUTOR-AUTONOMO"
        self.persistencia = persistencia
        self.tokens = {"sucesso": os.environ.get('DISCORD_SUCESSO'), "erro": os.environ.get('DISCORD_ERRO')}
        self.url_base = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{}.xlsx"
        self.perfis_alvo = {
            "Motorista": ["VEICULO", "CARRO", "AUTO", "CARGA", "CAMINHAO"],
            "Motociclista": ["MOTO", "MOTOCICLETA"],
            "Ciclista": ["BICI", "BICICLETA", "BIKE"],
            "Pedestre": ["CELULAR", "ONIBUS", "TRANSEUNTE", "PEDESTRE", "BOLSA"]
        }
        self.banco = self._conectar_nuvem() if persistencia else None
        self.telemetria = {
            "camadas": {"bronze": 0, "silver": 0, "gold": 0},
            "perfis": {p: 0 for p in self.perfis_alvo.keys()},
            "ia": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "c": 0.0},
            "cloud": {"eco": 0.0, "delta": 0, "downloads": 0}
        }

    def _conectar_nuvem(self):
        cfg = os.environ.get('FIREBASE_JSON')
        if cfg and not firebase_admin._apps:
            try:
                cred = credentials.Certificate(json.loads(cfg))
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except: return None
        return None

    def _limpar_texto(self, t):
        if pd.isna(t) or not isinstance(t, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _classificar_perfil(self, linha):
        t = self._limpar_texto(" ".join([str(v) for v in linha.values if pd.api.types.is_scalar(v)]))
        for p, keywords in self.perfis_alvo.items():
            if any(k in t for k in keywords): return p
        return "Outros"

    def _definir_peso(self, linha):
        t = self._limpar_texto(" ".join([str(v) for v in linha.values if pd.api.types.is_scalar(v)]))
        if any(w in t for w in ["LATROCINIO", "MORTE", "HOMICIDIO", "SEQUESTRO"]): return 10.0
        if "ROUBO" in t: return 8.5
        return 4.0 if "FURTO" in t else 1.5

    def _atualizar_bronze(self):
        os.makedirs('datalake/camada_bronze_bruta', exist_ok=True)
        ano_atual = datetime.now().year
        for ano in range(2024, ano_atual + 1):
            caminho_parquet = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
            if not os.path.exists(caminho_parquet):
                url = self.url_base.format(ano)
                try:
                    resp = requests.get(url, timeout=60)
                    if resp.status_code == 200:
                        df_tmp = pd.read_excel(io.BytesIO(resp.content))
                        df_tmp.to_parquet(caminho_parquet, index=False)
                        self.telemetria['cloud']['downloads'] += 1
                except: pass

    def _gerar_camada_silver(self, df):
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df[(df['LATITUDE'] < -19.0) & (df['LATITUDE'] > -26.0) & (df['LONGITUDE'] < -44.0) & (df['LONGITUDE'] > -54.0)]
        self.telemetria["camadas"]["silver"] = len(df)
        return df

    def _gerar_camada_ouro(self, dados):
        dados = self._gerar_camada_silver(dados)
        if dados.empty: return pd.DataFrame()
        dados['perfil'] = dados.apply(self._classificar_perfil, axis=1)
        dados['id_h3'] = dados.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)
        dados['nota'] = dados.apply(self._definir_peso, axis=1)
        
        v_perfil = dados['perfil'].value_counts().to_dict()
        for p in self.telemetria['perfis']: self.telemetria['perfis'][p] = v_perfil.get(p, 0)
        
        X, y = dados[['LATITUDE', 'LONGITUDE']], dados['nota']
        if len(X) > 30:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100)
            m.fit(xt, yt)
            p_vals = m.predict(xv)
            self.telemetria['ia'] = {
                "mae": round(mean_absolute_error(yv, p_vals), 4),
                "rmse": round(root_mean_squared_error(yv, p_vals), 4),
                "r2": round(r2_score(yv, p_vals), 4),
                "c": round(max(0.0, 100.0 - (mean_absolute_error(yv, p_vals) * 10)), 2)
            }
        
        res = dados.groupby(['id_h3', 'perfil']).agg({'nota': 'mean', 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['h3', 'p', 'r', 'lat', 'lon']
        self.telemetria["camadas"]["gold"] = len(res)
        return res

    def _sync_deltasync(self, dados):
        if not self.banco or dados.empty: return
        ref_path = 'datalake/camada_prata_confiavel/assinatura_anterior.parquet'
        dados['hash'] = dados.apply(lambda r: hashlib.sha256(f"{r['h3']}{r['p']}{r['r']}".encode()).hexdigest(), axis=1)
        if os.path.exists(ref_path):
            antigo = pd.read_parquet(ref_path)
            delta = dados[~dados['hash'].isin(antigo['hash'])].copy()
            self.telemetria['cloud']['eco'] = (1 - (len(delta) / len(dados))) * 100
            self.telemetria['cloud']['delta'] = len(delta)
        else:
            delta = dados
            self.telemetria['cloud']['eco'] = 0.0
            self.telemetria['cloud']['delta'] = len(dados)
        if not delta.empty:
            lote = self.banco.batch()
            for i, r in delta.iterrows():
                doc_id = f"{r['h3']}_{r['p']}"
                ref = self.banco.collection('malha_risco').document(doc_id)
                lote.set(ref, {'r': round(r['r'], 2), 'p': r['p'], 'c': [r['lat'], r['lon']], 'u': firestore.SERVER_TIMESTAMP}, merge=True)
                if (i + 1) % 400 == 0:
                    lote.commit(); lote = self.banco.batch()
            lote.commit()
        dados.to_parquet(ref_path, index=False)

    def _notificar(self, status, msg=None):
        url = self.tokens["sucesso"] if status else self.tokens["erro"]
        if not url: return
        p_info = "\n".join([f"🔹 **{k}:** {v}" for k, v in self.telemetria['perfis'].items()])
        c_info = f"📦 **Bronze:** {self.telemetria['camadas']['bronze']}\n🥈 **Silver:** {self.telemetria['camadas']['silver']}\n🏆 **Gold:** {self.telemetria['camadas']['gold']}"
        ia_info = f"📉 **MAE:** {self.telemetria['ia']['mae']} | **RMSE:** {self.telemetria['ia']['rmse']}\n📈 **R²:** {self.telemetria['ia']['r2']} | **Confiança:** {self.telemetria['ia']['c']}%"
        cloud_info = f"💰 **Economia:** {self.telemetria['cloud']['eco']:.2f}%\n🆕 **Delta:** {self.telemetria['cloud']['delta']} | 📥 **Downloads:** {self.telemetria['cloud']['downloads']}"
        embed = {"title": "🛡️ DASHBOARD INTEGRAL - SAFE DRIVER", "color": 3066993 if status else 15158332, "fields": [
            {"name": "📊 CAMADAS DE DADOS", "value": c_info, "inline": False},
            {"name": "👥 PERFIS MONITORADOS", "value": p_info, "inline": True},
            {"name": "🧠 INTELIGÊNCIA ARTIFICIAL", "value": ia_info, "inline": False},
            {"name": "☁️ SINCRONIZAÇÃO NUVEM", "value": cloud_info, "inline": True}
        ], "footer": {"text": f"Execução: {datetime.now().strftime('%Y%m%d%H%M')}"}}
        if msg: embed["description"] = f"**Status:** `{msg}`"
        requests.post(url, json={"embeds": [embed]})

    def processar(self):
        try:
            self._atualizar_bronze()
            mestre = pd.DataFrame()
            for ano in range(2024, datetime.now().year + 1):
                path = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
                if os.path.exists(path): mestre = pd.concat([mestre, pd.read_parquet(path)])
            if mestre.empty:
                self._notificar(True, msg="Sistema reconstruído: Zero registros identificados no portal SSP.")
            else:
                self.telemetria['camadas']['bronze'] = len(mestre)
                ouro = self._gerar_camada_ouro(mestre)
                self._sync_deltasync(ouro)
                self._notificar(True, msg="Malha de segurança reconstruída e sincronizada.")
        except Exception as e:
            self._notificar(False, str(e))

if __name__ == "__main__":
    MotorSeguranca().processar()
