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
        self.identificador = "SISTEMA-AUTONOMO-TELEMETRIA-AVANCADA"
        self.persistencia = persistencia
        self.tokens = {
            "sucesso": os.environ.get('DISCORD_SUCESSO'),
            "erro": os.environ.get('DISCORD_ERRO')
        }
        self.banco = self._conectar_nuvem() if persistencia else None
        self.auditoria = {
            "v": {"b": 0, "s": 0, "g": 0},
            "ia": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "c": 0.0},
            "sync": {"eco": 0.0, "delta": 0, "total": 0}
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

    def _definir_peso(self, linha):
        t = self._limpar_texto(" ".join([str(v) for v in linha.values if pd.api.types.is_scalar(v) and pd.notnull(v)]))
        if any(w in t for w in ["LATROCINIO", "MORTE", "HOMICIDIO"]): return 10.0
        if "ROUBO" in t: return 8.0
        if "FURTO" in t: return 4.0
        return 1.0

    def _gerar_camada_silver(self, df):
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df[(df['LATITUDE'] < -19.0) & (df['LATITUDE'] > -26.0) & 
                (df['LONGITUDE'] < -44.0) & (df['LONGITUDE'] > -54.0)]
        self.auditoria["v"]["s"] = len(df)
        return df

    def _gerar_camada_ouro(self, dados):
        dados = self._gerar_camada_silver(dados)
        if dados.empty: return pd.DataFrame()
        
        dados['id_geo'] = dados.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)
        dados['nota'] = dados.apply(self._definir_peso, axis=1)
        
        X, y = dados[['LATITUDE', 'LONGITUDE']], dados['nota']
        if len(X) > 20:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, learning_rate=0.1)
            m.fit(xt, yt)
            p = m.predict(xv)
            
            mae = mean_absolute_error(yv, p)
            rmse = root_mean_squared_error(yv, p)
            r2 = r2_score(yv, p)
            
            self.auditoria['ia'] = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "c": round(max(0.0, 100.0 - (mae * 10)), 2)
            }
        
        res = dados.groupby(['id_geo']).agg({'nota': 'mean', 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['h3', 'risco', 'lat', 'lon']
        
        os.makedirs('datalake/camada_ouro_refinada/esquema_estrela', exist_ok=True)
        res.to_csv('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv', index=False)
        return res

    def _sync_deltasync(self, dados):
        if not self.banco or dados.empty: return
        ref_path = 'datalake/camada_prata_confiavel/assinatura_anterior.parquet'
        dados['hash'] = dados.apply(lambda r: hashlib.sha256(f"{r['h3']}{r['risco']}".encode()).hexdigest(), axis=1)
        
        if os.path.exists(ref_path):
            antigo = pd.read_parquet(ref_path)
            delta = dados[~dados['hash'].isin(antigo['hash'])].copy()
            self.auditoria['sync']['eco'] = (1 - (len(delta) / len(dados))) * 100
            self.auditoria['sync']['delta'] = len(delta)
        else:
            delta = dados
            self.auditoria['sync']['eco'] = 0.0
            self.auditoria['sync']['delta'] = len(dados)

        if not delta.empty:
            lote = self.banco.batch()
            for i, r in delta.iterrows():
                doc = self.banco.collection('malha_risco').document(r['h3'])
                lote.set(doc, {'r': round(r['risco'], 2), 'l': [r['lat'], r['lon']], 'u': firestore.SERVER_TIMESTAMP}, merge=True)
                if (i + 1) % 450 == 0:
                    lote.commit(); lote = self.banco.batch()
            lote.commit()
        
        dados.to_parquet(ref_path, index=False)
        self.auditoria['v']['g'] = len(dados)
        self.auditoria['sync']['total'] = len(dados)

    def _notificar(self, ok, msg=None):
        url = self.tokens["sucesso"] if ok else self.tokens["erro"]
        if not url: return
        
        embed = {
            "title": "✅ RELATÓRIO DE INTEGRIDADE: MOTOR SAFE-DRIVER" if ok else "🚨 FALHA NO PROCESSAMENTO",
            "color": 3066993 if ok else 15158332,
            "fields": [
                {"name": "📦 VOLUMETRIA", "value": f"Bronze: {self.auditoria['v']['b']}\nSilver: {self.auditoria['v']['s']}\nGold: {self.auditoria['v']['g']}", "inline": False},
                {"name": "🧠 MÉTRICAS DE IA", "value": f"**Confiança:** {self.auditoria['ia']['c']}%\n**MAE:** {self.auditoria['ia']['mae']}\n**RMSE:** {self.auditoria['ia']['rmse']}\n**R²:** {self.auditoria['ia']['r2']}", "inline": True},
                {"name": "☁️ SINCRONIZAÇÃO NUVEM", "value": f"**Delta:** {self.auditoria['sync']['delta']}\n**Economia:** {self.auditoria['sync']['eco']:.2f}%\n**Total Cloud:** {self.auditoria['sync']['total']}", "inline": True}
            ],
            "footer": {"text": f"Execução Finalizada em: {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
        }
        if msg: embed["description"] = f"Detalhes: `{msg}`"
        requests.post(url, json={"embeds": [embed]})

    def processar(self):
        try:
            mestre = pd.DataFrame()
            for ano in range(2022, datetime.now().year + 1):
                path = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
                if os.path.exists(path):
                    mestre = pd.concat([mestre, pd.read_parquet(path)])
            
            if not mestre.empty:
                self.auditoria['v']['b'] = len(mestre)
                ouro = self._gerar_camada_ouro(mestre)
                self._sync_deltasync(ouro)
                self._notificar(True)
        except Exception as e:
            self._notificar(False, str(e))

if __name__ == "__main__":
    MotorSeguranca().processar()
