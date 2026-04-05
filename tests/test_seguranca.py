import os, hashlib, json, pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API V21 Industrial")
CHAVE = os.environ.get("API_KEY", "fatec_safe_2026")
auth = APIKeyHeader(name="X-API-KEY")

def verificar_integridade():
    man, out = Path("datalake/auditoria/manifesto.json"), Path("datalake/ouro/base_final_looker.csv")
    if not man.exists() or not out.exists(): return False
    with open(man, "r") as f: d = json.load(f)
    hash_calc = hashlib.sha256(open(out, "rb").read()).hexdigest()
    return hash_calc == d.get("hash_ouro")

@app.get("/v1/risco/{perfil}/{h3}")
def consultar(perfil: str, h3: str, api_key: str = Security(auth)):
    if api_key != CHAVE: raise HTTPException(status_code=403)
    if not verificar_integridade(): raise HTTPException(status_code=500)
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    r = df[(df['h3_index'] == h3) & (df['perfil'].str.lower() == perfil.lower())]
    if r.empty: raise HTTPException(status_code=404)
    return {
        "h3": h3, 
        "score": round(float(r['score_risco'].mean()), 2),
        "causa": "Localização" if abs(r['influencia_lat'].mean()) > abs(r['influencia_periodo_idx'].mean()) else "Horário"
    }
