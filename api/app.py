import os, hashlib, json, pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API V19")
TOKEN_PROD = os.environ.get("API_KEY", "fatec_safe_2026_br")
header_auth = APIKeyHeader(name="X-API-KEY")

def validar_assinatura():
    man, out = Path("datalake/auditoria/manifesto.json"), Path("datalake/ouro/base_final_bi.csv")
    if not man.exists() or not out.exists(): return False
    with open(man, "r") as f: d = json.load(f)
    return hashlib.sha256(open(out, "rb").read()).hexdigest() == d.get("hash_ouro")

@app.get("/v1/risco/{perfil}/{h3}")
def obter_risco(perfil: str, h3: str, api_key: str = Security(header_auth)):
    if api_key != TOKEN_PROD: raise HTTPException(status_code=403)
    if not validar_assinatura(): raise HTTPException(status_code=500)
    df = pd.read_csv("datalake/ouro/base_final_bi.csv")
    r = df[(df['h3_index'] == h3) & (df['perfil'].str.lower() == perfil.lower())]
    if r.empty: raise HTTPException(status_code=404)
    return {"h3": h3, "score": round(float(r['score_risco'].mean()), 2)}
