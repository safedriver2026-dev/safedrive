import os
import hashlib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API")
TOKEN = os.environ.get("API_KEY", "fatec_2026_v5")
auth_header = APIKeyHeader(name="X-API-KEY")

def validar_integridade():
    csv, sha = Path("datalake/ouro/base_looker.csv"), Path("datalake/ouro/base_looker.sha256")
    if not csv.exists() or not sha.exists(): return False
    with open(csv, "rb") as f:
        check = hashlib.sha256(f.read()).hexdigest()
    with open(sha, "r") as f:
        return check == f.read().strip()

@app.get("/v1/risco/{perfil}/{h3_index}")
def consultar_risco(perfil: str, h3_index: str, api_key: str = Security(auth_header)):
    if api_key != TOKEN: raise HTTPException(status_code=403)
    if not validar_integridade(): raise HTTPException(status_code=500, detail="Base Burlada")
    
    df = pd.read_csv("datalake/ouro/base_looker.csv")
    dado = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
    
    if dado.empty: raise HTTPException(status_code=404)
    return {"perfil": perfil, "h3": h3_index, "score": float(dado['score_risco'].mean())}
