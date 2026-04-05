import os
import hashlib
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path
import pandas as pd

app = FastAPI(title="SafeDriver API")
CHAVE_MESTRA = os.environ.get("API_KEY", "fatec_2026")
header_auth = APIKeyHeader(name="X-API-KEY")

def validar_integridade():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    if not csv.exists() or not sha.exists(): return False
    check = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    with open(sha, "r") as f:
        return check == f.read().strip()

@app.get("/v1/risco/{h3_index}")
def consultar_risco(h3_index: str, api_key: str = Security(header_auth)):
    if api_key != CHAVE_MESTRA: raise HTTPException(status_code=403)
    if not validar_integridade(): raise HTTPException(status_code=500, detail="Base Corrompida")
    
    df = pd.read_csv("datalake/ouro/base_looker.csv")
    dado = df[df['h3_index'] == h3_index]
    if dado.empty: raise HTTPException(status_code=404)
    
    return {"index": h3_index, "score": float(dado['score_risco'].values[0])}
