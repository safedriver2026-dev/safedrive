import os
import hashlib
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path
import pandas as pd

app = FastAPI(title="SafeDriver API")
TOKEN = os.environ.get("API_KEY", "fatec_2026")
auth_header = APIKeyHeader(name="X-API-KEY")

def verificar_assinatura():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    if not csv.exists() or not sha.exists(): return False
    check = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    with open(sha, "r") as f:
        return check == f.read().strip()

@app.get("/v1/risco/{h3_index}")
def consultar_risco(h3_index: str, api_key: str = Security(auth_header)):
    if api_key != TOKEN: raise HTTPException(status_code=403)
    if not verificar_assinatura(): raise HTTPException(status_code=500, detail="Base Comprometida")
    
    df = pd.read_csv("datalake/ouro/base_looker.csv")
    dado = df[df['h3_index'] == h3_index]
    if dado.empty: raise HTTPException(status_code=404)
    
    return {"h3_index": h3_index, "score": float(dado['score_risco'].values[0])}
