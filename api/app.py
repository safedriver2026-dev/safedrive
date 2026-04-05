import os
import hashlib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API", version="1.0.0")
API_KEY = os.environ.get("API_KEY", "fatec_2026_seguro")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

def validar_base():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    if not csv.exists() or not sha.exists(): return False
    with open(csv, "rb") as f:
        calculado = hashlib.sha256(f.read()).hexdigest()
    with open(sha, "r") as f:
        original = f.read().strip()
    return calculado == original

@app.get("/api/v1/risco")
def obter_risco(h3_index: str, chave: str = Security(api_key_header)):
    if chave != API_KEY: raise HTTPException(status_code=403, detail="Chave Inválida")
    if not validar_base(): raise HTTPException(status_code=500, detail="Integridade Comprometida")
    
    df = pd.read_csv("datalake/ouro/base_looker.csv")
    resultado = df[df['h3_index'] == h3_index]
    if resultado.empty: raise HTTPException(status_code=404, detail="H3 inexistente")
    
    return {"h3": h3_index, "score": round(float(resultado['score_risco'].values[0]), 4)}
