import os, hashlib, json, pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API")
CHAVE = os.environ.get("API_KEY", "fatec_safedriver_acesso")
seguranca = APIKeyHeader(name="X-API-KEY")

@app.get("/consultar/{perfil}/{h3_index}")
async def buscar(perfil: str, h3_index: str, api_key: str = Security(seguranca)):
    if api_key != CHAVE: raise HTTPException(status_code=403)
    
    df = pd.read_csv("datalake/ouro/predicao_risco_mapa.csv")
    busca = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
    
    if busca.empty: raise HTTPException(status_code=404)
    
    return {
        "hex": h3_index,
        "risco": round(float(busca['score_risco'].iloc[0]), 2),
        "integridade": "Auditado"
    }
