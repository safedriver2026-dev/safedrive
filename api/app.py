import os
import hashlib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API")
API_KEY = os.environ.get("API_KEY", "fatec_safedriver_acesso")
header_auth = APIKeyHeader(name="X-API-KEY")

def validar_base():
    try:
        with open("datalake/auditoria/controle_integridade.json", "r") as f:
            esperado = json.load(f).get("selo_ia")
        sha = hashlib.sha256()
        with open("datalake/ouro/predicao_risco_mapa.csv", "rb") as f:
            for b in iter(lambda: f.read(4096), b""): sha.update(b)
        return sha.hexdigest() == esperado
    except: return False

@app.get("/v1/risco/{perfil}/{h3_index}")
async def obter_risco(perfil: str, h3_index: str, key: str = Security(header_auth)):
    if key != API_KEY: raise HTTPException(status_code=403, detail="Chave Invalida")
    if not validar_base(): raise HTTPException(status_code=500, detail="Base corrompida ou nao auditada")
    
    df = pd.read_csv("datalake/ouro/predicao_risco_mapa.csv")
    dado = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
    
    if dado.empty: raise HTTPException(status_code=404, detail="Regiao sem dados")
    
    return {
        "h3": h3_index,
        "risco": round(float(dado['score_risco'].mean()), 2),
        "confianca": "Auditado via SHA-256",
        "fator_principal": "Sazonalidade" if dado['inf_is_pagamento'].iloc[0] > 0 else "Localidade"
    }
