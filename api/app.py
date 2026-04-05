import os
import hashlib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API V20 Industrial")
CHAVE_PROD = os.environ.get("API_KEY", "fatec_safe_2026_industrial")
auth_header = APIKeyHeader(name="X-API-KEY")

def verificar_auditagem():
    manifesto_path = Path("datalake/auditoria/manifesto.json")
    ouro_path = Path("datalake/ouro/base_final_looker.csv")
    
    if not manifesto_path.exists() or not ouro_path.exists(): return False
    
    with open(manifesto_path, "r") as f: manifesto = json.load(f)
    
    sha256 = hashlib.sha256()
    with open(ouro_path, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""): sha256.update(bloco)
    
    return sha256.hexdigest() == manifesto.get("hash_ouro")

@app.get("/v1/risco/{perfil}/{h3_index}")
def consultar_risco(perfil: str, h3_index: str, api_key: str = Security(auth_header)):
    if api_key != CHAVE_PROD: raise HTTPException(status_code=403)
    if not verificar_auditagem(): raise HTTPException(status_code=500, detail="Trilha de Auditoria Violada")
    
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    dado = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
    
    if dado.empty: raise HTTPException(status_code=404)
    
    pior_fator = "Perfil" if abs(dado['influencia_perfil_idx'].mean()) > abs(dado['influencia_periodo_idx'].mean()) else "Horário"
    
    return {
        "h3": h3_index,
        "perfil": perfil,
        "score": round(float(dado['score_risco'].mean()), 2),
        "fator_dominante": pior_fator
    }
