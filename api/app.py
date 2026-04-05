import os
import hashlib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API Profissional")
CHAVE_API = os.environ.get("API_KEY", "fatec_safe_2026")
auth_header = APIKeyHeader(name="X-API-KEY")

def verificar_integridade():
    csv = Path("datalake/ouro/base_final_looker.csv")
    sha = Path("datalake/ouro/assinatura.sha256")
    if not csv.exists() or not sha.exists(): return False
    calculado = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    with open(sha, "r") as f:
        return calculado == f.read().strip()

@app.get("/v1/risco/{perfil}/{h3_index}")
def consultar_risco(perfil: str, h3_index: str, api_key: str = Security(auth_header)):
    if api_key != CHAVE_API: raise HTTPException(status_code=403, detail="Chave inválida.")
    if not verificar_integridade(): raise HTTPException(status_code=500, detail="Base de dados corrompida.")
    
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    resultado = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
    
    if resultado.empty: raise HTTPException(status_code=404, detail="Região ou Perfil sem dados.")
    
    pior_fator = "Localização" if abs(resultado['influencia_lat'].iloc[0]) > abs(resultado['influencia_periodo_idx'].iloc[0]) else "Horário"
    
    return {
        "h3": h3_index,
        "perfil": perfil,
        "score": round(float(resultado['score_predito'].mean()), 2),
        "fator_dominante": pior_fator
    }
