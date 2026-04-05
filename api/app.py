import os, hashlib, json, pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver API V12")
TOKEN = os.environ.get("API_KEY", "fatec_safe_2026")
auth_header = APIKeyHeader(name="X-API-KEY")

def verificar_auditoria():
    manifesto_path = Path("datalake/auditoria/manifesto.json")
    if not manifesto_path.exists(): return False
    
    with open(manifesto_path, "r") as f:
        manifesto = json.load(f)
    
    # Verifica a camada Ouro como exemplo de integridade total
    caminho_ouro = Path("datalake/ouro/base_final.csv")
    if not caminho_ouro.exists(): return False
    
    hash_atual = hashlib.sha256(open(caminho_ouro, "rb").read()).hexdigest()
    return hash_atual == manifesto.get("ouro")

@app.get("/v1/risco/{perfil}/{h3_index}")
def consultar_risco(perfil: str, h3_index: str, api_key: str = Security(auth_header)):
    if api_key != TOKEN: raise HTTPException(status_code=403)
    if not verificar_auditoria(): raise HTTPException(status_code=500, detail="Trilha de Auditoria Violada.")
    
    df = pd.read_csv("datalake/ouro/base_final.csv")
    dado = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
    
    if dado.empty: raise HTTPException(status_code=404)
    return {
        "h3": h3_index,
        "risco": round(float(dado['score_predito'].mean()), 2),
        "explicacao": "Risco influenciado pelo Perfil" if abs(dado['shap_perfil_idx'].mean()) > abs(dado['shap_periodo_idx'].mean()) else "Risco influenciado pelo Horário"
    }
