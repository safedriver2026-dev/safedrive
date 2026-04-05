import os, hashlib, json, pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver Industrial API V17")
CHAVE_AUTENTICACAO = os.environ.get("API_KEY", "fatec_safe_2026_prod")
provedor_seguranca = APIKeyHeader(name="X-API-KEY")

def auditoria_integridade():
    path_manifesto = Path("datalake/auditoria/manifesto.json")
    path_ouro = Path("datalake/ouro/base_inteligencia.csv")
    if not path_manifesto.exists() or not path_ouro.exists(): return False
    
    with open(path_manifesto, "r") as f_audit: manifesto = json.load(f_audit)
    
    sha256_api = hashlib.sha256()
    with open(path_ouro, "rb") as f_bin:
        for bloco in iter(lambda: f_bin.read(4096), b""): sha256_api.update(bloco)
    
    return sha256_api.hexdigest() == manifesto.get("camada_ouro")

@app.get("/v1/risco/{perfil}/{h3_index}")
def consultar_risco_espacial(perfil: str, h3_index: str, api_key: str = Security(provedor_seguranca)):
    if api_key != CHAVE_AUTENTICACAO: raise HTTPException(status_code=403)
    if not auditoria_integridade(): raise HTTPException(status_code=500, detail="Audit Trail Violada")
    
    df_ouro = pd.read_csv("datalake/ouro/base_inteligencia.csv")
    dado_filtrado = df_ouro[(df_ouro['h3_index'] == h3_index) & (df_ouro['perfil_usuario'].str.lower() == perfil.lower())]
    
    if dado_filtrado.empty: raise HTTPException(status_code=404)
    
    return {
        "h3": h3_index,
        "score": round(float(dado_filtrado['score_final'].mean()), 2),
        "kpi_causa": "Perfil do Usuario" if abs(dado_filtrado['influencia_perfil_cod'].mean()) > abs(dado_filtrado['influencia_periodo_cod'].mean()) else "Periodo do Dia"
    }
