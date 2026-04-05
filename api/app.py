import os
import hashlib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pathlib import Path

app = FastAPI(title="SafeDriver: Portal de Consulta de Risco")

# Chave de segurança para evitar acessos não autorizados
CHAVE_SISTEMA = os.environ.get("API_KEY", "fatec_safedriver_acesso")
cabecalho_seguranca = APIKeyHeader(name="X-API-KEY")

def verificar_integridade_base():
    """Confere se a base de dados não foi alterada indevidamente"""
    caminho_controle = Path("datalake/auditoria/controle_integridade.json")
    caminho_dados = Path("datalake/ouro/predicao_risco_mapa.csv")
    
    if not caminho_controle.exists() or not caminho_dados.exists():
        return False
    
    with open(caminho_controle, "r") as f:
        controle = json.load(f)
    
    # Gera o hash atual do arquivo
    sha = hashlib.sha256()
    with open(caminho_dados, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""):
            sha.update(bloco)
    
    # Compara com o selo gravado pelo motor de dados
    return sha.hexdigest() == controle.get("selo_ia")

@app.get("/v1/risco/{perfil}/{h3_index}")
def consultar_risco_regiao(perfil: str, h3_index: str, api_key: str = Security(cabecalho_seguranca)):
    # 1. Valida a chave de acesso
    if api_key != CHAVE_SISTEMA:
        raise HTTPException(status_code=403, detail="Acesso não autorizado")
    
    # 2. Valida se os dados são confiáveis (Auditoria)
    if not verificar_integridade_base():
        raise HTTPException(status_code=500, detail="Erro de integridade: Base de dados violada ou desatualizada")
    
    # 3. Realiza a busca no arquivo de predição
    try:
        df = pd.read_csv("datalake/ouro/predicao_risco_mapa.csv")
        resultado = df[(df['h3_index'] == h3_index) & (df['perfil'].str.lower() == perfil.lower())]
        
        if resultado.empty:
            raise HTTPException(status_code=404, detail="Região ou Perfil não encontrado")
        
        # Retorna o score e o principal fator de influência
        return {
            "hexágono": h3_index,
            "perfil": perfil,
            "risco_estimado": round(float(resultado['score_risco'].iloc[0]), 2),
            "validacao_digital": "Verificada"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
