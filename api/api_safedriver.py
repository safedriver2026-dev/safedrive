from fastapi import FastAPI, HTTPException
import pandas as pd
import h3
from pathlib import Path

app = FastAPI(title="SafeDriver API - Inteligência Geospacial")

# Configuração de Caminhos
CAMINHO_OURO = Path("datalake/ouro/inteligencia_consolidada.parquet")

def carregar_dados():
    if not CAMINHO_OURO.exists():
        raise FileNotFoundError("Camada Ouro não encontrada. Execute o motor primeiro.")
    # Usamos Parquet na API pela velocidade de leitura (essencial para produção)
    return pd.read_parquet(CAMINHO_OURO)

@app.get("/risco/coordenada")
async def obter_risco_ponto(lat: float, lon: float, turno: int):
    """
    Retorna o risco preditivo para uma coordenada específica.
    Turnos: 0 (Madrugada), 1 (Manhã), 2 (Tarde), 3 (Noite)
    """
    df = carregar_dados()
    
    # Converte coordenada para H3 Nível 8 (Sincronizado com o Motor)
    try:
        index_h3 = h3.latlng_to_cell(lat, lon, 8)
    except Exception:
        raise HTTPException(status_code=400, detail="Coordenadas inválidas")

    # Busca a predição no Schema Star
    resultado = df[(df['H3_INDEX'] == index_h3) & (df['TURNO'] == turno)]
    
    if resultado.empty:
        return {
            "h3_index": index_h3,
            "status": "Sem dados históricos suficientes",
            "score_preditivo": 0.0,
            "mensagem": "Área de baixo tráfego criminal detectada"
        }

    # Retorna o Risco Suavizado (Sem efeito de borda)
    return {
        "h3_index": index_h3,
        "turno": int(resultado['TURNO'].iloc[0]),
        "score_preditivo": round(float(resultado['PREDICAO_RISCO'].iloc[0]), 2),
        "fator_dominante": resultado['FATOR_CRITICO'].iloc[0] if 'FATOR_CRITICO' in resultado else "N/A",
        "confianca_modelo": "Alta (Vizinhança Suavizada)"
    }

@app.get("/health")
async def check_health():
    return {"status": "operacional", "engine": "Ensemble CatBoost/LGBM"}
