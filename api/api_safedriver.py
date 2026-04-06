from fastapi import FastAPI, HTTPException
import pandas as pd
import h3
from pathlib import Path

app = FastAPI(title="SafeDriver API")
DB_PATH = Path("datalake/ouro/fato_risco_comparativo.parquet")

@app.get("/risco")
async def obter_risco(lat: float, lon: float, turno: int):
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database error")
    df = pd.read_parquet(DB_PATH)
    idx_h3 = h3.latlng_to_cell(lat, lon, 8)
    res = df[(df['H3_INDEX'] == idx_h3) & (df['TURNO'] == turno)]
    if res.empty:
        return {"h3": idx_h3, "risco_real": 0.0, "risco_predito": 0.0, "desvio": 0.0}
    return {
        "h3": idx_h3,
        "risco_real": round(float(res['RISCO_REAL'].mean()), 2),
        "risco_predito": round(float(res['RISCO_PREDITO'].mean()), 2),
        "desvio_medio": round(float(res['DESVIO_ABS'].mean()), 2)
    }
