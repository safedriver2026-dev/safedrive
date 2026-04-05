import base64, zlib, requests
from pathlib import Path

def produzir_img(mermaid, nome):
    cod = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{cod}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/diag_{nome}.png", "wb") as f: f.write(res.content)

produzir_img("""
graph TD
    A[GitHub Actions] --> B[Motor DeltaSync V19]
    B --> C{Remote Check: Content-Length}
    C -- Igual --> D[Acessa Parquet Local]
    C -- Diferente --> E[Download Novo XLSX]
    E --> F[Deduplicacao por PK Composta]
    F --> G[Ensemble IA: LGBM+CatB+KNN]
    G --> H[Exportacao SHAP para Looker]
""", "deltasync_br")

produzir_img("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_risco
        float influencia_perfil
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "h3"
    DIM_PERFIL ||--o{ FATO_RISCO : "idx"
    DIM_TEMPO ||--o{ FATO_RISCO : "idx"
""", "estrela_final")
