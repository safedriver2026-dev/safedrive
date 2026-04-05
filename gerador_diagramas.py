import base64, zlib, requests
from pathlib import Path

def extrair_diagrama(mermaid, nome):
    codificado = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{codificado}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome}.png", "wb") as f: f.write(res.content)

extrair_diagrama("""
graph TD
    A[GitHub Actions] --> B[Motor DeltaSync]
    B --> C{Auditoria Hash}
    C --> D[Ensemble: LGBM + CatBoost]
    D --> E[Export SHAP Data]
    E --> F[Manifesto JSON]
    F --> G[Notificacao Discord]
""", "automacao_auditavel")

extrair_diagrama("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_risco
        float influencia_perfil
        float influencia_horario
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "lat_lon"
    DIM_PERFIL ||--o{ FATO_RISCO : "idx"
    DIM_TEMPO ||--o{ FATO_RISCO : "idx"
""", "modelo_estrela")

extrair_diagrama("""
graph LR
    API[FastAPI] -->|Auth| GAS[Google Apps Script]
    GAS -->|Fetch| Sheet[Google Sheets]
    Sheet -->|Conector| Looker[Looker Studio]
""", "integracao_looker")
