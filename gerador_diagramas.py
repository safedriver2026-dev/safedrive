import base64, zlib, requests
from pathlib import Path

def gerar_diag(mermaid_code, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid_code.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome}.png", "wb") as f: f.write(res.content)

# 1. Diagrama de Automação (MLOps)
gerar_diag("""
graph TD
    A[Gatilho: GitHub Actions] --> B[Motor: DeltaSync]
    B --> C{Estado Local?}
    C -- Diferente --> D[Scan de Abas Excel]
    C -- Igual --> E[Cache Bronze Parquet]
    D --> F[Vetor de Exposição: Perfis]
    F --> G[Treino IA: LightGBM + SHAP]
    G --> H[Assinatura SHA256]
    H --> I[Relatórios Discord]
""", "automacao")

# 2. Modelo de Dados Estrela
gerar_diag("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_predito
        float influencia_perfil
        float influencia_horario
        float peso_severidade
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "lat/lon/h3"
    DIM_PERFIL ||--o{ FATO_RISCO : "perfil_idx"
    DIM_TEMPO ||--o{ FATO_RISCO : "periodo_idx"
""", "modelo_estrela")

# 3. Integração Looker
gerar_diag("""
graph LR
    API[FastAPI] -->|JSON| GAS[Google Apps Script]
    GAS -->|Append| Sheet[Google Sheets]
    Sheet -->|Conector Nativo| Looker[Looker Studio BI]
    Looker -->|Dashboard| User[Tomador de Decisão]
""", "integracao")
