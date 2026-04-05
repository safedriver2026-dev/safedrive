import base64, zlib, requests, os
from pathlib import Path

def salvar_diagrama(mermaid, nome):
    data = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{data}")
    if res.status_code == 200:
        with open(f"documentacao/{nome}.png", "wb") as f: f.write(res.content)

Path("documentacao").mkdir(exist_ok=True)

arq_mermaid = """
graph TD
    A[SSP-SP Portal] -->|Bronze| B(Data Lakehouse Parquet)
    B -->|Prata| C{Refinamento & GPS}
    C -->|Ouro| D[Ensemble AI: LGBM+CAT+KNN]
    D --> E[Looker Studio CSV]
    D --> F[SHAP Auditoria PNG]
    D --> G[Notificacao Discord]
"""

dados_mermaid = """
erDiagram
    CRIME_FATO ||--o{ TEMPO_DIM : registra
    CRIME_FATO ||--o{ GEO_DIM : localiza
    CRIME_FATO {
        string h3_index PK
        float score_risco
        int volume_crimes
    }
"""

salvar_diagrama(arq_mermaid, "arquitetura_automacao")
salvar_diagrama(dados_mermaid, "modelo_dados_star")
