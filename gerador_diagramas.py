import base64, zlib, requests
from pathlib import Path

def salvar_diagrama(mermaid, nome):
    data = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{data}")
    if res.status_code == 200:
        with open(f"documentacao/{nome}.png", "wb") as f: f.write(res.content)

Path("documentacao").mkdir(exist_ok=True)

arq = """
graph TD
    A[SSP-SP] -->|Bronze| B(Data Lakehouse)
    B -->|Prata| C{Limpeza/GPS}
    C -->|Ouro| D[Ensemble IA: LGBM+CAT+KNN]
    D --> E[API Real-time]
    D --> F[Looker Dashboard]
    D --> G[SHAP Audit]
"""

dados = """
erDiagram
    FATO_CRIMES ||--o{ DIM_TEMPO : ocorre
    FATO_CRIMES ||--o{ DIM_GEO : localiza
    FATO_CRIMES {
        string h3_index
        float score_risco
        int volume
    }
"""

salvar_diagrama(arq, "arquitetura_automacao")
salvar_diagrama(dados, "modelo_dados_star")
