import base64, zlib, requests
from pathlib import Path

def gerar_png(mermaid, nome):
    data = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{data}")
    if res.status_code == 200:
        with open(f"documentacao/{nome}.png", "wb") as f: f.write(res.content)

Path("documentacao").mkdir(exist_ok=True)

arq = """
graph TD
    A[SSP-SP Raw] -->|Delta Sync| B(Bronze Parquet)
    B -->|Weighting Layer| C(Prata: Pesos Penais)
    C -->|H3 Hexagons| D[Ensemble IA: LGBM+CAT+KNN]
    D --> E[Looker: CSV Export]
    D --> F[SHAP Audit PNG]
"""
gerar_png(arq, "arquitetura_automacao")
