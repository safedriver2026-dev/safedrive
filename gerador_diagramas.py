import base64, zlib, requests
from pathlib import Path

def publicar_png(mermaid_code, nome):
    payload = base64.urlsafe_b64encode(zlib.compress(mermaid_code.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{payload}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/{nome}.png", "wb") as f: f.write(res.content)

diag_automacao = """
graph TD
    A[Cron GitHub Actions] --> B[DeltaSync: Valida Tamanho]
    B --> C{Base Mudou?}
    C -- Sim --> D[Download SSP-SP]
    C -- Não --> E[Cache Bronze Parquet]
    D --> F[Recuperação Geo: OSM]
    E --> F
    F --> G[Ensemble: LGBM + Cat + KNN]
    G --> H[Assinatura SHA256]
"""

diag_dados = """
graph LR
    subgraph Ingestao
        A[XLSX] --> B[Bronze Parquet]
    end
    subgraph Inteligencia
        B --> C[Prata: Pesos Penais]
        C --> D[Ouro: Score IA]
    end
    subgraph Entrega
        D --> E[Looker Studio]
        D --> F[FastAPI Segura]
    end
"""

if __name__ == "__main__":
    publicar_png(diag_automacao, "fluxo_automacao")
    publicar_png(diag_dados, "arquitetura_dados")
