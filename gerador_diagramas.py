import base64, zlib, requests
from pathlib import Path

def gerar_png(mermaid, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome}.png", "wb") as f: f.write(res.content)

diag_automacao = """
graph TD
    A[Gatilho: GitHub Actions] --> B[Motor: DeltaSync]
    B --> C{Base Atualizada?}
    C -- Sim --> D[Carregar Cache Bronze]
    C -- Não --> E[Download SSP-SP XLSX]
    E --> D
    D --> F[Recuperação Geo: OSM]
    F --> G[Classificação de Perfis: Pedestre/Motorista/Ciclista]
    G --> H[Ensemble IA: LGBM + CatBoost]
    H --> I[Assinatura SHA256]
    I --> J[Publicação Docs e Push]
"""

diag_dados = """
graph LR
    subgraph Bronze
        A[XLSX Bruto] --> B[Parquet Original]
    end
    subgraph Prata
        B --> C[Pesos Penais]
        C --> D[Vetor de Exposição]
    end
    subgraph Ouro
        D --> E[H3 Hexagonal]
        E --> F[Score IA Ensemble]
        F --> G[CSV Looker e SHA256]
    end
"""

diag_api = """
graph TD
    Client[Cliente/APP] -->|X-API-KEY| API[FastAPI Endpoint]
    API -->|Validar| Hash{Assinatura Digital}
    Hash -- Match --> Data[Retorna Predição IA]
    Hash -- Erro --> Block[Bloqueia Acesso]
    Data --> BI[Looker Studio Dashboard]
"""

if __name__ == "__main__":
    gerar_png(diag_automacao, "automacao")
    gerar_png(diag_dados, "dados")
    gerar_png(diag_api, "api")
