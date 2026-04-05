import base64, zlib, requests
from pathlib import Path

def extrair_png(mermaid, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome}.png", "wb") as f: f.write(res.content)

# Diagrama 1: Automação (MLOps)
diag_automacao = """
graph TD
    A[Gatilho: GitHub Actions] --> B[Motor: DeltaSync]
    B --> C{Base Atualizada?}
    C -- Sim --> D[Cache Bronze Parquet]
    C -- Não --> E[Download SSP-SP XLSX]
    D --> F[Classificação de Perfis]
    E --> F
    F --> G[IA Ensemble e SHAP]
    G --> H[Assinatura SHA256]
"""

# Diagrama 2: Dados (Medalhão)

diag_dados = """
graph LR
    subgraph Bronze
        A[XLSX Bruto] --> B[Parquet Original]
    end
    subgraph Prata
        B --> C[Limpeza e Pesos]
        C --> D[Geocodificação OSM]
    end
    subgraph Ouro
        D --> E[H3 Hexagonal]
        E --> F[Score IA Ensemble]
        F --> G[CSV e SHA256]
    end
"""

# Diagrama 3: API
diag_api = """
graph TD
    Client[Cliente/APP] -->|Key| API[FastAPI Endpoint]
    API -->|Validar| Hash{Assinatura Digital}
    Hash -- Match --> Data[Retorna Predição IA]
    Hash -- Erro --> Block[Bloqueia Acesso]
    Data --> Looker[Looker Studio]
"""

if __name__ == "__main__":
    extrair_png(diag_automacao, "automacao")
    extrair_png(diag_dados, "dados")
    extrair_png(diag_api, "api")
