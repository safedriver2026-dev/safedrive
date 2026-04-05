import base64, zlib, requests
from pathlib import Path

def gerar_png(mermaid_text, nome):
    payload = base64.urlsafe_b64encode(zlib.compress(mermaid_text.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{payload}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/{nome}.png", "wb") as f: f.write(res.content)


diag_automacao = """
graph TD
    A[Gatilho: GitHub Actions] --> B[Motor: DeltaSync]
    B --> C{Base Atualizada?}
    C -- Não --> D[Download SSP-SP]
    C -- Sim --> E[Carregar Cache Bronze]
    D --> F[Processamento Prata]
    E --> F
    F --> G[Treinamento Ensemble IA]
    G --> H[Assinatura SHA256]
    H --> I[Publicação Docs e Git Push]
"""

# 2. Diagrama de Dados (Medalhão)

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


diag_seguranca = """
graph TD
    User[Cliente API] -->|X-API-KEY| API[FastAPI Endpoint]
    API -->|Validação| Hash{Assinatura SHA256}
    Hash -- Match --> Data[Retorna Dados Ouro]
    Hash -- Erro --> Block[Bloqueia Acesso]
    Data --> Looker[Looker Studio Dashboard]
"""

if __name__ == "__main__":
    gerar_png(diag_automacao, "arquitetura_automacao")
    gerar_png(diag_dados, "fluxo_dados")
    gerar_png(diag_seguranca, "seguranca_api")
