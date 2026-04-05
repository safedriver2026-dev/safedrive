import base64, zlib, requests
from pathlib import Path

def salvar_png(mermaid, nome):
    cod = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{cod}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/diag_{nome}.png", "wb") as f: f.write(res.content)

salvar_png("""
graph TD
    A[Gatilho Actions] --> B[Selenium Headless Auth]
    B --> C[Requests Session + Cookies Forjados]
    C --> D[Loop Links Diretos 2022-2026]
    D --> E{Streaming Abortivo: Leitura de Headers}
    E -- Tamanho Inalterado --> F[Abortar Download + Cache Parquet]
    E -- Tamanho Alterado --> G[Download Chunked 300MB+]
    G --> H[Processamento e Conversao Parquet]
    F --> I[Deduplicacao Rigorosa]
    H --> I
    I --> J[Feature Engineering: Feriados/Pagamento]
    J --> K[Ensemble IA e Explicabilidade SHAP]
    K --> L[Exportar base_final_looker.csv e Manifesto]
""", "arquitetura_v28_direta")

salvar_png("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_risco
        float influencia_perfil
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "h3"
    DIM_PERFIL ||--o{ FATO_RISCO : "idx"
    DIM_CALENDARIO ||--o{ FATO_RISCO : "sazonalidade"
""", "modelo_dados")

salvar_png("""
graph TD
    API[FastAPI Gateway] -->|Validacao X-API-KEY| S{Integridade SHA256}
    S -->|Verifica| M[manifesto.json]
    S -- OK --> D[Consulta base_final_looker.csv]
    D --> J[JSON Payload: SHAP e Score]
    J --> L[Looker Studio BI]
""", "funcionamento_api")
