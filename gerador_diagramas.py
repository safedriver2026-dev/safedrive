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
    B --> C[Requests Session + Cookies]
    C --> D{DeltaSync: Content-Length Check}
    D -- Alterado --> E[Streaming Download 300MB+ Chunked]
    E --> F[Salvar Temp /datalake/raw/.xlsx]
    F --> G[Extração de Aba Correta e Parse]
    G --> H[Salvar Otimizado /datalake/bronze/.parquet]
    H --> I[Exclusão do XLSX Temp I/O Free]
    D -- Inalterado --> J[Carregar Parquet Direto]
    I --> K[Deduplicação Composta]
    J --> K
    K --> L[Treino Ensemble e SHAP]
    L --> M[Exportar base_final_looker.csv]
""", "arquitetura_bigdata_streaming")

salvar_png("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_risco
        float influencia_perfil
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "h3"
    DIM_PERFIL ||--o{ FATO_RISCO : "idx"
    DIM_TEMPO ||--o{ FATO_RISCO : "idx"
""", "modelo_dados")

salvar_png("""
graph TD
    API[FastAPI Gateway] -->|Validacao X-API-KEY| S{Integridade SHA256}
    S -->|Verifica| M[manifesto.json]
    S -- OK --> D[Consulta base_final_looker.csv]
    D --> J[JSON Payload: SHAP e Score]
    J --> L[Looker Studio BI]
""", "funcionamento_api")
