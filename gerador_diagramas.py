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
    A[Início do Fluxo] --> B[Requests Session com Retry]
    B --> C[Loop Anual Links Diretos SSP]
    C --> D[Download Chunked Resiliente 1MB]
    D --> E{Verificação Criptográfica SHA256}
    E -- Hash Inalterado --> F[Abortar Processamento + Cache Parquet]
    E -- Hash Alterado --> G[Extração e Limpeza Otimizada]
    G --> H[Conversão Parquet /datalake/bronze/]
    F --> I[Deduplicação Composta]
    H --> I
    I --> J[Feature Engineering: Feriados/Pagamento]
    J --> K[Agrupamento H3 Geoespacial]
    K --> L[Ensemble IA e Cálculo SHAP]
    L --> M[Exportar base_final_looker.csv e Manifesto]
""", "arquitetura_safedriver")

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
""", "modelo_dados_safedriver")

salvar_png("""
graph TD
    API[FastAPI Gateway] -->|Validacao X-API-KEY| S{Integridade SHA256}
    S -->|Auditoria| M[manifesto.json]
    S -- Sucesso --> D[Consulta base_final_looker.csv]
    D --> J[JSON Payload: Fatores de Risco]
    J --> L[Integração Looker Studio]
""", "fluxo_api_safedriver")
