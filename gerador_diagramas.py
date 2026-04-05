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
    A[Início do Fluxo] --> B[Download Chunked Resiliente 1MB]
    B --> C{Verificação Criptográfica SHA256}
    C -- Hash Inalterado --> D[Abortar Processamento + Cache Parquet]
    C -- Hash Alterado --> E[Extração Otimizada - Descarte Vertical]
    E --> F[Deduplicação por NUM_BO]
    D --> F
    F --> G[Agrupamento H3 Geoespacial]
    G --> H[Train/Test Split 80/20]
    H --> I[Ensemble IA e Cálculo SHAP]
    I --> J[Alertas Discord: Operacional e Executivo]
    I --> K[Exportar base_final_looker.csv]
    F --> L[Exportar base_crimes_detalhados.csv]
""", "arquitetura_safedriver")

salvar_png("""
erDiagram
    BASE_FINAL_LOOKER {
        string h3_index
        float score_risco
        float influencia_is_pagamento
    }
    BASE_CRIMES_DETALHADOS {
        string num_bo
        string h3_index
        string crime_alvo
        date data_ocorrencia
    }
    BASE_FINAL_LOOKER ||--o{ BASE_CRIMES_DETALHADOS : "h3_index (Looker Join)"
""", "modelo_dados_safedriver")

salvar_png("""
graph TD
    API[FastAPI Gateway] -->|Validacao X-API-KEY| S{Integridade SHA256}
    S -->|Audita hash_ouro_ia| M[manifesto.json]
    S -- Sucesso --> D[Consulta base_final_looker.csv]
    D --> J[JSON Payload: Fatores de Risco]
    J --> L[Aplicações Externas]
""", "fluxo_api_safedriver")
