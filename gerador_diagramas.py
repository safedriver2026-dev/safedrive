import base64, zlib, requests
from pathlib import Path

def produzir_diagrama(mermaid, nome):
    cod = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{cod}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/diag_{nome}.png", "wb") as f: f.write(res.content)

produzir_diagrama("""
graph TD
    A[Gatilho: GitHub Action] --> B[Motor V22: Carregar Manifesto]
    B --> C[Loop de Anos: 2022-2026]
    C --> D{DeltaSync: Tamanho Mudou?}
    D -- Sim --> E[Download Camuflado Chrome]
    D -- Não --> F[Carregar Cache Bronze Parquet]
    E --> G[Deduplicacao por Chave Primaria Composta]
    G --> H[Ensemble IA: LGBM + CatB + KNN]
    H --> I[Explicabilidade SHAP para Looker]
    I --> J[Monitoramento: Webhooks Sucesso/Erro]
    J --> K[Sincronizar Repositorio e Auditoria]
""", "funcionamento_automacao")

produzir_diagrama("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_risco
        float influencia_perfil
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "h3"
    DIM_PERFIL ||--o{ FATO_RISCO : "idx"
    DIM_TEMPO ||--o{ FATO_RISCO : "idx"
""", "modelo_dados_estrela")

produzir_diagrama("""
graph TD
    API[FastAPI Gateway] -->|Chave X-API-KEY| S{Validar Integridade}
    S -->|SHA256| M[Manifesto Auditoria]
    S -- OK --> D[Consulta Camada Ouro CSV]
    D --> J[Retorno JSON: Score + Fator Dominante]
    J --> L[Looker Studio BI Dashboard]
""", "fluxo_api_integracao")
