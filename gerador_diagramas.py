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
    A[Camada Bronze Parquet] --> B[Extração: data_ocorrencia_bo]
    B --> C{Feature Engineering}
    C -->|holidays.Brazil| D[is_feriado]
    C -->|dt.day in 5,6,7,20,21| E[is_pagamento]
    C -->|dt.dayofweek >= 5| F[is_fim_semana]
    D & E & F --> G[Ensemble: XGBoost/CatB/KNN]
    G --> H[SHAP Values: Qual impacto do Feriado?]
    H --> I[Camada Ouro: base_final_looker.csv]
""", "arquitetura_feature_engineering")

salvar_png("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_risco
        int is_feriado
        int is_pagamento
        float shap_influencia_pagamento
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "h3"
    DIM_PERFIL ||--o{ FATO_RISCO : "idx"
    DIM_CALENDARIO ||--o{ FATO_RISCO : "sazonalidade"
""", "modelo_dados_comportamental")
