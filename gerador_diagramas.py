import base64, zlib, requests
from pathlib import Path

def salvar_diagrama(mermaid_code, nome_arquivo):
    try:
        puro = base64.urlsafe_b64encode(zlib.compress(mermaid_code.encode('utf-8'), 9)).decode('ascii')
        url = f"https://kroki.io/mermaid/png/{puro}"
        res = requests.get(url)
        if res.status_code == 200:
            Path("documentacao").mkdir(exist_ok=True)
            with open(f"documentacao/{nome_arquivo}.png", "wb") as f:
                f.write(res.content)
            print(f"Sucesso: {nome_arquivo}.png")
    except Exception as e:
        print(f"Erro em {nome_arquivo}: {e}")

# 1. DIAGRAMA DE FUNCIONAMENTO DA AUTOMAÇÃO (PIPELINE MLOPS)
fluxo_automacao = """
graph LR
    subgraph Agendamento
        A[GitHub Actions] -->|Cron 09:00| B(Motor SafeDriver)
    end
    
    subgraph Ingestao_Resiliente
        B -->|Download com Retry| C[Raw XLSX]
        C -->|Hash Check| D{Mudou?}
        D -->|Nao| E[Usa Cache Bronze]
        D -->|Sim| F[Nova Bronze Parquet]
    end
    
    subgraph Inteligencia_Ensemble
        F --> G[Pre-Processamento]
        G --> H[LGBM + CatBoost + KNN]
        H --> I[Explicabilidade SHAP]
    end
    
    subgraph Entrega_Auditada
        I --> J[Ouro: CSV/Parquet]
        J --> K[Auditoria SHA-256]
        K --> L[Git Push / Discord]
    end
"""

# 2. DIAGRAMA DE DADOS (ARQUITETURA STAR SCHEMA)
diagrama_dados = """
erDiagram
    FATO_PREDICAO_RISCO {
        string h3_index PK
        float score_risco
        float inf_is_pagamento
        float inf_hora
        float inf_mes
        int perfil_cod
    }
    DIM_CRIMES_DETALHADOS {
        string num_bo PK
        string h3_index FK
        datetime data_real
        string perfil
        string crime_tipo
        float latitude
        float longitude
    }
    FATO_PREDICAO_RISCO ||--o{ DIM_CRIMES_DETALHADOS : "Filtro Geospacial (H3)"
"""

salvar_diagrama(fluxo_automacao, "funcionamento_automacao")
salvar_diagrama(diagrama_dados, "arquitetura_dados")
