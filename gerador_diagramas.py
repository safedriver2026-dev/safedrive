import base64, zlib, requests
from pathlib import Path

def criar_diagrama(codigo_mermaid, nome_arquivo):
    """Gera uma imagem PNG a partir do código de diagrama"""
    try:
        conteudo = base64.urlsafe_b64encode(zlib.compress(codigo_mermaid.encode('utf-8'), 9)).decode('ascii')
        url = f"https://kroki.io/mermaid/png/{conteudo}"
        res = requests.get(url)
        if res.status_code == 200:
            Path("documentacao").mkdir(exist_ok=True)
            with open(f"documentacao/{nome_arquivo}.png", "wb") as f:
                f.write(res.content)
            print(f"Diagrama {nome_arquivo} gerado com sucesso.")
    except:
        print(f"Falha ao gerar diagrama {nome_arquivo}.")

# Diagrama de Fluxo de Dados (TCC)
criar_diagrama("""
graph TD
    A[Base Oficial SSP-SP] -->|Download Seguro| B(Motor de Dados)
    B --> C{Validação NUM_BO}
    C -->|Dados Limpos| D[Camada de Cache Parquet]
    D --> E[Inteligência Artificial Ensemble]
    E --> F[Base Predição - Looker]
    C --> G[Base Detalhes - Drilldown]
    F & G --> H[Selo Digital de Auditoria]
    H --> I[Acesso via API e BI]
""", "fluxo_projeto_safedriver")

# Diagrama do Esquema Estrela (Storytelling)
criar_diagrama("""
erDiagram
    PREDICAO_RISCO {
        string h3_index
        float score_risco
        float influencia_pagamento
    }
    CRIMES_DETALHADOS {
        string num_bo
        string h3_index
        string tipo_crime
        date data_real
    }
    PREDICAO_RISCO ||--o{ CRIMES_DETALHADOS : "União por Hexágono H3"
""", "esquema_estrela_dados")
