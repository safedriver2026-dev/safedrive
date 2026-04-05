import base64, zlib, requests
from pathlib import Path

def produzir_diagrama(mermaid_texto, nome_imagem):
    url_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid_texto.encode('utf-8'), 9)).decode('ascii')
    res_kroki = requests.get(f"https://kroki.io/mermaid/png/{url_kroki}")
    if res_kroki.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome_imagem}.png", "wb") as f_img:
            f_img.write(res_kroki.content)

produzir_diagrama("""
graph TD
    subgraph Infraestrutura_ColdStart
        A[GitHub Runner] --> B[Criar Pastas Datalake]
        B --> C[Loop de Download SSP-SP]
    end
    subgraph Core_Processamento
        C --> D[Auto-Sheet Scan]
        D --> E[Auditoria Hash SHA256]
        E --> F[Ensemble: LGBM + CatBoost + KNN]
        F --> G[Exportacao SHAP Data]
    end
    subgraph Monitoramento
        G --> H[Manifesto JSON]
        H --> I[Relatorios Discord]
    end
""", "automacao_industrial")

produzir_diagrama("""
erDiagram
    TABELA_FATO_RISCO {
        string h3_index
        float score_final
        float influencia_perfil
        float influencia_periodo
    }
    DIM_LOCALIZACAO ||--o{ TABELA_FATO_RISCO : "h3"
    DIM_PERFIL ||--o{ TABELA_FATO_RISCO : "cod"
    DIM_TEMPO ||--o{ TABELA_FATO_RISCO : "cod"
""", "modelo_estrela")
