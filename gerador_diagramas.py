import base64, zlib, requests
from pathlib import Path

def gerar_imagem(mermaid, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome}.png", "wb") as f: f.write(res.content)


gerar_imagem("""
graph TD
    subgraph Auditoria
        M[Manifesto JSON]
    end
    A[Gatilho Actions] --> B[DeltaSync Bronze]
    B -->|Hash| M
    B --> C[Processamento Prata]
    C -->|Hash| M
    C --> D[Treino IA Ouro]
    D -->|Hash| M
    D --> E[Relatórios Discord]
""", "automacao_auditoria")


gerar_imagem("""
erDiagram
    FATO_RISCO {
        string h3_index
        float score_predito
        float shap_perfil
        float shap_periodo
    }
    DIM_GEOGRAFIA ||--o{ FATO_RISCO : "lat/lon"
    DIM_PERFIL ||--o{ FATO_RISCO : "perfil_idx"
    DIM_TEMPO ||--o{ FATO_RISCO : "periodo_idx"
""", "modelo_estrela")


gerar_imagem("""
graph LR
    API[FastAPI] -->|Auth| GAS[Google Apps Script]
    GAS -->|Append| Sheets[Google Sheets]
    Sheets -->|Sync| Looker[Looker Studio]
""", "integracao_looker")
