import base64, zlib, requests
from pathlib import Path

def salvar_diagrama(mermaid, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/diag_{nome}.png", "wb") as f: f.write(res.content)

# DIAGRAMA 1: AUTOMAÇÃO E DESCOBERTA DINÂMICA
salvar_diagrama("""
graph TD
    A[Gatilho: GitHub Actions] --> B[Motor V21: Carregar Auditoria]
    B --> C[Scraping: Buscar Links em /estatistica/consultas]
    C --> D{Link Encontrado?}
    D -- Sim --> E[DeltaSync: Comparar Content-Length]
    D -- Não --> F[Tentar Link Estático Padronizado]
    E -- Mudou --> G[Download Camuflado Chrome]
    E -- Igual --> H[Acessar Cache Bronze Parquet]
    G --> I[Sheet Scan: Localizar Dados Criminais]
    I --> J[Treino Ensemble: LGBM + CatB + KNN]
    J --> K[Assinatura SHA256 e Notificação Discord]
""", "automacao_dinamica")

# DIAGRAMA 2: MODELO DE DADOS LAKEHOUSE (STAR SCHEMA)

salvar_diagrama("""
graph LR
    subgraph Camada_Bronze
        A[Parquet Bruto 2022-2026]
    end
    subgraph Camada_Prata
        A --> B[Deduplicação por PK Composta]
        B --> C[Vetorização de Perfis]
    end
    subgraph Camada_Ouro
        C --> D(TABELA_FATO_RISCO)
        D --> E[DIM_GEOGRAFIA: H3]
        D --> F[DIM_PERFIL: Categórico]
        D --> G[DIM_TEMPO: Período]
        D --> H[VALORES_SHAP: Influência]
    end
""", "modelo_dados_estrela")

# DIAGRAMA 3: FUNCIONAMENTO DA API E LOOKER
salvar_diagrama("""
graph TD
    U[Usuário/App] -->|Chave X-API-KEY| API[FastAPI Gateway]
    API -->|Validação| S{Integridade SHA256}
    S -- OK --> D[Consulta Ouro CSV]
    D --> J[Retorno JSON: Score + Causa]
    API -->|Alimentação| GAS[Google Apps Script]
    GAS -->|Append| Sheet[Google Sheets]
    Sheet -->|Sync| Looker[Looker Studio BI]
""", "api_e_looker")
