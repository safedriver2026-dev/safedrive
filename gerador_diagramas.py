import base64, zlib, requests
from pathlib import Path

def produzir_imagem(mermaid, nome):
    cod = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{cod}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/diag_{nome}.png", "wb") as f: f.write(res.content)

# FUNCIONAMENTO DA AUTOMAÇÃO MLOPS
salvar_diagrama("""
graph TD
    A[Início: GitHub Action] --> B[Criação de Pastas Datalake]
    B --> C[Leitura de manifesto.json]
    C --> D[Loop: Download Camuflado SSP]
    D --> E{DeltaSync: Tamanho Mudou?}
    E -- Sim --> F[Download Incremental XLSX]
    E -- Não --> G[Carregar Parquet Bronze Local]
    F --> H[Deduplicação por PK Composta]
    H --> I[Treino Ensemble: LGBM + CatB + KNN]
    I --> J[Cálculo de SHAP para Looker]
    J --> K[Gravação Camada Ouro e Manifesto]
    K --> L[Git Push: Auditoria e Dados]
""", "automacao_detalhada")

# MODELO DE DADOS E STAR SCHEMA

salvar_diagrama("""
graph LR
    subgraph Datalake_Bronze
        A[Arquivos Parquet Anuais]
    end
    subgraph Transformacao_Deduplicacao
        A --> B[PK: NUM_BO + ANO_BO + MUNICIPIO]
    end
    subgraph Datalake_Ouro
        B --> C(FATO_RISCO)
        C --> D[DIM_GEOGRAFIA: H3 Resolution 9]
        C --> E[DIM_PERFIL: Motorista/Pedestre/Ciclista]
        C --> F[DIM_TEMPO: Madrugada/Manhã/Tarde/Noite]
    end
""", "modelo_dados_estrela")

# FUNCIONAMENTO DA API E INTEGRAÇÃO
salvar_diagrama("""
graph TD
    U[Usuário/Dashboard] -->|Chave API| API[FastAPI Gateway]
    API -->|Validação| S{Integridade SHA256}
    S -- OK --> D[Consulta Ouro CSV]
    D --> J[Retorno JSON: Score + Causa Dominante]
    API -->|Trigger| GAS[Google Apps Script]
    GAS -->|AppendRow| Looker[Looker Studio BI]
""", "api_e_looker")
