import base64, zlib, requests
from pathlib import Path

def salvar_diagrama(mermaid, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/diag_{nome}.png", "wb") as f: f.write(res.content)

# DIAGRAMA 1: FUNCIONAMENTO DA AUTOMAÇÃO (MLOPS & AUDITORIA)
salvar_diagrama("""
graph TD
    A[Gatilho: GitHub Actions] --> B[Setup Ambiente Python 3.10]
    B --> C[Instalar requirements.txt]
    C --> D[Motor V20: Carregar Manifesto.json]
    D --> E[Iniciar Loop de Anos: 2022 até Hoje]
    E --> F{DeltaSync: requests.HEAD}
    F -- Tamanho Igual --> G[Carregar Parquet Bronze Local]
    F -- Tamanho Diferente --> H[requests.GET camuflado Chrome]
    F -- Erro Rede --> G
    H --> I[Identificar Aba de Dados Reais]
    I --> J[Salvar Parquet Bronze]
    J --> K[Atualizar Manifesto: Hash e Tamanho]
    G --> L[Concatenar DataFrames]
    K --> L
    L --> M[Deduplicação por Chave Primária Composta]
    M --> N[Classificação Perfil e Severidade]
    N --> O[H3 Indexing Resolução 9]
    O --> P[Treinar Ensemble: LGBM + CatB + KNN]
    P --> Q[Calcular SHAP (IA Explicável)]
    Q --> R[Salvar Ouro: base_final_looker.csv]
    R --> S[Registrar Hash Ouro no Manifesto]
    S --> T[Gravar manifesto.json definitivo]
    T --> U[Post Discord Operacional e Executivo]
    U --> V[Pytest: Validar Trilha de Auditoria]
    V --> W[Git Commit & Push Ouro + Manifesto]
""", "automacao_detalhada")

# DIAGRAMA 2: MODELO DE DADOS (LAKEHOUSE & STAR SCHEMA)

salvar_diagrama("""
graph TD
    subgraph Datalake_Bronze
        A[Link SSP XLSX] --> B[bruto_2022.parquet]
        A --> C[bruto_2023.parquet]
        A --> D[bruto_2026.parquet Incremental]
    end
    subgraph Processamento_Memoria
        B & C & D --> E[Normalização Colunas]
        E --> F[Vetor Exposição: Pedestre/Motorista/Ciclista]
        F --> G[H3 Geo-Join]
    end
    subgraph Datalake_Ouro_StarSchema
        G --> H(TABELA_FATO_RISCO)
        H -->|h3_index| I[DIM_GEOGRAFIA]
        H -->|perfil_idx| J[DIM_PERFIL]
        H -->|periodo_idx| K[DIM_TEMPO]
        H --> L[SHAP_influencia_perfil]
        H --> M[SHAP_influencia_horario]
        H --> N[score_risco_ensemble]
    end
    subgraph Camada_Auditoria
        O[manifesto.json] -->|Valida Hash| B & C & D & H
    end
""", "dados_detalhado")

# DIAGRAMA 3: FUNCIONAMENTO DA API E INTEGRAÇÃO LOOKER
salvar_diagrama("""
graph TD
    U[Aplicativo Cliente/Frontend] -->|Request X-API-KEY| API[FastAPI Gateway]
    subgraph API_Seguranca
        API --> V{Validar Chave}
        V -- Inválida --> E1[403 Forbidden]
        V -- OK --> I{Verificar Integridade}
        I -->|Calcula SHA256 Ouro| M[datalake/auditoria/manifesto.json]
        I -- Hash Divergente --> E2[500 Integridade Violada]
    end
    subgraph API_Processamento
        I -- OK --> D[Acessar datalake/ouro/base_final_looker.csv]
        D --> F[Filtrar Perfil e H3]
        F --> J[JSON Payload: Score + Fator Dominante]
    end
    J --> U
    subgraph Integracao_BI
        API -->|JSON| GAS[Google Apps Script]
        GAS -->|AppendRow| Sheet[Google Sheets]
        Sheet -->|Conector Nativo| Looker[Looker Studio Dashboard]
    end
""", "api_integracao_detalhado")
