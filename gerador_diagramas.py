import base64
import zlib
import requests
import os

def exportar_diagrama(codigo_mermaid, nome_arquivo):
    # Compacta e codifica o código Mermaid para enviar à API open-source do Kroki
    codificado = base64.urlsafe_b64encode(zlib.compress(codigo_mermaid.encode('utf-8'), 9)).decode('ascii')
    url = f"https://kroki.io/mermaid/png/{codificado}"
    
    print(f"Processando {nome_arquivo}...")
    resposta = requests.get(url)
    
    if resposta.status_code == 200:
        with open(nome_arquivo, 'wb') as f:
            f.write(resposta.content)
        print(f"✅ Sucesso: {nome_arquivo} salvo no diretório.")
    else:
        print(f"❌ Erro ao gerar {nome_arquivo}: {resposta.status_code}")

# 1. Diagrama de Arquitetura (Data Lakehouse)
arq_pipeline = """
graph TD
    subgraph Fonte_Externa
        SSP[Site SSP SP - XLSX]
    end

    subgraph Data_Lake_Bronze
        BR[Camada Bronze: Bruto Parquet]
    end

    subgraph Data_Lake_Prata
        PR[Camada Prata: Dados Limpos]
        OSM[OpenStreetMap API]
        PR -->|Geocodificação| OSM
    end

    subgraph Data_Lake_Ouro
        OU[Camada Ouro: Analytics]
        ML[LightGBM: Modelo Preditivo]
        H3[Indexação H3 Res 9]
        OU --> ML
        OU --> H3
    end

    subgraph Entrega
        API[FastAPI: Risco Real-Time]
        HTML[Mapa Folium: Visualização]
        LK[Looker Studio: Dashboards]
    end

    SSP -->|Ingestão| BR
    BR -->|Limpeza| PR
    PR -->|Enriquecimento| OU
    OU --> API
    OU --> HTML
    OU --> LK
"""

# 2. Diagrama de Modelagem Dimensional (Star Schema)
arq_banco = """
erDiagram
    FATO_OCORRENCIAS {
        string h3_index PK
        float score_risco
        float latitude
        float longitude
        int fk_tempo FK
        int fk_natureza FK
    }
    DIM_TEMPO {
        int id_tempo PK
        date data
        int ano
        int mes
        string periodo_dia
    }
    DIM_NATUREZA {
        int id_natureza PK
        string rubrica
        string conduta
    }
    DIM_GEOGRAFIA {
        string h3_index PK
        string bairro
        string municipio
        string logradouro
    }

    FATO_OCORRENCIAS }|--|| DIM_TEMPO : "ocorre em"
    FATO_OCORRENCIAS }|--|| DIM_NATUREZA : "classificada como"
    FATO_OCORRENCIAS ||--|| DIM_GEOGRAFIA : "localizada em"
"""

# 3. Diagrama de Sequência da API
arq_api = """
sequenceDiagram
    participant App as Aplicativo Cliente
    participant API as FastAPI (SafeDriver)
    participant Cache as Memória Ouro
    
    App->>API: GET /risco/lat/lon
    API->>API: Converter Lat/Lon para H3 Index
    API->>Cache: Buscar Score do Polígono H3
    alt Existe no Histórico
        Cache-->>API: Retorna Score Preditivo
        API-->>App: JSON {Score, Status, H3}
    else Sem Histórico
        API-->>App: JSON {Score: 0.0, Status: SEGURO}
    end
"""

if __name__ == "__main__":
    exportar_diagrama(arq_pipeline, "diagrama_arquitetura.png")
    exportar_diagrama(arq_banco, "diagrama_star_schema.png")
    exportar_diagrama(arq_api, "diagrama_api_sequencia.png")
