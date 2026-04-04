import base64
import zlib
import requests

def exportar_diagrama(codigo_mermaid, nome_arquivo):
    codificado = base64.urlsafe_b64encode(zlib.compress(codigo_mermaid.encode('utf-8'), 9)).decode('ascii')
    url = f"https://kroki.io/mermaid/png/{codificado}"
    resposta = requests.get(url)
    if resposta.status_code == 200:
        with open(nome_arquivo, 'wb') as f:
            f.write(resposta.content)

arq_sistemico = """
graph LR
    subgraph Fontes_de_Dados [Fontes de Dados]
        SSP[Portal SSP-SP: Planilha Anual]
        OSM[OpenStreetMap: Geocodificação]
    end
    subgraph Processamento_de_Dados [Motor de Engenharia de Dados]
        direction TB
        BR[Camada Bronze: Ingestão de Dados Brutos]
        PR[Camada Prata: Tratamento e Limpeza]
        OU[Camada Ouro: Predição e Indexação H3]
        BR --> PR
        PR --> OU
    end
    subgraph Produtos_de_Dados [Interfaces de Saída]
        API[API em Tempo Real: FastAPI]
        MAP[Mapa Estratégico: Folium HTML]
        BI[Painel Gerencial: Looker Studio]
    end
    SSP --> BR
    PR <--> OSM
    OU --> API
    OU --> MAP
    OU --> BI
"""

arq_banco = """
erDiagram
    FATO_OCORRENCIAS {
        string indice_h3 PK
        float pontuacao_risco
        float latitude
        float longitude
        int chave_tempo FK
        int chave_natureza FK
    }
    DIMENSAO_TEMPO {
        int chave_tempo PK
        date data_ocorrencia
        int ano_registro
        int mes_registro
    }
    DIMENSAO_GEOGRAFIA {
        string indice_h3 PK
        string nome_bairro
        string nome_municipio
    }
    DIMENSAO_NATUREZA {
        int chave_natureza PK
        string classificacao_crime
    }
    FATO_OCORRENCIAS }|--|| DIMENSAO_TEMPO : ""
    FATO_OCORRENCIAS }|--|| DIMENSAO_GEOGRAFIA : ""
    FATO_OCORRENCIAS }|--|| DIMENSAO_NATUREZA : ""
"""

arq_api = """
sequenceDiagram
    participant Usuario
    participant Servidor
    participant Memoria
    Usuario->>Servidor: Requisitar Análise (Lat, Lon)
    Servidor->>Servidor: Converter para H3
    Servidor->>Memoria: Consultar Risco
    Memoria-->>Servidor: Retorna Pontuação
    Servidor-->>Usuario: JSON {Pontuação, Status}
"""

if __name__ == "__main__":
    exportar_diagrama(arq_sistemico, "diagrama_01_arquitetura_sistemica.png")
    exportar_diagrama(arq_banco, "diagrama_02_modelagem_estrela.png")
    exportar_diagrama(arq_api, "diagrama_03_fluxo_api.png")
