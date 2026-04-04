import base64
import zlib
import requests

def exportar_diagrama(codigo_mermaid, nome_arquivo):
    print(f"Gerando o diagrama: {nome_arquivo}...")
    
    # Compactação e codificação exigidas pela API da Kroki
    codificado = base64.urlsafe_b64encode(zlib.compress(codigo_mermaid.encode('utf-8'), 9)).decode('ascii')
    url = f"https://kroki.io/mermaid/png/{codificado}"
    
    resposta = requests.get(url)
    
    if resposta.status_code == 200:
        with open(nome_arquivo, 'wb') as f:
            f.write(resposta.content)
        print(f"✅ Arquivo gerado e salvo com sucesso: {nome_arquivo}")
    else:
        print(f"❌ Falha de comunicação com a API ao gerar: {resposta.status_code}")

# =====================================================================
# 1. DIAGRAMA DE ARQUITETURA SISTÊMICA (Data Lakehouse)
# =====================================================================
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
        
        BR -->|Filtragem| PR
        PR -->|Integração e Algoritmos| OU
    end

    subgraph Produtos_de_Dados [Interfaces de Saída]
        API[API em Tempo Real: FastAPI]
        MAP[Mapa Estratégico: Folium HTML]
        BI[Painel Gerencial: Looker Studio]
    end

    SSP -->|Carga Incremental| BR
    PR <-->|Recuperação de Coordenadas| OSM
    OU -->|Fornece JSON| API
    OU -->|Renderiza Visual| MAP
    OU -->|Alimenta Relatório| BI

    style Processamento_de_Dados fill:#f8f9fa,stroke:#ced4da,stroke-width:2px
    style OU fill:#d4edda,stroke:#28a745,stroke-width:2px
"""

# =====================================================================
# 2. DIAGRAMA DE MODELAGEM DIMENSIONAL (Esquema Estrela)
# =====================================================================
arq_banco = """
erDiagram
    FATO_OCORRENCIAS {
        string indice_h3 PK "Chave do Polígono"
        float pontuacao_risco "Calculado via Inteligência Artificial"
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
        string periodo_do_dia "Manhã, Tarde, Noite, Madrugada"
    }
    DIMENSAO_GEOGRAFIA {
        string indice_h3 PK
        string nome_bairro
        string nome_municipio
        string logradouro
    }
    DIMENSAO_NATUREZA {
        int chave_natureza PK
        string classificacao_crime
        string natureza_apurada
    }
    
    FATO_OCORRENCIAS }|--|| DIMENSAO_TEMPO : "ocorre no período"
    FATO_OCORRENCIAS }|--|| DIMENSAO_GEOGRAFIA : "localiza-se na região"
    FATO_OCORRENCIAS }|--|| DIMENSAO_NATUREZA : "classifica-se criminalmente como"
"""

# =====================================================================
# 3. DIAGRAMA DE SEQUÊNCIA DA API (Fluxo de Requisição)
# =====================================================================
arq_api = """
sequenceDiagram
    participant Usuario as Sistema Cliente (Aplicativo)
    participant Servidor as API Principal (SafeDriver)
    participant Memoria as Banco de Dados Integrado (Ouro)
    
    Usuario->>Servidor: Requisitar Análise de Risco (Latitude, Longitude)
    Servidor->>Servidor: Converter Coordenadas para Polígono H3
    Servidor->>Memoria: Consultar Pontuação de Risco do Polígono
    
    alt Possui Histórico Criminal
        Memoria-->>Servidor: Retorna Pontuação Calculada
        Servidor-->>Usuario: Resposta JSON {Pontuação, Status: ALERTA}
    else Não Possui Histórico
        Servidor-->>Usuario: Resposta JSON {Pontuação: 0.0, Status: SEGURO}
    end
"""

if __name__ == "__main__":
    exportar_diagrama(arq_sistemico, "diagrama_01_arquitetura_sistemica.png")
    exportar_diagrama(arq_banco, "diagrama_02_modelagem_estrela.png")
    exportar_diagrama(arq_api, "diagrama_03_fluxo_api.png")
