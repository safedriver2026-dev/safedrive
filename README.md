# SafeDriver Autobot: Inteligência de Dados Aplicada à Segurança Pública

Projeto de Análise Geoespacial e Predição de Risco Criminal baseado em Arquitetura Lakehouse Efêmera, desenvolvido como Trabalho de Conclusão de Curso (TCC) em Análise e Desenvolvimento de Sistemas.

## Sobre o Projeto
O **SafeDriver** é um ecossistema de engenharia de dados desenvolvido para processar e analisar volumetrias massivas de registros criminais da Secretaria de Segurança Pública de São Paulo (SSP-SP). O sistema utiliza uma arquitetura Medalhão (Bronze, Prata e Ouro) para transformar dados brutos em inteligência preditiva geográfica, auxiliando na tomada de decisão estratégica e no policiamento preditivo.

## Arquitetura do Sistema
O projeto segue uma abordagem *serverless* e efêmera, orquestrada via **GitHub Actions**, garantindo escalabilidade e baixo custo operacional:

* **Camada Bronze (Raw):** Ingestão automatizada de arquivos XLSX brutos da SSP-SP e metadados do IBGE sem alterações estruturais.
* **Camada Prata (Trusted):** Saneamento, anonimização criptográfica e aplicação do algoritmo de **Funil de Resgate Geográfico 1-2-3**.
* **Camada Ouro (Refined):** Consolidação da Tabela Analítica de Base (ABT) enriquecida com dados socioeconômicos do Censo 2022.

## Tecnologias e Stack
* **Linguagem:** Python 3.11+
* **Processamento:** Polars (Alta Performance / Multithread) e DuckDB (Spatial Joins)
* **Armazenamento:** Cloudflare R2 (Object Storage S3-compatible) e Google BigQuery (ID: `safe-driver-fc3a9`)
* **Inteligência Artificial:** CatBoost (Regressão com Distribuição Tweedie) e SHAP Values (Inteligência Artificial Explicável - XAI)
* **Geoprocessamento:** Uber H3 (Indexação Hexagonal - Resolução 9)
* **Automação:** GitHub Actions (CI/CD Pipeline)

## Módulos de Destaque

### Funil de Resgate Geográfico 1-2-3
Algoritmo hierárquico desenvolvido para mitigar o "vazio geográfico" em boletins de ocorrência sem coordenadas originais. Ele atua em três frentes de redundância:
1. **Match por Rua Exata:** Cruzamento direto com a malha viária espacial do IBGE.
2. **Match por Prefixo:** Recuperação de vias com nomes abreviados, suprimidos ou com erros de digitação.
3. **Centroide de Bairro:** Vinculação residual à coordenada central do distrito/bairro informado.

### Equalização de Pontas
Técnica de padronização de códigos de setor censitário para garantir 100% de integridade relacional estrutural entre as bases criminais (eventos) e demográficas (contexto).

## Resultados e Deploy
Os resultados são consolidados em uma **One Big Table (OBT)** e disponibilizados no **Google BigQuery**, alimentando painéis analíticos no Looker Studio. O modelo preditivo projetou com sucesso mais de 1,8 milhão de predições futuras de risco para a malha urbana, processando um histórico de mais de 4,5 milhões de registros originais.

## Conformidade e Segurança (LGPD)
O projeto cumpre integralmente a **Lei Geral de Proteção de Dados (LGPD)**. Todo o pipeline utiliza criptografia `SHA-256` com *Pepper* para a anonimização irreversível de identificadores sensíveis (como o Número do B.O.), garantindo a privacidade dos cidadãos sem comprometer a acurácia estatística da análise.

## 👥 Autores
* Fernando Molina
* Lucas da Silva Pereira
* Renato Cesar Izidoro
* Guilherme Balbo Dias
* Gustavo Nascimento Alves

---
*Faculdade de Tecnologia (FATEC) - São Caetano do Sul (Antônio Russo)* *Orientador: Prof. Msc. Flávio Viotti*
