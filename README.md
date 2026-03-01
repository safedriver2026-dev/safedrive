Inteligência Preditiva e Geoprocessamento para Mobilidade Urbana

Plataforma de Data Intelligence que automatiza o ciclo de extração, análise e predição de riscos criminais na capital paulista, utilizando dados oficiais da SSP-SP.

Objetivos Estratégicos
Automação (ZeroOps): Coleta e processamento mensal via GitHub Actions.

Predição: Geração de scores de risco (0-10) via algoritmo XGBoost.

Personalização: Modelagem de risco distinta para Motoristas, Motociclistas e Pedestres.

Arquitetura e Stack
Linguagem: Python 3.10 (Pandas, Scikit-Learn, PyGeohash).

IA: Modelo de regressão para análise de séries temporais e tendências.

Infraestrutura: GitHub Actions (CI/CD) e Google Firebase Firestore (NoSQL).

Consumo: API para App Android e Conector Web para Power BI.

Fluxo de Execução
ETL: Extração e limpeza de dados históricos da SSP-SP.

IA: Treinamento do modelo com foco em sazonalidade e geofencing (Geohash L6).

Deploy: Sincronização automática com Firebase e geração de base analítica (.csv/.parquet).

Monitoramento: Notificações de status via Webhook (Discord).
