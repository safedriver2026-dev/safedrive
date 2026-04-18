name: "02. Build: Malha Mestra Consolidada"

on:
  workflow_dispatch:

jobs:
  build-malha:
    runs-on: ubuntu-latest
    steps:
      - name: 📥 Checkout
        uses: actions/checkout@v3

      - name: 🐍 Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: 📦 Instalar Dependências
        run: pip install polars h3==3.7.6 requests duckdb boto3 pyarrow

      - name: 🌍 Download IBGE (Resiliente)
        run: |
          mkdir -p dados_ibge
          
          # Função para download com retry pesado (vence o erro 503)
          smart_download() {
            wget --retry-connrefused --waitretry=15 --read-timeout=30 --timeout=20 --tries=20 -O $1 $2 || exit 1
            sleep 10
          }

          echo "🚀 Baixando Malhas 2025..."
          smart_download mun.zip https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2025/UFs/SP/SP_Municipios_2025.zip
          smart_download rgi.zip https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2025/UFs/SP/SP_RG_Imediatas_2025.zip
          smart_download rint.zip https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2025/UFs/SP/SP_RG_Intermediarias_2025.zip
          
          echo "🚀 Baixando Faces de Logradouro 2022..."
          # Link corrigido sem o sufixo que causa 404
          smart_download faces.zip https://geoftp.ibge.gov.br/recortes_para_fins_estatisticos/malhas_territoriais/malhas_de_setores_censitarios/censo_2022/base_de_faces_de_logradouros_versao_2022/json/SP_faces_de_logradouros_2022_json.zip
          
          unzip -o mun.zip -d dados_ibge/municipios
          unzip -o rgi.zip -d dados_ibge/imediata
          unzip -o rint.zip -d dados_ibge/intermediaria
          unzip -o faces.zip -d dados_ibge/faces

      - name: 🚀 Executar Construção da Malha
        env:
          R2_ENDPOINT_URL: ${{ secrets.R2_ENDPOINT_URL }}
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          R2_BUCKET_NAME: ${{ secrets.R2_BUCKET_NAME }}
        run: python autobot/setup_malha_mestra.py
