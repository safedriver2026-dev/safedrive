name: SafeDriver_Core

on:
  schedule:
    - cron: '0 3 * * *'
  push:
    branches: [ main ]

jobs:
  execucao:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Sincronizar
        uses: actions/checkout@v4

      - name: Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Instalar
        run: |
          pip install pandas pyarrow openpyxl requests xgboost lightgbm catboost folium scikit-learn google-genai shap h3 pytest
          pip install -e .

      - name: Auditar
        run: pytest tests/test_contracts.py

      - name: Operar
        env:
          GEMINI_JSON: ${{ secrets.GEMINI_JSON }}
          DISCORD_SUCESSO: ${{ secrets.DISCORD_SUCESSO }}
          DISCORD_ERRO: ${{ secrets.DISCORD_ERRO }}
        run: python -c "from autobot.autobot_engine import MotorSafeDriver; MotorSafeDriver().gerenciar_ciclo_vida()"

      - name: Persistir
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "🤖 Sistema: Sincronização Delta e Malha H3 [skip ci]"
          file_pattern: "datalake/camada_ouro_refinada/* datalake/camada_prata_confiavel/* controle_delta.json"
