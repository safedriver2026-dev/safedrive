import base64, zlib, requests
from pathlib import Path

def salvar_png(mermaid, nome):
    id_kroki = base64.urlsafe_b64encode(zlib.compress(mermaid.encode('utf-8'), 9)).decode('ascii')
    res = requests.get(f"https://kroki.io/mermaid/png/{id_kroki}")
    if res.status_code == 200:
        Path("documentacao").mkdir(exist_ok=True)
        with open(f"documentacao/arquitetura_{nome}.png", "wb") as f: f.write(res.content)


salvar_png("""
graph TD
    A[GitHub Actions] --> B[DeltaSync]
    B --> C[Auto-Sheet Discovery]
    C --> D[Vetor de Exposição]
    D --> E[Ensemble IA + SHAP]
    E --> F[Assinatura SHA256]
""", "automacao")


salvar_png("""
graph LR
    B[Bronze: Bruto] --> P[Prata: Perfis]
    P --> O[Ouro: Score IA]
    O --> L[Looker BI]
""", "dados")


salvar_png("""
graph TD
    U[App Cliente] --> API[FastAPI]
    API --> H{Valida SHA256}
    H -- OK --> D[Retorna Risco]
    H -- Erro --> B[Bloqueia]
""", "api")
