import requests, os, sys

def download_file(url, nome_local):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    print(f"📡 A tentar descarregar de: {url}")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            if r.status_code == 200:
                with open(nome_local, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
                return True
    except:
        return False
    return False

# 1. Tenta São Paulo primeiro
url_sp = "https://download.geofabrik.de/south-america/brazil/sao-paulo-latest-free.shp.zip"
if download_file(url_sp, "dados_osm.zip"):
    print("✅ Sucesso: Base de São Paulo obtida.")
    sys.exit(0)

# 2. Se falhar, tenta Sudeste
print("⚠️ Falha ao obter base de SP. A tentar região Sudeste...")
url_sudeste = "https://download.geofabrik.de/south-america/brazil/sudeste-latest-free.shp.zip"
if download_file(url_sudeste, "dados_osm.zip"):
    print("✅ Sucesso: Base do Sudeste obtida como fallback.")
    sys.exit(0)

print("❌ ERRO CRÍTICO: Ambas as fontes falharam.")
sys.exit(1)
