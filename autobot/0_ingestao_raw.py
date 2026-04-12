import os
import requests
import boto3
import pandas as pd
import traceback
from datetime import datetime
from comunicador import RoboComunicador

def executar_ingestao(robo: RoboComunicador):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

    ano = datetime.now().year
    url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    bucket = os.environ.get("R2_BUCKET_NAME")
    path_raw = f"safedriver/datalake/raw/ssp_{ano}_incremental.parquet"

    try:
        res_head = requests.head(url, timeout=15)
        tamanho_ssp = int(res_head.headers.get('Content-Length', 0))

        print(f"📥 A extrair dados da SSP: {url}")
        res_data = requests.get(url, timeout=120)

        if ano == 2026:
            xls = pd.ExcelFile(res_data.content)
            df_novo = pd.DataFrame()
            for sheet_name in xls.sheet_names:
                try:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df_sheet.empty:
                        df_novo = pd.concat([df_novo, df_sheet], ignore_index=True)
                except Exception:
                    pass
            if df_novo.empty:
                raise ValueError("Nenhum dado válido encontrado nas abas do arquivo SSP.")
        else:
            df_novo = pd.read_excel(res_data.content)

        try:
            s3.download_file(bucket, path_raw, "raw_local.parquet")
            df_atual = pd.read_parquet("raw_local.parquet")

            df_final = pd.concat([df_atual, df_novo]).drop_duplicates(
                subset=['NUM_BO', 'ANO_BO', 'NOME_DELEGACIA'], keep='first'
            )
            novos_registos = len(df_final) - len(df_atual)
        except Exception:
            df_final = df_novo
            novos_registos = len(df_final)

        df_final.to_parquet("raw_final.parquet", index=False)
        s3.upload_file("raw_final.parquet", bucket, path_raw)

        if os.path.exists("raw_local.parquet"):
            os.remove("raw_local.parquet")
        if os.path.exists("raw_final.parquet"):
            os.remove("raw_final.parquet")

        robo.enviar_relatorio_operacional("Sincronização com a SSP concluída.", 
                                   {"Novos Registos": novos_registos, 
                                    "Tamanho Fonte": f"{tamanho_ssp} bytes",
                                    "Camada": "Bronze (Raw)"})
    except Exception:
        robo.enviar_alerta_tecnico("Ingestão Raw Bronze", traceback.format_exc())
