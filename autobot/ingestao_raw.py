import os
import requests
import boto3
import pandas as pd
import traceback
from datetime import datetime
from .comunicador import RoboComunicador

def executar_ingestao(robo: RoboComunicador, ano: int):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

    url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    bucket = os.environ.get("R2_BUCKET_NAME")
    path_raw = f"safedriver/datalake/raw/ssp_{ano}_incremental.parquet"

    try:
        res_head = requests.head(url, timeout=15)
        tamanho_ssp_fonte = int(res_head.headers.get('Content-Length', 0))

        if tamanho_ssp_fonte == 0:
            robo.enviar_relatorio_operacional(f"⚠️ Camada Bronze: Fonte SSP ({ano}) retornou tamanho 0. Ingestão pulada.", 
                                       {"URL Fonte": url, "Camada": "Bronze (Raw)"})
            return False

        base_existente_no_r2 = False
        tamanho_ssp_r2 = 0
        try:
            obj_info = s3.head_object(Bucket=bucket, Key=path_raw)
            tamanho_ssp_r2 = obj_info['ContentLength']
            base_existente_no_r2 = True
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NotFound':
                base_existente_no_r2 = False
            else:
                raise # Re-lançar outros erros do S3

        if base_existente_no_r2 and tamanho_ssp_fonte == tamanho_ssp_r2:
            robo.enviar_relatorio_operacional(f"✅ Camada Bronze: Fonte SSP ({ano}) não atualizada. Ingestão pulada.", 
                                       {"Tamanho Fonte": f"{tamanho_ssp_fonte} bytes",
                                        "Camada": "Bronze (Raw)"})
            return False

        res_data = requests.get(url, timeout=120)

        df_novo = pd.DataFrame()
        if ano >= 2022:
            xls = pd.ExcelFile(res_data.content)
            for sheet_name in xls.sheet_names:
                try:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df_sheet.empty:
                        df_novo = pd.concat([df_novo, df_sheet], ignore_index=True)
                except Exception as sheet_e:
                    robo.enviar_alerta_tecnico(f"Ingestão Raw (Bronze) - Erro ao ler aba '{sheet_name}' para o ano {ano}", str(sheet_e))
        else:
            df_novo = pd.read_excel(res_data.content)

        if df_novo.empty:
            robo.enviar_relatorio_operacional(f"⚠️ Camada Bronze: DataFrame da SSP ({ano}) vazio após leitura. Ingestão pulada.", 
                                       {"URL Fonte": url, "Camada": "Bronze (Raw)"})
            return False

        df_novo.columns = df_novo.columns.str.upper().str.strip().str.replace(' ', '_')

        novos_registros = len(df_novo)

        if base_existente_no_r2:
            s3.download_file(bucket, path_raw, "raw_local.parquet")
            df_existente = pd.read_parquet("raw_local.parquet")

            df_final = pd.concat([df_existente, df_novo]).drop_duplicates().reset_index(drop=True)
            novos_registros = len(df_final) - len(df_existente)
        else:
            df_final = df_novo

        df_final.to_parquet("raw_final.parquet", index=False)
        s3.upload_file("raw_final.parquet", bucket, path_raw)

        if os.path.exists("raw_local.parquet"):
            os.remove("raw_local.parquet")
        if os.path.exists("raw_final.parquet"):
            os.remove("raw_final.parquet")

        robo.enviar_relatorio_operacional(f"Sincronização com a SSP ({ano}) concluída.", 
                                   {"Novos Registros": novos_registros, 
                                    "Tamanho Fonte": f"{tamanho_ssp_fonte} bytes",
                                    "Camada": "Bronze (Raw)"})
        return True
    except Exception:
        robo.enviar_alerta_tecnico(f"Ingestão Raw (Bronze) - Ano {ano}", traceback.format_exc())
        return False
