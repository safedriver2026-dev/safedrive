import os
import requests
import boto3
import pandas as pd
import traceback
from datetime import datetime
from .comunicador import RoboComunicador

def executar_ingestao(robo: RoboComunicador):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

    ano = datetime.now().year
    url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    bucket = os.environ.get("R2_BUCKET_NAME")
    path_raw = f"safedriver/datalake/raw/ssp_{ano}_incremental.parquet"

    print(f"DEBUG INGESTÃO: Iniciando execução para o ano {ano}.")
    print(f"DEBUG INGESTÃO: URL da SSP: {url}")
    print(f"DEBUG INGESTÃO: Bucket R2: {bucket}")
    print(f"DEBUG INGESTÃO: Caminho Raw no R2: {path_raw}")

    try:
        res_head = requests.head(url, timeout=15)
        tamanho_ssp_fonte = int(res_head.headers.get('Content-Length', 0))
        print(f"DEBUG INGESTÃO: Tamanho da fonte SSP: {tamanho_ssp_fonte} bytes.")

        if tamanho_ssp_fonte == 0:
            print(f"⚠️ Camada Bronze: Fonte SSP ({url}) retornou tamanho 0. Não há dados para ingestão.")
            robo.enviar_relatorio_operacional("Camada Bronze: Fonte SSP retornou tamanho 0. Ingestão pulada.", 
                                       {"URL Fonte": url, "Camada": "Bronze (Raw)"})
            return False

        base_existente_no_r2 = False
        tamanho_ssp_r2 = 0
        try:
            obj_info = s3.head_object(Bucket=bucket, Key=path_raw)
            tamanho_ssp_r2 = obj_info['ContentLength']
            base_existente_no_r2 = True
            print(f"DEBUG INGESTÃO: Arquivo {path_raw} ENCONTRADO no R2. Tamanho: {tamanho_ssp_r2} bytes.")
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NotFound':
                base_existente_no_r2 = False
                print(f"DEBUG INGESTÃO: Arquivo {path_raw} NÃO encontrado no R2 (NotFound).")
            else:
                print(f"DEBUG INGESTÃO: Erro inesperado ao verificar {path_raw} no R2: {e}")
                raise
        except Exception as e:
            print(f"DEBUG INGESTÃO: Erro GENÉRICO ao verificar {path_raw} no R2: {e}")
            raise

        if base_existente_no_r2 and tamanho_ssp_fonte == tamanho_ssp_r2:
            print(f"DEBUG INGESTÃO: Condição de pular ATENDIDA: base_existente_no_r2={base_existente_no_r2} e tamanho_ssp_fonte={tamanho_ssp_fonte} == tamanho_ssp_r2={tamanho_ssp_r2}.")
            print(f"✅ Camada Bronze: Fonte SSP não atualizada. Tamanho {tamanho_ssp_fonte} bytes. Pulando ingestão.")
            robo.enviar_relatorio_operacional("Camada Bronze: Fonte SSP não atualizada. Ingestão pulada.", 
                                       {"Tamanho Fonte": f"{tamanho_ssp_fonte} bytes",
                                        "Camada": "Bronze (Raw)"})
            return False
        else:
            print(f"DEBUG INGESTÃO: Condição de pular NÃO ATENDIDA: base_existente_no_r2={base_existente_no_r2}, tamanho_ssp_fonte={tamanho_ssp_fonte}, tamanho_ssp_r2={tamanho_ssp_r2}.")
            print("DEBUG INGESTÃO: Prosseguindo com a ingestão da SSP.")

        print(f"📥 A extrair dados da SSP: {url}")
        res_data = requests.get(url, timeout=120)

        df_novo = pd.DataFrame()
        if ano == 2026:
            xls = pd.ExcelFile(res_data.content)
            for sheet_name in xls.sheet_names:
                try:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df_sheet.empty:
                        df_novo = pd.concat([df_novo, df_sheet], ignore_index=True)
                except Exception as sheet_e:
                    print(f"Aviso: Falha ao ler a aba '{sheet_name}': {sheet_e}")
        else:
            df_novo = pd.read_excel(res_data.content)

        if df_novo.empty:
            print("Aviso: DataFrame da SSP está vazio após leitura. Nenhuma ingestão será realizada.")
            robo.enviar_relatorio_operacional("Camada Bronze: DataFrame da SSP vazio. Ingestão pulada.", 
                                       {"URL Fonte": url, "Camada": "Bronze (Raw)"})
            return False

        df_novo.columns = df_novo.columns.str.upper().str.strip().str.replace(' ', '_')

        novos_registros = len(df_novo)

        if base_existente_no_r2:
            print(f"DEBUG INGESTÃO: Baixando base existente do R2 para concatenação: {path_raw}")
            s3.download_file(bucket, path_raw, "raw_local.parquet")
            df_existente = pd.read_parquet("raw_local.parquet")

            df_final = pd.concat([df_existente, df_novo]).drop_duplicates().reset_index(drop=True)
            novos_registros = len(df_final) - len(df_existente)
            print(f"DEBUG INGESTÃO: Base existente e nova concatenadas. Total de registros: {len(df_final)}. Novos: {novos_registros}")
        else:
            print("DEBUG INGESTÃO: Primeira ingestão da camada Bronze. Criando arquivo no R2.")
            df_final = df_novo

        df_final.to_parquet("raw_final.parquet", index=False)
        s3.upload_file("raw_final.parquet", bucket, path_raw)

        if os.path.exists("raw_local.parquet"):
            os.remove("raw_local.parquet")
        if os.path.exists("raw_final.parquet"):
            os.remove("raw_final.parquet")

        robo.enviar_relatorio_operacional("Sincronização com a SSP concluída.", 
                                   {"Novos Registros": novos_registros, 
                                    "Tamanho Fonte": f"{tamanho_ssp_fonte} bytes",
                                    "Camada": "Bronze (Raw)"})
        return True
    except Exception:
        robo.enviar_alerta_tecnico("Ingestão Raw (Bronze)", traceback.format_exc())
        return False
