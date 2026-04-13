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

    print(f"DEBUG INGESTÃO: Iniciando execução para o ano {ano}.")
    print(f"DEBUG INGESTÃO: URL da SSP: {url}")
    print(f"DEBUG INGESTÃO: Bucket R2: {bucket}")
    print(f"DEBUG INGESTÃO: Caminho Raw no R2: {path_raw}")

    try:
        # 1. Obter tamanho do arquivo na fonte SSP
        res_head = requests.head(url, timeout=15)
        tamanho_ssp_fonte = int(res_head.headers.get('Content-Length', 0))
        print(f"DEBUG INGESTÃO: Tamanho da fonte SSP: {tamanho_ssp_fonte} bytes.")

        if tamanho_ssp_fonte == 0:
            print(f"⚠️ Camada Bronze: Fonte SSP ({url}) retornou tamanho 0. Não há dados para ingestão.")
            robo.enviar_relatorio_operacional("Camada Bronze: Fonte SSP retornou tamanho 0. Ingestão pulada.", 
                                       {"URL Fonte": url, "Camada": "Bronze (Raw)"})
            return False

        # 2. Verificar se o arquivo existe no R2 e obter seu tamanho
        base_existente_no_r2 = False
        tamanho_ssp_r2 = 0
        try:
            obj_info = s3.head_object(Bucket=bucket, Key=path_raw)
            tamanho_ssp_r2 = obj_info['ContentLength']
            base_existente_no_r2 = True
            print(f"DEBUG INGESTÃO: Arquivo {path_raw} ENCONTRADO no R2. Tamanho: {tamanho_ssp_r2} bytes.")
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NotFound':
                base_existente_no_r2 = False # Explicitamente False se não encontrado
                print(f"DEBUG INGESTÃO: Arquivo {path_raw} NÃO encontrado no R2 (NotFound).")
            else:
                print(f"DEBUG INGESTÃO: Erro inesperado ao verificar {path_raw} no R2: {e}")
                raise # Re-lança outros erros do S3
        except Exception as e:
            print(f"DEBUG INGESTÃO: Erro GENÉRICO ao verificar {path_raw} no R2: {e}")
            raise # Re-lança outros erros

        # 3. Lógica para decidir se deve pular a ingestão
        # AQUI ESTÁ A MUDANÇA CRÍTICA:
        # Só pula a ingestão se o arquivo EXISTE NO R2 E o tamanho da fonte é o MESMO do R2.
        # Se base_existente_no_r2 for False, a condição de pular NUNCA será verdadeira.
        # Se base_existente_no_r2 for True, mas os tamanhos forem diferentes, a condição de pular NUNCA será verdadeira.
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

        # Se chegamos aqui, significa que precisamos fazer a ingestão.
        print(f"📥 A extrair dados da SSP: {url}")
        res_data = requests.get(url, timeout=120)

        df_novo = pd.DataFrame()
        if ano == 2026: # Lógica específica para 2026 com múltiplas abas
            xls = pd.ExcelFile(res_data.content)
            for sheet_name in xls.sheet_names:
                try:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df_sheet.empty:
                        df_novo = pd.concat([df_novo, df_sheet], ignore_index=True)
                except Exception as sheet_e:
                    print(f"Aviso: Falha ao ler a aba '{sheet_name}': {sheet_e}")
            if df_novo.empty:
                raise ValueError("Nenhum dado válido encontrado nas abas do arquivo SSP.")
        else:
            df_novo = pd.read_excel(res_data.content)

        novos_registros = len(df_novo) # Inicializa com o tamanho do df_novo

        if base_existente_no_r2: # Se a base já existia no R2, tentamos fazer a concatenação incremental
            try:
                s3.download_file(bucket, path_raw, "raw_local.parquet")
                df_atual = pd.read_parquet("raw_local.parquet")

                # Concatena e remove duplicatas para garantir incrementalidade
                df_final = pd.concat([df_atual, df_novo]).drop_duplicates(
                    subset=['NUM_BO', 'ANO_BO', 'NOME_DELEGACIA'], keep='first'
                )
                novos_registros = len(df_final) - len(df_atual)
                print(f"Adicionados {novos_registros} novos registros à base existente.")
            except Exception as concat_e:
                print(f"Aviso: Falha ao carregar base existente do R2 para concatenação: {concat_e}. Prosseguindo com apenas os novos dados.")
                df_final = df_novo # Usa apenas os novos dados se falhar ao carregar o existente
        else: # Se a base NÃO existia no R2, é a primeira ingestão
            print("Primeira ingestão da camada Bronze. Criando arquivo no R2.")
            df_final = df_novo # df_final é apenas o df_novo na primeira ingestão

        # Salvar o DataFrame final no R2
        df_final.to_parquet("raw_final.parquet", index=False)
        s3.upload_file("raw_final.parquet", bucket, path_raw)

        # Limpeza de arquivos temporários
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
