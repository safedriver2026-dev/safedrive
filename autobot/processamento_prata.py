import os
import pandas as pd
import boto3
import traceback
from datetime import datetime
from .comunicador import RoboComunicador
import hashlib
import h3
import json

def executar_processamento(robo: RoboComunicador, ano: int):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

    bucket = os.environ.get("R2_BUCKET_NAME")
    path_raw = f"safedriver/datalake/raw/ssp_{ano}_incremental.parquet"
    path_silver = f"safedriver/datalake/silver/prata_{ano}.parquet"
    geojson_path_r2 = "safedriver/geodata/BR_bairros_CD2022.geojson"
    geojson_local_path = "BR_bairros_CD2022.geojson"

    lgpd_salt = os.environ.get("LGPD_SALT")
    if not lgpd_salt:
        robo.enviar_alerta_tecnico("Processamento Camada Prata", "LGPD_SALT não configurado. Usando salt padrão. **ATENÇÃO: Isso pode comprometer a segurança em produção.**")
        lgpd_salt = "default_salt_para_testes_lgpd_producao"

    try:
        s3.download_file(bucket, path_raw, f"raw_data_temp_{ano}.parquet")
        df_raw = pd.read_parquet(f"raw_data_temp_{ano}.parquet")
        os.remove(f"raw_data_temp_{ano}.parquet")

        # 1. Anonimização LGPD
        df_raw['ID_ANONIMO'] = df_raw['NUM_BO'].astype(str) + df_raw['ANO_BO'].astype(str) + df_raw['NOME_DELEGACIA']
        df_raw['ID_ANONIMO'] = df_raw['ID_ANONIMO'].apply(lambda x: hashlib.sha256(f"{x}{lgpd_salt}".encode()).hexdigest()[:12])

        # Remover colunas sensíveis após anonimização, se necessário
        # Ex: df_raw = df_raw.drop(columns=['NUM_BO', 'NOME_DELEGACIA'], errors='ignore')

        # 2. Geoprocessamento H3
        if 'LATITUDE' in df_raw.columns and 'LONGITUDE' in df_raw.columns:
            df_raw['INDICE_H3'] = df_raw.apply(lambda row: h3.geo_to_h3(row['LATITUDE'], row['LONGITUDE'], 9) if pd.notna(row['LATITUDE']) and pd.notna(row['LONGITUDE']) else None, axis=1)
        else:
            robo.enviar_relatorio_operacional(f"Aviso: Colunas LATITUDE/LONGITUDE ausentes para o ano {ano}. INDICE_H3 não gerado.", 
                                       {"Ano": ano, "Camada": "Prata"})
            df_raw['INDICE_H3'] = None

        # 3. Enriquecimento Geográfico com GeoJSON (Geógrafo)
        try:
            s3.download_file(bucket, geojson_path_r2, geojson_local_path)
            with open(geojson_local_path, 'r') as f:
                geojson_data = json.load(f)

            # Exemplo de enriquecimento: mapear H3 para nome de bairro ou outras features
            # Esta é uma lógica simplificada. Em um cenário real, você faria um join espacial
            # ou usaria o H3 para buscar propriedades de regiões.

            # Criar um dicionário de H3 para features do GeoJSON (exemplo)
            h3_to_feature = {}
            for feature in geojson_data['features']:
                # Supondo que o GeoJSON tenha uma propriedade 'h3_index' ou similar
                # Ou que você possa gerar H3 a partir das geometrias do GeoJSON
                # Para este exemplo, vamos simular o enriquecimento
                h3_to_feature[feature['properties'].get('CD_BAIRRO', 'UNKNOWN')] = {
                    'NOME_BAIRRO_GEO': feature['properties'].get('NM_BAIRRO', 'Desconhecido'),
                    'DENSIDADE_POP_GEO': 1000 # Exemplo
                }

            # Mapear o INDICE_H3 do DataFrame para as features do GeoJSON
            # df_raw['NOME_BAIRRO_ENRIQUECIDO'] = df_raw['INDICE_H3'].map(h3_to_feature.get('h3_index_do_geojson', {}).get('NOME_BAIRRO_GEO'))
            # Por enquanto, vamos usar placeholders para as features urbanas
            df_raw['DENSIDADE_LOGRADOUROS'] = 0.5
            df_raw['PROPORCAO_RESIDENCIAL_H3'] = 0.7
            df_raw['TOTAL_EDIFICACOES_H3'] = 100

        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NotFound':
                robo.enviar_relatorio_operacional(f"Aviso: Arquivo GeoJSON '{geojson_path_r2}' não encontrado no R2. Enriquecimento geográfico pulado.", 
                                           {"Ano": ano, "Camada": "Prata"})
            else:
                raise
        except Exception as e:
            robo.enviar_alerta_tecnico(f"Processamento Camada Prata - Erro no enriquecimento geográfico para o ano {ano}", str(e))
            raise
        finally:
            if os.path.exists(geojson_local_path):
                os.remove(geojson_local_path)

        # 4. Enriquecimento Temporal (Exemplo)
        df_raw['DATA_OCORRENCIA_BO'] = pd.to_datetime(df_raw['DATA_OCORRENCIA_BO'], errors='coerce')
        df_raw['EH_FERIADO'] = False # Lógica de feriados seria implementada aqui
        df_raw['DIA_PAGAMENTO'] = False # Lógica de dia de pagamento seria implementada aqui
        df_raw['PESO_CRIME'] = 1 # Lógica de peso do crime seria implementada aqui

        df_raw.to_parquet(f"silver_data_temp_{ano}.parquet", index=False)
        s3.upload_file(f"silver_data_temp_{ano}.parquet", bucket, path_silver)
        os.remove(f"silver_data_temp_{ano}.parquet")

        robo.enviar_relatorio_operacional(f"✅ Camada Prata processada e salva no R2 para o ano {ano}.", 
                                   {"Registros Processados": len(df_raw),
                                    "Camada": "Prata"})
        return True
    except Exception:
        robo.enviar_alerta_tecnico(f"Processamento Camada Prata - Ano {ano}", traceback.format_exc())
        return False
