import unicodedata

def normalizar_texto(texto):
    if not isinstance(texto, str): 
        return ""
    texto_norm = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return " ".join(texto_norm.lower().split())

# Mapeamento: Trecho-chave da Descricao Oficial -> Nome Canonico do SafeDriver
REGRAS_DESCRICAO = {
    "latitude da ocorrencia": "LATITUDE",
    "longitude da ocorrencia": "LONGITUDE",
    "natureza juridica": "RUBRICA",
    "classificacao da natureza criminal": "NATUREZA_APURADA",
    "paragrafos, incisos ou circunstancia": "DESCR_CONDUTA",
    
    # Mapeamento para sanitizacao LGPD
    "numero do boletim": "NUM_BO",
    "endereco dos fatos": "LOGRADOURO",
    "numero do logradouro": "NUMERO_LOGRADOURO",
    "delegacia responsavel pelo registro": "NOME_DELEGACIA",
    "departamento responsavel pelo registro": "NOME_DEPARTAMENTO",
    "seccional responsavel pelo registro": "NOME_SECCIONAL",
    "delegacia de circunscricao": "NOME_DELEGACIA_CIRCUNSCRICAO",
    "departamento de circunscricao": "NOME_DEPARTAMENTO_CIRCUNSCRICAO",
    "seccional de circunscricao": "NOME_SECCIONAL_CIRCUNSCRICAO",
    "municipio da delegacia de circunscricao": "NOME_MUNICIPIO_CIRCUNSCRICAO"
}

def identificar_coluna_pela_descricao(descricao_suja):
    desc_norm = normalizar_texto(descricao_suja)
    
    for trecho_chave, nome_canonico in REGRAS_DESCRICAO.items():
        if trecho_chave in desc_norm:
            return nome_canonico
            
    return None
