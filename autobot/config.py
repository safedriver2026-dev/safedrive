# 5. CLASSIFICAÇÃO POR GRAVIDADE (Base Código Penal)
CATALOGO_CRIMES = {
    "LATROCINIO": {"perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"], "peso": 5.0},
    "EXTORSAO MEDIANTE SEQUESTRO": {"perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"], "peso": 5.0},
    "ROUBO DE VEICULO": {"perfis": ["Motorista", "Motociclista"], "peso": 4.0},
    "ROUBO DE CARGA": {"perfis": ["Motorista"], "peso": 4.0},
    "ROUBO A TRANSEUNTE": {"perfis": ["Pedestre", "Ciclista"], "peso": 4.0},
    "FURTO DE VEICULO": {"perfis": ["Motorista", "Motociclista"], "peso": 3.0},
    "FURTO DE CARGA": {"perfis": ["Motorista"], "peso": 3.0},
    "FURTO DE CELULAR": {"perfis": ["Pedestre", "Ciclista"], "peso": 3.0},
    "DANO": {"perfis": ["Motorista", "Motociclista"], "peso": 2.0},
    "OUTROS": {"perfis": ["Pedestre"], "peso": 1.0}
}

PALAVRAS_CHAVE_PERFIL = {
    "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
    "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"],
    "Motorista": ["VEICULO", "CARGA", "CARRO", "CAMINHAO", "AUTOMOVEL", "ESTACIONAMENTO"],
    "Pedestre": ["TRANSEUNTE", "CELULAR", "PEDESTRE", "CALCADA", "PONTO DE ONIBUS"],
}

LIMITES_SP = {"lat": (-25.5, -19.5), "lon": (-53.5, -44.0)}

ESQUEMA_RAW_CANONICO = {
    "NUM_BO": "string", "DATA_OCORRENCIA_BO": "datetime", "HORA_OCORRENCIA_BO": "string",
    "LATITUDE": "float", "LONGITUDE": "float", "NATUREZA_APURADA": "string", 
    "DESCR_TIPOLOCAL": "string", "DESCR_SUBTIPOLOCAL": "string", "ANO_BASE": "int"
}

COLUNAS_REFINED_EVENTOS = [
    "NUM_BO", "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO", "LATITUDE", "LONGITUDE",
    "NATUREZA_APURADA", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "ANO_BASE"
]

# 1. NORMALIZAÇÃO SEMÂNTICA (Dicionário de Equivalência)
MAPA_SEMANTICO_COLUNAS = {
    "NUM_BO": ["NUM_BO", "NUMERO_BO", "BO_NUMERO", "N_BO", "BO"],
    "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO", "DATAOCORRENCIA", "DATA"],
    "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO", "HORAOCORRENCIA", "HORA"],
    "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL", "LATITUDE_Y"],
    "LONGITUDE": ["LONGITUDE", "LON", "LONG", "LONGITUDEDECIMAL", "LONGITUDE_X"],
    "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA", "TIPO_CRIME", "CRIME", "RUBRICA"],
    "DESCR_TIPOLOCAL": ["DESCR_TIPOLOCAL", "TIPOLOCAL", "LOCAL", "TIPO_LOCAL"],
    "DESCR_SUBTIPOLOCAL": ["DESCR_SUBTIPOLOCAL", "SUBTIPOLOCAL"]
}
