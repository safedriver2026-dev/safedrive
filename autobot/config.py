CATALOGO_CRIMES = {
    "FURTO DE VEICULO": {"perfis": ["Motorista", "Motociclista"], "peso": 1.0},
    "ROUBO DE VEICULO": {"perfis": ["Motorista", "Motociclista"], "peso": 2.5},
    "ROUBO DE CARGA": {"perfis": ["Motorista"], "peso": 2.5},
    "FURTO DE CARGA": {"perfis": ["Motorista"], "peso": 1.0},
    "LATROCINIO": {"perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"], "peso": 5.0},
    "EXTORSAO MEDIANTE SEQUESTRO": {"perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"], "peso": 5.0},
    "ROUBO A TRANSEUNTE": {"perfis": ["Pedestre", "Ciclista"], "peso": 2.5},
    "FURTO DE CELULAR": {"perfis": ["Pedestre", "Ciclista"], "peso": 1.0},
}

PALAVRAS_CHAVE_PERFIL = {
    "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
    "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"],
    "Motorista": ["VEICULO", "CARGA", "CARRO", "CAMINHAO", "AUTOMOVEL", "ESTACIONAMENTO"],
    "Pedestre": ["TRANSEUNTE", "CELULAR", "PEDESTRE", "CALCADA", "PONTO DE ONIBUS"],
}

TIPOS_LOCAL_PERMITIDOS = ["VIA PUBLICA", "RODOVIA/ESTRADA"]

LIMITES_SP = {"lat": (-24.0, -23.3), "lon": (-47.0, -46.3)}

ESQUEMA_RAW_CANONICO = {
    "NUM_BO": "string", "DATA_OCORRENCIA_BO": "datetime", "HORA_OCORRENCIA_BO": "string",
    "LATITUDE": "float", "LONGITUDE": "float", "NATUREZA_APURADA": "string", 
    "DESCR_TIPOLOCAL": "string", "DESCR_SUBTIPOLOCAL": "string", "ANO_BASE": "int"
}

MAPA_SEMANTICO_COLUNAS = {
    "NUM_BO": ["NUM_BO", "NUMERO_BO"],
    "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO"],
    "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO"],
    "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL"],
    "LONGITUDE": ["LONGITUDE", "LON", "LONGITUDEDECIMAL"],
    "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA"],
    "DESCR_TIPOLOCAL": ["DESCR_TIPOLOCAL", "TIPOLOCAL"]
}
