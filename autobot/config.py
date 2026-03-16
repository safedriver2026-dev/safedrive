CATALOGO_CRIMES = {
    "LATROCINIO": {"peso": 5.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]},
    "EXTORSAO": {"peso": 5.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]},
    "ROUBO": {"peso": 4.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]},
    "FURTO": {"peso": 3.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]},
    "DANO": {"peso": 2.0, "perfis": ["Motorista", "Motociclista"]},
    "HOMICIDIO": {"peso": 5.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]}
}

PALAVRAS_CHAVE_PERFIL = {
    "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
    "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"],
    "Motorista": ["VEICULO", "CARGA", "CARRO", "CAMINHAO", "AUTOMOVEL", "ESTACIONAMENTO"],
    "Pedestre": ["TRANSEUNTE", "CELULAR", "PEDESTRE", "CALCADA", "PONTO DE ONIBUS"],
}

LIMITES_GEOGRAFICOS = {"lat": (-25.5, -19.5), "lon": (-53.5, -44.0)}

ESQUEMA_CANONICO = {
    "NUM_BO": "string", "DATA_OCORRENCIA_BO": "datetime", "HORA_OCORRENCIA_BO": "string",
    "LATITUDE": "float", "LONGITUDE": "float", "NATUREZA_APURADA": "string", 
    "DESCR_TIPOLOCAL": "string", "ANO_BASE": "int"
}

DICIONARIO_SEMANTICO = {
    "NUM_BO": ["NUM_BO", "NUMERO_BO", "BO_NUMERO", "N_BO"],
    "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO", "DATAOCORRENCIA", "DATA"],
    "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO", "HORAOCORRENCIA"],
    "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL", "LATITUDE_Y"],
    "LONGITUDE": ["LONGITUDE", "LON", "LONG", "LONGITUDEDECIMAL", "LONGITUDE_X"],
    "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA", "TIPO_CRIME", "RUBRICA"],
    "DESCR_TIPOLOCAL": ["DESCR_TIPOLOCAL", "TIPOLOCAL", "LOCAL", "TIPO_LOCAL"]
}
COLUNAS_REFINED = list(ESQUEMA_CANONICO.keys())
