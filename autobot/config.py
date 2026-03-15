CATALOGO_CRIMES = {
    "LATROCINIO": {"peso": 5.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]},
    "EXTORSAO MEDIANTE SEQUESTRO": {"peso": 5.0, "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"]},
    "ROUBO DE VEICULO": {"peso": 4.0, "perfis": ["Motorista", "Motociclista"]},
    "ROUBO DE CARGA": {"peso": 4.0, "perfis": ["Motorista"]},
    "ROUBO A TRANSEUNTE": {"peso": 4.0, "perfis": ["Pedestre", "Ciclista"]},
    "FURTO DE VEICULO": {"peso": 3.0, "perfis": ["Motorista", "Motociclista"]},
    "FURTO DE CARGA": {"peso": 3.0, "perfis": ["Motorista"]},
    "FURTO DE CELULAR": {"peso": 3.0, "perfis": ["Pedestre", "Ciclista"]},
    "DANO": {"peso": 2.0, "perfis": ["Motorista", "Motociclista"]},
    "OUTROS": {"peso": 1.0, "perfis": ["Pedestre"]}
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
