CATALOGO_CRIMES = {
    "FURTO DE VEICULO": {
        "perfis": ["Motorista", "Motociclista"],
        "peso": 1.0,
    },
    "ROUBO DE VEICULO": {
        "perfis": ["Motorista", "Motociclista"],
        "peso": 2.5,
    },
    "ROUBO DE CARGA": {
        "perfis": ["Motorista"],
        "peso": 2.5,
    },
    "FURTO DE CARGA": {
        "perfis": ["Motorista"],
        "peso": 1.0,
    },
    "LATROCINIO": {
        "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"],
        "peso": 5.0,
    },
    "EXTORSAO MEDIANTE SEQUESTRO": {
        "perfis": ["Motorista", "Motociclista", "Pedestre", "Ciclista"],
        "peso": 5.0,
    },
    "ROUBO A TRANSEUNTE": {
        "perfis": ["Pedestre", "Ciclista"],
        "peso": 2.5,
    },
    "FURTO DE CELULAR": {
        "perfis": ["Pedestre", "Ciclista"],
        "peso": 1.0,
    },
}

PALAVRAS_CHAVE_PERFIL = {
    "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
    "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"],
    "Motorista": ["VEICULO", "CARGA", "CARRO", "CAMINHAO", "AUTOMOVEL", "ESTACIONAMENTO"],
    "Pedestre": ["TRANSEUNTE", "CELULAR", "PEDESTRE", "CALCADA", "PONTO DE ONIBUS"],
}

TIPOS_LOCAL_PERMITIDOS = [
    "VIA PUBLICA",
    "RODOVIA/ESTRADA",
]

SUBTIPOS_LOCAL_PERMITIDOS = [
    "VIA PUBLICA",
    "TRANSEUNTE",
    "ACOSTAMENTO",
    "AREA DE DESCANSO",
    "BALANCA",
    "CICLOFAIXA",
    "CICLOVIA",
    "DE FRENTE A RESIDENCIA DA VITIMA",
    "FEIRA LIVRE",
    "INTERIOR DE VEICULO DE CARGA",
    "INTERIOR DE VEICULO DE PARTICULAR",
    "POSTO DE AUXILIO",
    "POSTO DE FISCALIZACAO",
    "POSTO POLICIAL",
    "PRACA",
    "PRACA DE PEDAGIO",
    "SEMAFORO",
    "TUNEL/VIADUTO/PONTE",
    "VEICULO EM MOVIMENTO",
]

LIMITES_SP = {
    "lat": (-25.5, -19.5),
    "lon": (-53.5, -44.0),
}

ESQUEMA_RAW_CANONICO = {
    "NOME_DEPARTAMENTO": "string",
    "NOME_SECCIONAL": "string",
    "NOME_DELEGACIA": "string",
    "NOME_MUNICIPIO": "string",
    "NUM_BO": "string",
    "ANO_BO": "int",
    "DATA_REGISTRO": "datetime",
    "DATA_OCORRENCIA_BO": "datetime",
    "HORA_OCORRENCIA_BO": "string",
    "DESC_PERIODO": "string",
    "DESCR_TIPOLOCAL": "string",
    "DESCR_SUBTIPOLOCAL": "string",
    "BAIRRO": "string",
    "LOGRADOURO": "string",
    "NUMERO_LOGRADOURO": "string",
    "LATITUDE": "float",
    "LONGITUDE": "float",
    "NOME_DELEGACIA_CIRCUNSCRICAO": "string",
    "NOME_DEPARTAMENTO_CIRCUNSCRICAO": "string",
    "NOME_SECCIONAL_CIRCUNSCRICAO": "string",
    "NOME_MUNICIPIO_CIRCUNSCRICAO": "string",
    "RUBRICA": "string",
    "DESCR_CONDUTA": "string",
    "NATUREZA_APURADA": "string",
    "MES_ESTATISTICA": "int",
    "ANO_ESTATISTICA": "int",
    "CMD": "string",
    "BTL": "string",
    "CIA": "string",
    "COD IBGE": "string",
    "ANO_BASE": "int",
}

ESQUEMA_TRUSTED = ESQUEMA_RAW_CANONICO.copy()

COLUNAS_REFINED_EVENTOS = [
    "NUM_BO",
    "DATA_OCORRENCIA_BO",
    "HORA_OCORRENCIA_BO",
    "LATITUDE",
    "LONGITUDE",
    "NATUREZA_APURADA",
    "DESCR_TIPOLOCAL",
    "DESCR_SUBTIPOLOCAL",
    "DESCR_CONDUTA",
    "RUBRICA",
    "ANO_BASE",
]

MAPA_SEMANTICO_COLUNAS = {
    "NOME_DEPARTAMENTO": [
        "NOME_DEPARTAMENTO",
        "DEPARTAMENTO",
    ],
    "NOME_SECCIONAL": [
        "NOME_SECCIONAL",
        "SECCIONAL",
    ],
    "NOME_DELEGACIA": [
        "NOME_DELEGACIA",
        "DELEGACIA",
    ],
    "NOME_MUNICIPIO": [
        "NOME_MUNICIPIO",
        "CIDADE",
        "MUNICIPIO",
        "NOME_CIDADE",
    ],
    "NUM_BO": [
        "NUM_BO",
        "NUMERO_BO",
        "N_BO",
        "BO",
    ],
    "ANO_BO": [
        "ANO_BO",
    ],
    "DATA_REGISTRO": [
        "DATA_REGISTRO",
        "DATA_COMUNICACAO_BO",
        "DATA_COMUNICACAO",
        "DATA_REGISTRO_BO",
    ],
    "DATA_OCORRENCIA_BO": [
        "DATA_OCORRENCIA_BO",
        "DATA_FATO",
        "DATA_OCORRENCIA",
    ],
    "HORA_OCORRENCIA_BO": [
        "HORA_OCORRENCIA_BO",
        "HORA_FATO",
        "HORA_OCORRENCIA",
    ],
    "DESC_PERIODO": [
        "DESC_PERIODO",
        "DESCR_PERIODO",
        "PERIODO",
    ],
    "DESCR_TIPOLOCAL": [
        "DESCR_TIPOLOCAL",
        "TIPO_LOCAL",
        "TIPOLOCAL",
    ],
    "DESCR_SUBTIPOLOCAL": [
        "DESCR_SUBTIPOLOCAL",
        "SUBTIPO_LOCAL",
        "SUBTIPOLOCAL",
    ],
    "BAIRRO": [
        "BAIRRO",
    ],
    "LOGRADOURO": [
        "LOGRADOURO",
        "ENDERECO",
    ],
    "NUMERO_LOGRADOURO": [
        "NUMERO_LOGRADOURO",
        "NUMERO",
    ],
    "LATITUDE": [
        "LATITUDE",
        "LAT",
    ],
    "LONGITUDE": [
        "LONGITUDE",
        "LON",
        "LONG",
    ],
    "NOME_DELEGACIA_CIRCUNSCRICAO": [
        "NOME_DELEGACIA_CIRCUNSCRICAO",
    ],
    "NOME_DEPARTAMENTO_CIRCUNSCRICAO": [
        "NOME_DEPARTAMENTO_CIRCUNSCRICAO",
    ],
    "NOME_SECCIONAL_CIRCUNSCRICAO": [
        "NOME_SECCIONAL_CIRCUNSCRICAO",
    ],
    "NOME_MUNICIPIO_CIRCUNSCRICAO": [
        "NOME_MUNICIPIO_CIRCUNSCRICAO",
    ],
    "RUBRICA": [
        "RUBRICA",
        "RUBRICA_CRIME",
    ],
    "DESCR_CONDUTA": [
        "DESCR_CONDUTA",
        "CONDUTA",
        "DESCRICAO_CONDUTA",
    ],
    "NATUREZA_APURADA": [
        "NATUREZA_APURADA",
        "NATUREZA",
        "TIPO_CRIME",
    ],
    "MES_ESTATISTICA": [
        "MES_ESTATISTICA",
    ],
    "ANO_ESTATISTICA": [
        "ANO_ESTATISTICA",
    ],
    "CMD": [
        "CMD",
    ],
    "BTL": [
        "BTL",
    ],
    "CIA": [
        "CIA",
    ],
    "COD IBGE": [
        "COD IBGE",
        "COD_IBGE",
    ],
}
