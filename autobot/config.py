# Catalogo basico de fallback para taxonomia de crimes
CATALOGO_CRIMES = {
    'FURTO DE VEICULO': {'perfis': ['Motorista', 'Motociclista'], 'peso': 1.0},
    'ROUBO DE VEICULO': {'perfis': ['Motorista', 'Motociclista'], 'peso': 2.5},
    'ROUBO DE CARGA': {'perfis': ['Motorista'], 'peso': 2.5},
    'FURTO DE CARGA': {'perfis': ['Motorista'], 'peso': 1.0},
    'LATROCINIO': {'perfis': ['Motorista', 'Motociclista', 'Pedestre', 'Ciclista'], 'peso': 5.0},
    'EXTORSAO MEDIANTE SEQUESTRO': {'perfis': ['Motorista', 'Motociclista', 'Pedestre', 'Ciclista'], 'peso': 5.0},
    'ROUBO A TRANSEUNTE': {'perfis': ['Pedestre', 'Ciclista'], 'peso': 2.5},
    'FURTO DE CELULAR': {'perfis': ['Pedestre', 'Ciclista'], 'peso': 1.0}
}

# Dicionario de NLP para inferencia contextual na camada Trusted
PALAVRAS_CHAVE_PERFIL = {
    'Ciclista': ['BICI', 'CICLO', 'BICICLETA', 'PEDALAR'],
    'Motociclista': ['MOTO', 'MOTOCICLETA', 'CAPACETE', 'MOTOBOY'],
    'Motorista': ['VEICULO', 'CARGA', 'CARRO', 'CAMINHAO', 'AUTOMOVEL', 'ESTACIONAMENTO'],
    'Pedestre': ['TRANSEUNTE', 'CELULAR', 'PEDESTRE', 'CALCADA', 'PONTO DE ONIBUS']
}

TIPOS_LOCAL_PERMITIDOS = ['VIA PUBLICA', 'RODOVIA/ESTRADA']
SUBTIPOS_LOCAL_PERMITIDOS = [
    'VIA PUBLICA', 'TRANSEUNTE', 'ACOSTAMENTO', 'AREA DE DESCANSO',
    'BALANCA', 'CICLOFAIXA', 'CICLOVIA', 'DE FRENTE A RESIDENCIA DA VITIMA',
    'FEIRA LIVRE', 'INTERIOR DE VEICULO DE CARGA', 'INTERIOR DE VEICULO DE PARTICULAR',
    'POSTO DE AUXILIO', 'POSTO DE FISCALIZACAO', 'POSTO POLICIAL',
    'PRACA', 'PRACA DE PEDAGIO', 'SEMAFORO', 'TUNEL/VIADUTO/PONTE',
    'VEICULO EM MOVIMENTO'
]

LIMITES_SP = {'lat': (-25.5, -19.5), 'lon': (-53.5, -44.0)}

ESQUEMA_TRUSTED = {
    'NOME_DEPARTAMENTO': 'string', 'NOME_SECCIONAL': 'string', 'NOME_DELEGACIA': 'string',
    'NOME_MUNICIPIO': 'string', 'NUM_BO': 'string', 'ANO_BO': 'int',
    'DATA_REGISTRO': 'datetime', 'DATA_OCORRENCIA_BO': 'datetime', 'HORA_OCORRENCIA_BO': 'string',
    'DESC_PERIODO': 'string', 'DESCR_SUBTIPOLOCAL': 'string', 'BAIRRO': 'string',
    'LOGRADOURO': 'string', 'NUMERO_LOGRADOURO': 'string', 'LATITUDE': 'float',
    'LONGITUDE': 'float', 'NOME_DELEGACIA_CIRCUNSCRICAO': 'string', 'NOME_DEPARTAMENTO_CIRCUNSCRICAO': 'string',
    'NOME_SECCIONAL_CIRCUNSCRICAO': 'string', 'NOME_MUNICIPIO_CIRCUNSCRICAO': 'string', 'RUBRICA': 'string',
    'DESCR_CONDUTA': 'string', 'NATUREZA_APURADA': 'string', 'MES_ESTATISTICA': 'int',
    'ANO_ESTATISTICA': 'int', 'CMD': 'string', 'BTL': 'string', 'CIA': 'string',
    'COD IBGE': 'string', 'ANO_BASE': 'int'
}

# A camada Refined agora herda campos descritivos para a IA minerar
COLUNAS_REFINED = [
    'NUM_BO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO', 
    'LATITUDE', 'LONGITUDE', 'NATUREZA_APURADA', 
    'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 'DESCR_CONDUTA', 'RUBRICA', 'ANO_BASE'
]
