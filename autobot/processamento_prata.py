import polars as pl
import pandas as pd
import h3
import logging

# Configuração do Log de Alta Visibilidade
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self, raio_contagio=2):
        """
        Inicializa a Camada Prata Otimizada.
        :param raio_contagio: Aumentado para 2 para capturar contágio de vizinhos indiretos.
        """
        self.raio_contagio = raio_contagio

    def _gerar_features_espaciais_avancadas(self, df_prata: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o contágio espacial ponderado e densidade de risco.
        """
        logger.info(f"🗺️ Extraindo inteligência geográfica avançada (Raio H3 = {self.raio_contagio})...")
        
        df_pd = df_prata.to_pandas()
        
        if 'TOTAL_CRIMES' not in df_pd.columns:
            df_pd['TOTAL_CRIMES'] = 0
            
        crimes_dit = dict(zip(df_pd['H3_INDEX'], df_pd['TOTAL_CRIMES']))
        usar_grid_disk = hasattr(h3, 'grid_disk')
        
        contagio_ponderado = []
        
        for h3_index in df_pd['H3_INDEX']:
            try:
                # Raio 1 (Peso 1.0)
                vizinhos_r1 = set(h3.grid_disk(h3_index, 1) if usar_grid_disk else h3.k_ring(h3_index, 1))
                vizinhos_r1.discard(h3_index)
                crimes_r1 = sum(crimes_dit.get(viz, 0) for viz in vizinhos_r1)
                
                # Raio 2 (Peso 0.5) - Captura a tendência da região ampliada
                vizinhos_total = set(h3.grid_disk(h3_index, 2) if usar_grid_disk else h3.k_ring(h3_index, 2))
                vizinhos_r2 = vizinhos_total - vizinhos_r1
                vizinhos_r2.discard(h3_index)
                crimes_r2 = sum(crimes_dit.get(viz, 0) for viz in vizinhos_r2)
                
                # Cálculo ponderado: Vizinho próximo vale mais
                score_final = (crimes_r1 * 1.0) + (crimes_r2 * 0.5)
                contagio_ponderado.append(score_final)
                
            except:
                contagio_ponderado.append(0)
                
        df_prata = df_prata.with_columns([
            pl.Series(name="CONTAGIO_PONDERADO", values=contagio_ponderado, dtype=pl.Float64)
        ])
        
        # Cria Feature de 'Pressão de Risco' (Contágio / Densidade)
        # Evita divisão por zero com 0.001
        return df_prata.with_columns(
            (pl.col("CONTAGIO_PONDERADO") / (pl.col("DENSIDADE") + 0.001)).alias("PRESSAO_RISCO_LOCAL")
        )

    def processar_dados(self, df_bronze: pl.DataFrame) -> pl.DataFrame:
        """
        Orquestra o refinamento completo da Prata para o Treinador IA.
        """
        logger.info("⚙️ Iniciando processamento avançado da Camada Prata...")
        
        df_prata = df_bronze.clone()
        
        # 1. Tratamento de Datas (Sazonalidade Preditiva)
        if "DATA_OCORRENCIA" in df_prata.columns:
            df_prata = df_prata.with_columns([
                pl.col("DATA_OCORRENCIA").dt.month().alias("MES_OCORRENCIA"),
                pl.col("DATA_OCORRENCIA").dt.weekday().alias("DIA_SEMANA_OCORRENCIA")
            ])

        # 2. Tipagem e Nulos (Blindagem contra erros de tipos no CatBoost/LightGBM)
        if "TOTAL_CRIMES" in df_prata.columns:
            df_prata = df_prata.with_columns(pl.col("TOTAL_CRIMES").fill_null(0).cast(pl.Int32))
            
        if "DENSIDADE" in df_prata.columns:
            df_prata = df_prata.with_columns(pl.col("DENSIDADE").fill_null(0.0).cast(pl.Float64))
            
        colunas_categoricas = ["NM_BAIRRO", "NM_MUN", "PERFIL_AREA", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL"]
        df_prata = df_prata.with_columns([
            pl.col(col).fill_null("INDEFINIDO").cast(pl.Utf8) 
            for col in colunas_categoricas if col in df_prata.columns
        ])
            
        # 3. Inteligência Geográfica Avançada
        if "H3_INDEX" in df_prata.columns:
            df_prata = self._gerar_features_espaciais_avancadas(df_prata)
        else:
            logger.error("🚨 H3_INDEX ausente: Falha na inteligência espacial.")
        
        logger.info(f"✅ Camada Prata pronta! Novas colunas preditivas geradas. Total: {df_prata.height} registros.")
        
        return df_prata
