# ... (mantenha os imports e o __init__ que você já tem)

    def _obter_contexto_preditivo_total(self):
        try:
            resp_m = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            df_malha = pl.read_parquet(io.BytesIO(resp_m['Body'].read()))

            ano_atual = datetime.now().year
            path_prata = self._get_path("prata", f"ssp_consolidada_{ano_atual}.parquet")
            df_prata = pl.read_parquet(io.BytesIO(self.s3.get_object(Bucket=self.bucket, Key=path_prata)['Body'].read()))

            # AGORA AS COLUNAS EXISTEM NA PRATA
            df_prata_agg = df_prata.group_by("H3_INDEX").agg([
                pl.col("TOTAL_CRIMES").sum().alias("TOTAL_CRIMES"),
                pl.col("RANKING_RISCO_LOCAL").mean().alias("RANKING_PRATA"),
                pl.col("INDICE_EXPOSICAO").mean().alias("EXPOSICAO_PRATA")
            ])

            df_enriquecida = self._gerar_contagio_na_malha_total(df_malha, df_prata_agg)

            hoje = datetime.now()
            df_final = df_enriquecida.with_columns([
                pl.col("NM_MUN").fill_null("SAO PAULO"),
                pl.col("NM_BAIRRO").fill_null("INDEFINIDO"),
                pl.lit("TARDE").alias("PERIODO_DIA"), 
                pl.lit("PEDESTRE").alias("PERFIL_ALVO"),
                pl.lit("VIA PUBLICA").alias("TIPO_LOCAL"),
                pl.lit(hoje.month).alias("MES_OCORRENCIA"),
                pl.lit(hoje.weekday()).alias("DIA_SEMANA"), # Alinhado com a Prata
                pl.lit(1 if 5 <= hoje.day <= 10 else 0).alias("IS_PAGAMENTO"),
                pl.lit(1 if hoje.weekday() >= 6 else 0).alias("IS_FDS"),
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE"),
                pl.col("RANKING_PRATA").fill_null(0.5).alias("RANKING_RISCO_LOCAL"),
                pl.col("EXPOSICAO_PRATA").fill_null(0.1).alias("INDICE_EXPOSICAO")
            ])

            df_pd = df_final.to_pandas()
            # Tipagem forçada para o CatBoost não reclamar
            for col in self.features_categoricas:
                df_pd[col] = df_pd[col].astype(str).astype('category')
            for col in self.features_numericas:
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce').fillna(0.0)
            
            return df_pd
        except Exception as e:
            logger.error(f"OURO: Erro ao consolidar contexto: {e}")
            return None
# ... (mantenha o restante do arquivo)
