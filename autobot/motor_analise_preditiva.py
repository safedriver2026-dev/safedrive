        if not precisa_reconstruir:
            print("Nenhuma atualização pendente. Camada Ouro já aderente às novas regras.", file=sys.stdout)

            # 1) Garante que BigQuery está sincronizado mesmo sem rebuild
            status_bq = self.publicar_bigquery_a_partir_de_arquivos(
                bq_project, bq_dataset, bq_cred_json
            )

            # 2) Calcula métricas reais a partir dos arquivos existentes
            registros = 0
            media_risco = 0.0

            prata_path = self.pastas["prata"] / "camada_prata.parquet"
            valid_path = self.pastas["ouro"] / "validacao_modelo.parquet"

            try:
                if prata_path.exists():
                    df_prata = pl.read_parquet(prata_path)
                    registros = df_prata.height

                if valid_path.exists():
                    df_valid = pl.read_parquet(valid_path)
                    if "ESCORE_RISCO" in df_valid.columns and df_valid.height > 0:
                        media_risco = float(df_valid["ESCORE_RISCO"].mean())
            except Exception as e:
                # Se der qualquer erro de leitura, loga mas não derruba o pipeline
                print(f"⚠️ Falha ao calcular métricas de resumo (sem rebuild): {e}", file=sys.stderr)

            # 3) Backup R2
            status_cloud = "❌ Desconectado"
            if self.s3:
                try:
                    for f in self.pastas["ouro"].glob("*.parquet"):
                        self.s3.upload_file(str(f), self.bucket, f"ouro/{f.name}")
                    status_cloud = "✅ Upload Realizado"
                except Exception:
                    status_cloud = "⚠️ Falha no Backup R2"

            # 4) Notifica no Discord com os valores ATUAIS
            tempo_total = time.time() - self.t_inicio
            self.discord.notificar_sucesso(
                "Execução Concluída (Sem Rebuild de Modelo)",
                tempo_total,
                registros=registros,
                media_risco=media_risco,
                status_s3=status_cloud,
                status_bq=status_bq,
            )
            return
