from motor_safedriver import SafeDriverEngine


def test_engine_instancia():
    engine = SafeDriverEngine()
    assert hasattr(engine, "executar_pipeline")
    assert callable(engine.executar_pipeline)


def test_engine_tem_metodos_privados_essenciais():
    engine = SafeDriverEngine()
    for nome in [
        "_carregar_historico",
        "_sanear_geo",
        "_normalizar_campos_texto",
        "_filtrar_conteudo",
        "_aplicar_pesos",
        "_preparar_grid",
        "_montar_features_xgb",
        "_treinar_xgb",
        "_aplicar_modelo_xgb",
    ]:
        assert hasattr(engine, nome), f"Engine deveria expor o método {nome}"
