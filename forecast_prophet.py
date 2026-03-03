import logging
import os
from datetime import timedelta

import pandas as pd
from prophet import Prophet
import holidays


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def carregar_base_analitica(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        logging.info(
            "Base analítica %s não encontrada. Prophet não será executado.",
            path_csv,
        )
        return pd.DataFrame()

    df = pd.read_csv(path_csv, encoding="utf-8-sig")

    if "DATA_OCORRENCIA_BO" not in df.columns:
        logging.warning("Coluna DATA_OCORRENCIA_BO não encontrada no CSV.")
        return pd.DataFrame()

    if "geohash" not in df.columns:
        logging.warning("Coluna geohash não encontrada no CSV.")
        return pd.DataFrame()

    df["DATA_OCORRENCIA_BO"] = pd.to_datetime(
        df["DATA_OCORRENCIA_BO"], errors="coerce"
    )
    df = df.dropna(subset=["DATA_OCORRENCIA_BO"])

    if df.empty:
        logging.info("Nenhum registro temporal válido na base analítica.")
        return pd.DataFrame()

    return df


def gerar_previsoes_prophet(
    df_final: pd.DataFrame,
    horizonte_dias: int = 7,
    top_k: int = 20,
) -> pd.DataFrame:
    df_ts = df_final.copy()
    df_ts = df_ts.dropna(subset=["DATA_OCORRENCIA_BO"])
    if df_ts.empty:
        logging.info("Nenhum dado temporal válido para Prophet.")
        return pd.DataFrame()

    series = (
        df_ts.groupby(["DATA_OCORRENCIA_BO", "geohash"])
        .size()
        .reset_index(name="y")
    )
    series = series.rename(columns={"DATA_OCORRENCIA_BO": "ds"})

    data_min = series["ds"].min().date()
    data_max = series["ds"].max().date() + timedelta(days=horizonte_dias)

    calendario = holidays.Brazil()
    lista_feriados = [
        {"ds": pd.to_datetime(d), "holiday": "feriado_nacional"}
        for d in calendario
        if data_min <= d <= data_max
    ]
    holidays_df = pd.DataFrame(lista_feriados) if lista_feriados else None

    soma_geo = series.groupby("geohash")["y"].sum().reset_index()
    soma_geo = soma_geo.sort_values("y", ascending=False).head(top_k)

    forecasts = []
    logging.info(
        "Gerando previsões Prophet para top %d geohashes (horizonte %d dias, com feriados).",
        top_k,
        horizonte_dias,
    )

    for _, row in soma_geo.iterrows():
        geo = row["geohash"]
        df_geo = series[series["geohash"] == geo].copy()
        if len(df_geo) < 10:
            logging.info(
                "Geohash %s com poucos pontos (%d). Pulando Prophet aqui.",
                geo,
                len(df_geo),
            )
            continue

        modelo = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays_df,
        )

        try:
            modelo.fit(df_geo[["ds", "y"]])
        except Exception as e:
            logging.warning(
                "Prophet falhou para geohash %s (%s). Pulando.", geo, e
            )
            continue

        futuro = modelo.make_future_dataframe(periods=horizonte_dias)
        forecast = modelo.predict(futuro)
        forecast["geohash"] = geo

        forecasts.append(
            forecast[["ds", "geohash", "yhat", "yhat_lower", "yhat_upper"]]
        )

    if not forecasts:
        logging.info("Nenhuma previsão Prophet foi gerada (sem dados suficientes).")
        return pd.DataFrame()

    df_forecast = pd.concat(forecasts, ignore_index=True)
    return df_forecast


def main() -> None:
    path_csv = "analise_consolidada_safedriver.csv"
    df = carregar_base_analitica(path_csv)

    if df.empty:
        logging.info("Base analítica indisponível ou vazia. Encerrando Prophet.")
        return

    horizonte = int(os.environ.get("PROPHET_HORIZON_DIAS", "7"))
    top_k = int(os.environ.get("PROPHET_TOP_GEOHASH", "20"))

    df_forecast = gerar_previsoes_prophet(
        df,
        horizonte_dias=horizonte,
        top_k=top_k,
    )

    if df_forecast.empty:
        logging.info("Sem previsões geradas. Nada a exportar.")
        return

    saida = "forecast_prophet_safedriver.csv"
    df_forecast.to_csv(saida, index=False, encoding="utf-8-sig")
    logging.info("Exportação de previsões Prophet concluída (%s).", saida)


if __name__ == "__main__":
    main()
