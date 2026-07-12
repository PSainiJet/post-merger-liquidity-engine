"""Time-series component (ARIMA-based) of the liquidity ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class TimeSeriesResult:
    models: dict[str, object]
    order: tuple[int, int, int]
    aic_scores: dict[str, float]


def _fit_currency_series(
    series: pd.Series,
    order: tuple[int, int, int],
) -> tuple[object, float]:
    """Fit SARIMAX model for a single currency liquidity series."""
    clean = series.dropna()
    if len(clean) < 50:
        return None, float("inf")

    model = SARIMAX(
        clean,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)
    return fitted, fitted.aic


def train_time_series_models(
    df: pd.DataFrame,
    order: tuple[int, int, int] = (2, 1, 2),
    target_col: str = "liquidity_buffer_m",
) -> TimeSeriesResult:
    """Train per-currency ARIMA models on liquidity buffer time series."""
    models: dict[str, object] = {}
    aic_scores: dict[str, float] = {}

    for currency in df["currency"].unique():
        ccy_df = df[df["currency"] == currency].sort_values("timestamp")
        series = ccy_df.set_index("timestamp")[target_col]
        fitted, aic = _fit_currency_series(series, order)
        if fitted is not None:
            models[currency] = fitted
            aic_scores[currency] = aic

    return TimeSeriesResult(models=models, order=order, aic_scores=aic_scores)


def predict_time_series(
    result: TimeSeriesResult,
    df: pd.DataFrame,
    horizon_steps: int,
) -> dict[str, np.ndarray]:
    """Forecast liquidity buffer per currency over the prediction horizon."""
    predictions: dict[str, np.ndarray] = {}

    for currency, model in result.models.items():
        forecast = model.forecast(steps=horizon_steps)
        predictions[currency] = np.asarray(forecast)

    return predictions


def predict_time_series_at_horizon(
    result: TimeSeriesResult,
    horizon_steps: int,
) -> dict[str, float]:
    """Single-point forecast at the 48h horizon for each currency."""
    point_forecasts: dict[str, float] = {}
    for currency, model in result.models.items():
        forecast = model.forecast(steps=horizon_steps)
        point_forecasts[currency] = float(forecast.iloc[-1])
    return point_forecasts


def save_time_series_models(result: TimeSeriesResult, path: str) -> None:
    joblib.dump(
        {
            "models": result.models,
            "order": result.order,
            "aic_scores": result.aic_scores,
        },
        path,
    )


def load_time_series_models(path: str) -> TimeSeriesResult:
    data = joblib.load(path)
    return TimeSeriesResult(**data)
