"""Ensemble model combining Random Forest and time-series forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.random_forest_model import (
    RandomForestResult,
    load_random_forest,
    predict_random_forest,
    save_random_forest,
    train_random_forest,
)
from src.models.time_series_model import (
    TimeSeriesResult,
    load_time_series_models,
    predict_time_series_at_horizon,
    save_time_series_models,
    train_time_series_models,
)


@dataclass
class EnsembleResult:
    random_forest: RandomForestResult
    time_series: TimeSeriesResult
    rf_weight: float
    ts_weight: float


def train_ensemble(df: pd.DataFrame, config: dict | None = None) -> EnsembleResult:
    """Train both model components and return combined ensemble."""
    cfg = config or load_config()
    rf_cfg = cfg["models"]["random_forest"]
    ts_cfg = cfg["models"]["time_series"]

    rf_result = train_random_forest(
        df,
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
    )
    ts_result = train_time_series_models(
        df,
        order=tuple(ts_cfg["arima_order"]),
    )

    return EnsembleResult(
        random_forest=rf_result,
        time_series=ts_result,
        rf_weight=rf_cfg["weight"],
        ts_weight=ts_cfg["weight"],
    )


def predict_ensemble_liquidity(
    ensemble: EnsembleResult,
    df: pd.DataFrame,
    horizon_hours: int = 48,
    bucket_minutes: int = 15,
) -> pd.DataFrame:
    """
    Produce weighted ensemble liquidity forecast per currency.

    Returns a DataFrame with predicted 48h liquidity, shortfall risk, and alert level.
    """
    horizon_steps = int(horizon_hours * 60 / bucket_minutes)

    # RF: predict on latest row per currency
    latest = df.sort_values("timestamp").groupby("currency", as_index=False).tail(1)
    rf_preds = predict_random_forest(
        ensemble.random_forest.model,
        latest,
        ensemble.random_forest.currency_encoder,
    )

    # TS: point forecast at horizon
    ts_preds = predict_time_series_at_horizon(ensemble.time_series, horizon_steps)

    cfg = load_config()
    thresholds = cfg["thresholds"]
    currencies = cfg["project"]["currencies"]

    rows = []
    for i, ccy in enumerate(latest["currency"].values):
        if ccy not in currencies:
            continue

        rf_val = rf_preds[i]
        ts_val = ts_preds.get(ccy, rf_val)
        ensemble_pred = (
            ensemble.rf_weight * rf_val + ensemble.ts_weight * ts_val
        )

        min_required = thresholds[ccy]["min_liquid_assets_m"]
        buffer_pct = (ensemble_pred - min_required) / min_required * 100

        if buffer_pct <= thresholds[ccy]["critical_buffer_pct"]:
            alert = "CRITICAL"
        elif buffer_pct <= thresholds[ccy]["warning_buffer_pct"]:
            alert = "WARNING"
        else:
            alert = "OK"

        shortfall_m = max(min_required - ensemble_pred, 0)

        rows.append(
            {
                "currency": ccy,
                "predicted_liquidity_48h_m": round(ensemble_pred, 2),
                "min_required_m": min_required,
                "buffer_pct": round(buffer_pct, 2),
                "shortfall_m": round(shortfall_m, 2),
                "rf_prediction_m": round(rf_val, 2),
                "ts_prediction_m": round(ts_val, 2),
                "alert_level": alert,
            }
        )

    return pd.DataFrame(rows)


def save_ensemble(ensemble: EnsembleResult, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_random_forest(ensemble.random_forest, str(artifacts_dir / "random_forest.joblib"))
    save_time_series_models(ensemble.time_series, str(artifacts_dir / "time_series.joblib"))


def load_ensemble(artifacts_dir: Path, config: dict | None = None) -> EnsembleResult:
    cfg = config or load_config()
    rf = load_random_forest(str(artifacts_dir / "random_forest.joblib"))
    ts = load_time_series_models(str(artifacts_dir / "time_series.joblib"))
    return EnsembleResult(
        random_forest=rf,
        time_series=ts,
        rf_weight=cfg["models"]["random_forest"]["weight"],
        ts_weight=cfg["models"]["time_series"]["weight"],
    )
