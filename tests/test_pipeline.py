"""Basic tests for the liquidity prediction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_simulation.pipeline import generate_all_streams
from src.models.ensemble import predict_ensemble_liquidity, train_ensemble


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def small_dataset(config):
    """Generate a smaller dataset for faster tests."""
    config = config.copy()
    config["data_simulation"] = {**config["data_simulation"], "history_days": 14}
    return generate_all_streams(config)


def test_config_loads(config):
    assert config["project"]["horizon_hours"] == 48
    assert "CHF" in config["project"]["currencies"]


def test_generate_all_streams(small_dataset):
    assert "master_features" in small_dataset
    master = small_dataset["master_features"]
    assert len(master) > 0
    assert "currency" in master.columns
    assert set(master["currency"].unique()) == {"CHF", "USD", "EUR"}


def test_train_and_predict(small_dataset, config):
    master = small_dataset["master_features"].dropna(subset=["target_liquidity_48h_m"])
    ensemble = train_ensemble(master, config)
    forecast = predict_ensemble_liquidity(
        ensemble,
        master,
        horizon_hours=48,
        bucket_minutes=15,
    )
    assert len(forecast) == 3
    assert "alert_level" in forecast.columns
    assert set(forecast["alert_level"]).issubset({"OK", "WARNING", "CRITICAL"})


def test_forecast_has_required_columns(small_dataset, config):
    master = small_dataset["master_features"].dropna(subset=["target_liquidity_48h_m"])
    ensemble = train_ensemble(master, config)
    forecast = predict_ensemble_liquidity(ensemble, master)
    required = {"currency", "predicted_liquidity_48h_m", "shortfall_m", "alert_level"}
    assert required.issubset(set(forecast.columns))
