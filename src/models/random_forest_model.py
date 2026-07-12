"""Random Forest component of the liquidity ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS = [
    "total_liquid_assets_m",
    "total_hourly_outflow_m",
    "total_pending_settlements_m",
    "overnight_rate_pct",
    "interbank_spread_bps",
    "borrowing_cost_index",
    "withdrawal_amount_m",
    "withdrawal_velocity_ma_4",
    "total_margin_calls_m",
    "vm_calls_m",
    "im_calls_m",
    "net_outflow_pressure_m",
    "liquidity_buffer_m",
]

TARGET_COLUMN = "target_liquidity_48h_m"


@dataclass
class RandomForestResult:
    model: RandomForestRegressor
    currency_encoder: LabelEncoder
    feature_importance: pd.DataFrame
    test_r2: float


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target, dropping rows with NaN target."""
    work = df.dropna(subset=[TARGET_COLUMN]).copy()
    encoder = LabelEncoder()
    work["currency_code"] = encoder.fit_transform(work["currency"])

    feature_cols = FEATURE_COLUMNS + ["currency_code"]
    X = work[feature_cols]
    y = work[TARGET_COLUMN]
    return X, y


def train_random_forest(
    df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 12,
    test_size: float = 0.2,
    seed: int = 42,
) -> RandomForestResult:
    """Train Random Forest on historical liquidity features."""
    X, y = prepare_features(df)
    currency_encoder = LabelEncoder()
    currency_encoder.fit(df["currency"].unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return RandomForestResult(
        model=model,
        currency_encoder=currency_encoder,
        feature_importance=importance,
        test_r2=test_r2,
    )


def predict_random_forest(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    currency_encoder: LabelEncoder,
) -> np.ndarray:
    """Generate RF predictions for current feature rows."""
    work = df.copy()
    work["currency_code"] = currency_encoder.transform(work["currency"])
    feature_cols = FEATURE_COLUMNS + ["currency_code"]
    return model.predict(work[feature_cols])


def save_random_forest(result: RandomForestResult, path: str) -> None:
    joblib.dump(
        {
            "model": result.model,
            "currency_encoder": result.currency_encoder,
            "feature_importance": result.feature_importance,
            "test_r2": result.test_r2,
        },
        path,
    )


def load_random_forest(path: str) -> RandomForestResult:
    data = joblib.load(path)
    return RandomForestResult(**data)
