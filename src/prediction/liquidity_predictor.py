"""Core liquidity prediction engine for Treasury operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data_simulation.pipeline import generate_all_streams, load_simulated_data
from src.models.ensemble import (
    EnsembleResult,
    load_ensemble,
    predict_ensemble_liquidity,
    train_ensemble,
)
from src.stress_testing.bank_run_scenarios import run_all_stress_scenarios


@dataclass
class LiquidityReport:
    generated_at: datetime
    horizon_hours: int
    currency_forecasts: pd.DataFrame
    at_risk_currencies: list[str]
    recommended_actions: list[str]


def _build_recommendations(forecast: pd.DataFrame) -> list[str]:
    actions = []
    for _, row in forecast.iterrows():
        if row["alert_level"] == "CRITICAL":
            actions.append(
                f"URGENT: Borrow or liquidate HQLA for {row['currency']} — "
                f"projected shortfall of {row['shortfall_m']:.0f}M in 48h"
            )
        elif row["alert_level"] == "WARNING":
            actions.append(
                f"MONITOR: Pre-position liquidity for {row['currency']} — "
                f"buffer at {row['buffer_pct']:.1f}% (below warning threshold)"
            )
    if not actions:
        actions.append("All currencies within acceptable liquidity buffers for the next 48 hours.")
    return actions


def generate_liquidity_report(
    ensemble: EnsembleResult,
    master_df: pd.DataFrame,
    config: dict | None = None,
) -> LiquidityReport:
    """Generate a 48-hour liquidity risk report for the Treasury desk."""
    cfg = config or load_config()
    horizon = cfg["project"]["horizon_hours"]
    bucket = cfg["data_simulation"]["intraday_buckets_minutes"]

    forecast = predict_ensemble_liquidity(ensemble, master_df, horizon, bucket)
    at_risk = forecast[forecast["alert_level"].isin(["WARNING", "CRITICAL"])]["currency"].tolist()

    return LiquidityReport(
        generated_at=datetime.now(UTC),
        horizon_hours=horizon,
        currency_forecasts=forecast,
        at_risk_currencies=at_risk,
        recommended_actions=_build_recommendations(forecast),
    )


class LiquidityEngine:
    """Main orchestrator: load data, train/load models, produce reports."""

    def __init__(self, config: dict | None = None, project_root: Path | None = None):
        self.config = config or load_config()
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_root / self.config["paths"]["simulated_data"]
        self.artifacts_dir = self.project_root / self.config["paths"]["model_artifacts"]
        self.ensemble: EnsembleResult | None = None
        self.master_df: pd.DataFrame | None = None

    def initialize_data(self, regenerate: bool = False) -> pd.DataFrame:
        if regenerate or not (self.data_dir / "master_features.parquet").exists():
            datasets = generate_all_streams(self.config)
            from src.data_simulation.pipeline import save_simulated_data

            save_simulated_data(self.data_dir, datasets)
            self.master_df = datasets["master_features"]
        else:
            data = load_simulated_data(self.data_dir)
            self.master_df = data["master_features"]
        return self.master_df

    def train(self) -> EnsembleResult:
        if self.master_df is None:
            self.initialize_data()
        self.ensemble = train_ensemble(self.master_df, self.config)
        from src.models.ensemble import save_ensemble

        save_ensemble(self.ensemble, self.artifacts_dir)
        return self.ensemble

    def load_models(self) -> EnsembleResult:
        from src.models.ensemble import load_ensemble

        self.ensemble = load_ensemble(self.artifacts_dir, self.config)
        return self.ensemble

    def get_report(self) -> LiquidityReport:
        if self.ensemble is None:
            if (self.artifacts_dir / "random_forest.joblib").exists():
                self.load_models()
            else:
                self.train()
        if self.master_df is None:
            self.initialize_data()
        return generate_liquidity_report(self.ensemble, self.master_df, self.config)

    def run_stress_tests(self) -> dict:
        if self.ensemble is None:
            self.load_models() if (self.artifacts_dir / "random_forest.joblib").exists() else self.train()
        if self.master_df is None:
            self.initialize_data()
        return run_all_stress_scenarios(self.master_df, self.ensemble, self.config)
