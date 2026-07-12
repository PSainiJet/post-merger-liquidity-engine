"""Bank-run stress testing scenarios for liquidity drain simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import load_config
from src.data_simulation.interbank_rates import apply_stress_to_rates
from src.data_simulation.margin_calls import apply_margin_stress
from src.data_simulation.withdrawal_history import compute_withdrawal_velocity
from src.models.ensemble import EnsembleResult, predict_ensemble_liquidity


@dataclass
class StressScenarioResult:
    scenario_name: str
    baseline_forecast: pd.DataFrame
    stressed_forecast: pd.DataFrame
    liquidity_impact: pd.DataFrame


def apply_withdrawal_stress(
    df: pd.DataFrame,
    velocity_multiplier: float,
    from_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Accelerate withdrawal velocity under bank-run conditions."""
    stressed = df.copy()
    if from_timestamp is None:
        from_timestamp = stressed["timestamp"].iloc[len(stressed) // 2]

    mask = stressed["timestamp"] >= from_timestamp
    for col in ["withdrawal_amount_m", "withdrawal_velocity_ma_4", "net_outflow_pressure_m"]:
        if col in stressed.columns:
            stressed.loc[mask, col] *= velocity_multiplier

    stressed.loc[mask, "liquidity_buffer_m"] -= (
        stressed.loc[mask, "net_outflow_pressure_m"] * (velocity_multiplier - 1)
    )
    return stressed


def run_stress_scenario(
    master_df: pd.DataFrame,
    ensemble: EnsembleResult,
    scenario_name: str,
    config: dict | None = None,
) -> StressScenarioResult:
    """Run a named stress scenario and compare baseline vs stressed forecasts."""
    cfg = config or load_config()
    scenario = cfg["stress_scenarios"][scenario_name]
    horizon = cfg["project"]["horizon_hours"]
    bucket = cfg["data_simulation"]["intraday_buckets_minutes"]

    baseline = predict_ensemble_liquidity(ensemble, master_df, horizon, bucket)

    # Apply stress from latest timestamp
    stress_start = master_df["timestamp"].max() - pd.Timedelta(hours=6)
    stressed_df = apply_withdrawal_stress(
        master_df,
        scenario["withdrawal_velocity_multiplier"],
        stress_start,
    )

    if "interbank_rates" not in stressed_df.columns:
        pass  # rates already merged in master

    stressed_df = apply_margin_stress(
        stressed_df,
        scenario["margin_call_spike_pct"],
        stress_start,
    )

    # Widen spreads on rate columns if present
    if "interbank_spread_bps" in stressed_df.columns:
        mask = stressed_df["timestamp"] >= stress_start
        stressed_df.loc[mask, "interbank_spread_bps"] += scenario["interbank_spread_widen_bps"]
        stressed_df.loc[mask, "borrowing_cost_index"] *= (
            1 + scenario["interbank_spread_widen_bps"] / 10000
        )

    stressed = predict_ensemble_liquidity(ensemble, stressed_df, horizon, bucket)

    impact = baseline.merge(
        stressed,
        on="currency",
        suffixes=("_baseline", "_stressed"),
    )
    impact["liquidity_drop_m"] = (
        impact["predicted_liquidity_48h_m_baseline"]
        - impact["predicted_liquidity_48h_m_stressed"]
    )
    impact["new_alert"] = impact["alert_level_stressed"]

    return StressScenarioResult(
        scenario_name=scenario_name,
        baseline_forecast=baseline,
        stressed_forecast=stressed,
        liquidity_impact=impact,
    )


def run_all_stress_scenarios(
    master_df: pd.DataFrame,
    ensemble: EnsembleResult,
    config: dict | None = None,
) -> dict[str, StressScenarioResult]:
    """Execute all configured stress scenarios."""
    cfg = config or load_config()
    results = {}
    for name in cfg["stress_scenarios"]:
        results[name] = run_stress_scenario(master_df, ensemble, name, cfg)
    return results
