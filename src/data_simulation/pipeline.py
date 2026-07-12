"""Orchestrates generation of all simulated data streams."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data_simulation.interbank_rates import generate_interbank_rates
from src.data_simulation.margin_calls import generate_margin_calls
from src.data_simulation.system_feeds import (
    LegacyCreditSuisseFeed,
    UBSFeed,
    build_intraday_timestamps,
    combine_system_feeds,
)
from src.data_simulation.withdrawal_history import (
    compute_withdrawal_velocity,
    generate_withdrawal_history,
)


def generate_all_streams(
    config: dict | None = None,
    start: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate consolidated dataset from all simulated feeds."""
    cfg = config or load_config()
    sim_cfg = cfg["data_simulation"]
    currencies = cfg["project"]["currencies"]

    if start is None:
        start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(
            days=sim_cfg["history_days"]
        )

    days = sim_cfg["history_days"]
    bucket = sim_cfg["intraday_buckets_minutes"]
    seed = sim_cfg["seed"]

    cs_feed = LegacyCreditSuisseFeed(seed=seed)
    ubs_feed = UBSFeed(seed=seed + 1)
    timestamps = build_intraday_timestamps(start, days, bucket)

    cs_positions = cs_feed.generate_positions(timestamps, currencies)
    ubs_positions = ubs_feed.generate_positions(timestamps, currencies)
    consolidated = combine_system_feeds(cs_positions, ubs_positions)

    rates = generate_interbank_rates(start, days, bucket, currencies, seed=seed + 2)
    withdrawals = generate_withdrawal_history(start, days, bucket, currencies, seed=seed + 3)
    withdrawal_velocity = compute_withdrawal_velocity(withdrawals)
    margins = generate_margin_calls(start, days, bucket, currencies, seed=seed + 4)

    master = consolidated.merge(rates, on=["timestamp", "currency"], how="left")
    master = master.merge(
        withdrawal_velocity[["timestamp", "currency", "withdrawal_amount_m", "withdrawal_velocity_ma_4"]],
        on=["timestamp", "currency"],
        how="left",
    )
    master = master.merge(margins, on=["timestamp", "currency"], how="left")
    master = master.fillna(0)

    # Target: net liquidity position 48h ahead (simplified proxy for training)
    horizon_steps = int(48 * 60 / bucket)
    master["net_outflow_pressure_m"] = (
        master["total_hourly_outflow_m"]
        + master["withdrawal_amount_m"]
        + master["total_margin_calls_m"]
    )
    master["liquidity_buffer_m"] = (
        master["total_liquid_assets_m"] - master["total_pending_settlements_m"]
    )
    master["target_liquidity_48h_m"] = (
        master.groupby("currency")["liquidity_buffer_m"]
        .shift(-horizon_steps)
    )

    return {
        "consolidated_positions": consolidated,
        "cs_legacy": cs_positions,
        "ubs_core": ubs_positions,
        "interbank_rates": rates,
        "withdrawals": withdrawals,
        "margin_calls": margins,
        "master_features": master,
    }


def save_simulated_data(output_dir: Path, datasets: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)


def load_simulated_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    datasets = {}
    for path in input_dir.glob("*.parquet"):
        datasets[path.stem] = pd.read_parquet(path)
    return datasets
