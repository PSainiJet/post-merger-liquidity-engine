"""Simulated interbank lending rate streams."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Approximate baseline overnight rates (annualized %)
BASE_RATES = {"CHF": 1.25, "USD": 5.15, "EUR": 3.65}


def generate_interbank_rates(
    start: datetime,
    days: int,
    bucket_minutes: int,
    currencies: list[str],
    seed: int = 44,
) -> pd.DataFrame:
    """Generate intraday interbank lending rates with spread dynamics."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start=start,
        end=start + timedelta(days=days),
        freq=f"{bucket_minutes}min",
        inclusive="left",
    )

    rows = []
    for ccy in currencies:
        base = BASE_RATES[ccy]
        spread_bps = 12.0  # starting spread
        for ts in timestamps:
            # Random walk on spread — widens under stress
            spread_bps += rng.normal(0, 0.5)
            spread_bps = float(np.clip(spread_bps, 5, 250))
            rate = base + spread_bps / 100
            rows.append(
                {
                    "timestamp": ts,
                    "currency": ccy,
                    "overnight_rate_pct": round(rate, 4),
                    "interbank_spread_bps": round(spread_bps, 2),
                    "borrowing_cost_index": round(rate * (1 + spread_bps / 1000), 4),
                }
            )
    return pd.DataFrame(rows)


def apply_stress_to_rates(
    rates_df: pd.DataFrame,
    spread_widen_bps: float,
    from_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Widen interbank spreads from a given point forward (stress scenario)."""
    df = rates_df.copy()
    if from_timestamp is None:
        from_timestamp = df["timestamp"].iloc[len(df) // 2]

    mask = df["timestamp"] >= from_timestamp
    df.loc[mask, "interbank_spread_bps"] += spread_widen_bps
    df.loc[mask, "overnight_rate_pct"] += spread_widen_bps / 100
    df.loc[mask, "borrowing_cost_index"] = (
        df.loc[mask, "overnight_rate_pct"]
        * (1 + df.loc[mask, "interbank_spread_bps"] / 1000)
    )
    return df
