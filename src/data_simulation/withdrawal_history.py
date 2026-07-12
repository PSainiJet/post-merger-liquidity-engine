"""Corporate withdrawal history simulation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Corporate client segments with different withdrawal behavior
SEGMENTS = {
    "institutional": {"share": 0.35, "run_sensitivity": 1.8},
    "wealth_management": {"share": 0.30, "run_sensitivity": 2.2},
    "corporate_treasury": {"share": 0.20, "run_sensitivity": 1.5},
    "retail_corporate": {"share": 0.15, "run_sensitivity": 3.0},
}


def generate_withdrawal_history(
    start: datetime,
    days: int,
    bucket_minutes: int,
    currencies: list[str],
    seed: int = 45,
) -> pd.DataFrame:
    """Simulate corporate withdrawal flows by segment and currency."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start=start,
        end=start + timedelta(days=days),
        freq=f"{bucket_minutes}min",
        inclusive="left",
    )

    base_withdrawal_m = {"CHF": 18, "USD": 28, "EUR": 22}

    rows = []
    for ts in timestamps:
        # Intraday pattern: spikes at market open and close
        hour = ts.hour + ts.minute / 60
        intraday_multiplier = 1.0
        if 8 <= hour <= 10 or 15 <= hour <= 17:
            intraday_multiplier = 1.6
        elif hour < 6 or hour > 20:
            intraday_multiplier = 0.4

        for ccy in currencies:
            for segment, props in SEGMENTS.items():
                base = base_withdrawal_m[ccy] * props["share"]
                noise = rng.lognormal(0, 0.15)
                amount = base * intraday_multiplier * noise
                rows.append(
                    {
                        "timestamp": ts,
                        "currency": ccy,
                        "segment": segment,
                        "withdrawal_amount_m": round(amount, 3),
                        "run_sensitivity": props["run_sensitivity"],
                        "cumulative_daily_m": 0.0,  # filled below
                    }
                )

    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.date
    df["cumulative_daily_m"] = df.groupby(["date", "currency", "segment"])[
        "withdrawal_amount_m"
    ].cumsum()
    df = df.drop(columns=["date"])
    return df


def compute_withdrawal_velocity(withdrawals_df: pd.DataFrame) -> pd.DataFrame:
    """Rolling withdrawal velocity (rate of change) per currency."""
    agg = (
        withdrawals_df.groupby(["timestamp", "currency"], as_index=False)["withdrawal_amount_m"]
        .sum()
        .sort_values(["currency", "timestamp"])
    )
    agg["withdrawal_velocity"] = agg.groupby("currency")["withdrawal_amount_m"].diff().fillna(0)
    agg["withdrawal_velocity_ma_4"] = (
        agg.groupby("currency")["withdrawal_velocity"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    return agg
