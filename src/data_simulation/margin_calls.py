"""Derivative margin call simulation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


DESK_EXPOSURE = {
    "rates_derivatives": {"CHF": 120, "USD": 340, "EUR": 210},
    "fx_derivatives": {"CHF": 85, "USD": 520, "EUR": 190},
    "equity_derivatives": {"CHF": 45, "USD": 280, "EUR": 95},
    "credit_derivatives": {"CHF": 60, "USD": 410, "EUR": 175},
}


def generate_margin_calls(
    start: datetime,
    days: int,
    bucket_minutes: int,
    currencies: list[str],
    seed: int = 46,
) -> pd.DataFrame:
    """Simulate intraday derivative margin call obligations."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start=start,
        end=start + timedelta(days=days),
        freq=f"{bucket_minutes}min",
        inclusive="left",
    )

    rows = []
    for ts in timestamps:
        # Margin calls cluster around market events
        event_multiplier = 1.0
        if ts.hour in (9, 10, 14, 15) and rng.random() < 0.15:
            event_multiplier = rng.uniform(2.0, 4.5)

        for desk, exposures in DESK_EXPOSURE.items():
            for ccy in currencies:
                base = exposures[ccy] / 96  # spread across ~96 buckets/day
                spike = rng.exponential(base) * event_multiplier
                rows.append(
                    {
                        "timestamp": ts,
                        "currency": ccy,
                        "desk": desk,
                        "margin_call_m": round(spike, 3),
                        "is_vm_call": rng.random() > 0.3,  # variation margin vs IM
                    }
                )

    df = pd.DataFrame(rows)
    vm = (
        df[df["is_vm_call"]]
        .groupby(["timestamp", "currency"], as_index=False)["margin_call_m"]
        .sum()
        .rename(columns={"margin_call_m": "vm_calls_m"})
    )
    im = (
        df[~df["is_vm_call"]]
        .groupby(["timestamp", "currency"], as_index=False)["margin_call_m"]
        .sum()
        .rename(columns={"margin_call_m": "im_calls_m"})
    )
    agg = (
        df.groupby(["timestamp", "currency"], as_index=False)["margin_call_m"]
        .sum()
        .rename(columns={"margin_call_m": "total_margin_calls_m"})
    )
    agg = agg.merge(vm, on=["timestamp", "currency"], how="left")
    agg = agg.merge(im, on=["timestamp", "currency"], how="left")
    agg["vm_calls_m"] = agg["vm_calls_m"].fillna(0)
    agg["im_calls_m"] = agg["im_calls_m"].fillna(0)
    return agg


def apply_margin_stress(
    margin_df: pd.DataFrame,
    spike_pct: float,
    from_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply stress scenario margin call spike."""
    df = margin_df.copy()
    if from_timestamp is None:
        from_timestamp = df["timestamp"].iloc[len(df) // 2]
    mask = df["timestamp"] >= from_timestamp
    multiplier = 1 + spike_pct / 100
    for col in ["total_margin_calls_m", "vm_calls_m", "im_calls_m"]:
        df.loc[mask, col] *= multiplier
    return df
