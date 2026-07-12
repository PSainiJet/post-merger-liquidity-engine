"""Simulated data streams from legacy Credit Suisse and UBS core systems."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class SystemFeedConfig:
    system_id: str
    base_liquid_assets_m: dict[str, float]
    daily_outflow_mean_m: dict[str, float]
    volatility: float = 0.08


class LegacyCreditSuisseFeed:
    """Simulates residual Credit Suisse legacy infrastructure cash positions."""

    def __init__(self, seed: int = 42):
        self.config = SystemFeedConfig(
            system_id="CS_LEGACY",
            base_liquid_assets_m={"CHF": 2800, "USD": 1900, "EUR": 1600},
            daily_outflow_mean_m={"CHF": 420, "USD": 380, "EUR": 310},
            volatility=0.12,  # Higher volatility — legacy infra less integrated
        )
        self.rng = np.random.default_rng(seed)

    def generate_positions(
        self, timestamps: pd.DatetimeIndex, currencies: list[str]
    ) -> pd.DataFrame:
        rows = []
        for ts in timestamps:
            for ccy in currencies:
                hour_factor = 1.0 + 0.15 * np.sin(2 * np.pi * ts.hour / 24)
                noise = self.rng.normal(0, self.config.volatility)
                liquid = self.config.base_liquid_assets_m[ccy] * (1 + noise) * hour_factor
                outflow_rate = (
                    self.config.daily_outflow_mean_m[ccy]
                    / 24
                    * (1 + self.rng.normal(0, 0.05))
                )
                rows.append(
                    {
                        "timestamp": ts,
                        "system_id": self.config.system_id,
                        "currency": ccy,
                        "liquid_assets_m": max(liquid, 0),
                        "hourly_outflow_m": max(outflow_rate, 0),
                        "pending_settlements_m": self.rng.uniform(50, 200),
                    }
                )
        return pd.DataFrame(rows)


class UBSFeed:
    """Simulates UBS core treasury liquidity positions."""

    def __init__(self, seed: int = 43):
        self.config = SystemFeedConfig(
            system_id="UBS_CORE",
            base_liquid_assets_m={"CHF": 7200, "USD": 9800, "EUR": 6400},
            daily_outflow_mean_m={"CHF": 890, "USD": 1200, "EUR": 780},
            volatility=0.06,
        )
        self.rng = np.random.default_rng(seed)

    def generate_positions(
        self, timestamps: pd.DatetimeIndex, currencies: list[str]
    ) -> pd.DataFrame:
        rows = []
        for ts in timestamps:
            for ccy in currencies:
                hour_factor = 1.0 + 0.10 * np.sin(2 * np.pi * ts.hour / 24)
                noise = self.rng.normal(0, self.config.volatility)
                liquid = self.config.base_liquid_assets_m[ccy] * (1 + noise) * hour_factor
                outflow_rate = (
                    self.config.daily_outflow_mean_m[ccy]
                    / 24
                    * (1 + self.rng.normal(0, 0.04))
                )
                rows.append(
                    {
                        "timestamp": ts,
                        "system_id": self.config.system_id,
                        "currency": ccy,
                        "liquid_assets_m": max(liquid, 0),
                        "hourly_outflow_m": max(outflow_rate, 0),
                        "pending_settlements_m": self.rng.uniform(80, 350),
                    }
                )
        return pd.DataFrame(rows)


def build_intraday_timestamps(
    start: datetime, days: int, bucket_minutes: int
) -> pd.DatetimeIndex:
    end = start + timedelta(days=days)
    return pd.date_range(start=start, end=end, freq=f"{bucket_minutes}min", inclusive="left")


def combine_system_feeds(
    cs_df: pd.DataFrame, ubs_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate legacy + core positions into consolidated treasury view."""
    combined = pd.concat([cs_df, ubs_df], ignore_index=True)
    agg = (
        combined.groupby(["timestamp", "currency"], as_index=False)
        .agg(
            total_liquid_assets_m=("liquid_assets_m", "sum"),
            total_hourly_outflow_m=("hourly_outflow_m", "sum"),
            total_pending_settlements_m=("pending_settlements_m", "sum"),
            source_systems=("system_id", lambda x: ",".join(sorted(set(x)))),
        )
    )
    return agg
