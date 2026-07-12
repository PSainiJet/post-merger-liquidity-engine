"""Train the ensemble liquidity prediction model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_simulation.pipeline import load_simulated_data
from src.models.ensemble import save_ensemble, train_ensemble


def main() -> None:
    config = load_config()
    data_dir = PROJECT_ROOT / config["paths"]["simulated_data"]
    artifacts_dir = PROJECT_ROOT / config["paths"]["model_artifacts"]

    master_path = data_dir / "master_features.parquet"
    if not master_path.exists():
        print("Simulated data not found. Run: python scripts/generate_simulated_data.py")
        sys.exit(1)

    print("Loading simulated data...")
    data = load_simulated_data(data_dir)
    master_df = data["master_features"]

    print(f"Training ensemble on {len(master_df):,} rows...")
    ensemble = train_ensemble(master_df, config)
    save_ensemble(ensemble, artifacts_dir)

    print(f"  Random Forest R²: {ensemble.random_forest.test_r2:.4f}")
    print(f"  Time-series AIC scores: {ensemble.time_series.aic_scores}")
    print(f"  Artifacts saved to: {artifacts_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
