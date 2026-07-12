"""Generate simulated data streams for all feeds."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_simulation.pipeline import generate_all_streams, save_simulated_data


def main() -> None:
    config = load_config()
    output_dir = PROJECT_ROOT / config["paths"]["simulated_data"]

    print("Generating simulated liquidity data streams...")
    datasets = generate_all_streams(config)
    save_simulated_data(output_dir, datasets)

    for name, df in datasets.items():
        print(f"  {name}: {len(df):,} rows -> {output_dir / f'{name}.parquet'}")

    print("Done.")


if __name__ == "__main__":
    main()
