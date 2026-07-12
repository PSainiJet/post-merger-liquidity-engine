"""Launch the Treasury liquidity dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.app import run_dashboard


def main() -> None:
    run_dashboard(project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
