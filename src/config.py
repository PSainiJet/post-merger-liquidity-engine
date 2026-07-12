"""Load and expose project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / relative
