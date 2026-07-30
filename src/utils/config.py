"""Configuration loader utility for reading project settings."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Loads configuration dictionary from YAML file.

    Args:
        config_path (Optional[str]): Path to configuration YAML file.
            Defaults to configs/config.yaml.

    Returns:
        Dict[str, Any]: Loaded configuration options.
    """
    if config_path is None:
        # Default relative to root
        root_dir = Path(__file__).resolve().parent.parent.parent
        config_path = str(root_dir / "configs" / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
