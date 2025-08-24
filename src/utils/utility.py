import yaml
from typing import Any


def load_yaml(file_path: str) -> Any:
    """Load a YAML file and return the parsed data."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
