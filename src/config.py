import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config.yaml"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    # resolve relative paths to absolute
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = _ROOT / rel
    return cfg


CONFIG = load_config()
