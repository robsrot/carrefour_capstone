from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten(value, dotted_key))
        else:
            flattened[dotted_key] = value
    return flattened


def test_config_folder_contains_only_authoritative_mode_files():
    assert sorted(path.name for path in CONFIG_DIR.glob("*.y*ml")) == [
        "base.yaml",
        "dev.yaml",
        "prod.yaml",
    ]


def test_mode_configs_only_override_real_differences():
    base = _flatten(_read_yaml(CONFIG_DIR / "base.yaml"))
    failures: list[str] = []

    for mode_file in ["dev.yaml", "prod.yaml"]:
        overrides = _flatten(_read_yaml(CONFIG_DIR / mode_file))
        for key, value in sorted(overrides.items()):
            if key not in base:
                failures.append(f"{mode_file}: unknown override key {key}")
            elif base[key] == value:
                failures.append(f"{mode_file}: redundant override {key}")

    assert failures == []


def test_stage1_config_has_no_legacy_second_artifact_path():
    base = _read_yaml(CONFIG_DIR / "base.yaml")
    downsampling = base["baskets"]["downsampling"]

    assert "output" not in downsampling
    assert "summary_md" not in downsampling
