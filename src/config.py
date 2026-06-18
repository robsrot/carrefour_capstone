"""Central configuration for the Carrefour tribe discovery pipeline."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _current_mode(base: dict[str, Any]) -> str:
    run_cfg = base.get("run", {})
    env_name = run_cfg.get("mode_env", "CARREFOUR_MODE")
    default = run_cfg.get("default_mode", "prod")
    mode = os.getenv(env_name, default).strip().lower()
    if mode not in {"dev", "prod"}:
        raise ValueError(f"{env_name} must be 'dev' or 'prod', got {mode!r}")
    return mode


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved config plus mode-aware project paths."""

    values: dict[str, Any]
    mode: str
    root: Path = ROOT

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def random_seed(self) -> int:
        return int(self.get("run.random_seed", 42))

    @property
    def data_processed(self) -> Path:
        key = "paths.dev" if self.mode == "dev" else "paths.processed"
        return self.root / self.get(key)

    @property
    def data_prod(self) -> Path:
        return self.root / self.get("paths.processed")

    @property
    def raw_parquet(self) -> Path:
        return self.root / self.get("paths.raw_parquet")

    @property
    def raw_csv(self) -> Path:
        return self.root / self.get("paths.raw_csv")

    @property
    def models(self) -> Path:
        return self.outputs / "models"

    @property
    def outputs(self) -> Path:
        return self.root / self.get("paths.outputs") / self.mode

    @property
    def experiments(self) -> Path:
        return self.outputs / "experiments"

    @property
    def experiments_enabled(self) -> bool:
        return bool(self.get("experiments.enabled", self.mode == "dev"))

    @property
    def reports(self) -> Path:
        return self.outputs / "reports"

    @property
    def artifacts(self) -> Path:
        return self.outputs / "artifacts"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def model_selection(self) -> Path:
        return self.artifacts / "model_selection"

    @property
    def model_selection_cache(self) -> Path:
        return self.models / "model_selection"

    @property
    def prepared_transactions_path(self) -> Path:
        return self.data_processed / self.get("data.prepared_transactions")

    @property
    def customer_kpis_path(self) -> Path:
        mode_path = self.data_processed / self.get("data.customer_kpis")
        if mode_path.exists():
            return mode_path
        return self.data_prod / self.get("data.customer_kpis")

    @property
    def product_master_path(self) -> Path:
        return self.raw_parquet / self.get("data.product_master")

    def artifact_path(self, section: str, key: str, directory: Path | None = None) -> Path:
        name = self.get(f"{section}.{key}")
        if not name:
            raise KeyError(f"Missing config key: {section}.{key}")
        return (directory or self.data_processed) / str(name).format(mode=self.mode)

    def ensure_directories(self) -> None:
        paths = [
            self.data_processed,
            self.models,
            self.outputs,
            self.reports,
            self.artifacts,
            self.figures,
            self.outputs / "embeddings",
            self.outputs / "features",
            self.outputs / "profiles",
            self.model_selection,
            self.model_selection_cache,
            self.artifacts / "stage7",
            self.figures / "tribe_lifts",
        ]
        if self.experiments_enabled:
            paths.append(self.experiments)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


def load_config(mode: str | None = None) -> PipelineConfig:
    base = _read_yaml(CONFIG_DIR / "base.yaml")
    resolved_mode = mode.strip().lower() if mode else _current_mode(base)
    if resolved_mode not in {"dev", "prod"}:
        raise ValueError(f"mode must be 'dev' or 'prod', got {resolved_mode!r}")

    override_path = CONFIG_DIR / f"{resolved_mode}.yaml"
    values = _deep_merge(base, _read_yaml(override_path)) if override_path.exists() else base
    return PipelineConfig(values=values, mode=resolved_mode)


CONFIG = load_config()
MODE = CONFIG.mode
RUN_MODE = CONFIG.mode
RANDOM_SEED = CONFIG.random_seed
MIN_TICKETS_PER_CUSTOMER = int(CONFIG.get("data.min_tickets_per_customer", 3))

DATA_PROCESSED = CONFIG.data_processed
DATA_PROD = CONFIG.data_prod
RAW_PARQUET = CONFIG.raw_parquet
RAW_CSV = CONFIG.raw_csv
MODELS = CONFIG.models
OUTPUTS = CONFIG.outputs
REPORTS = CONFIG.reports
ARTIFACTS = CONFIG.artifacts
FIGURES = CONFIG.figures
MODEL_SELECTION_CACHE = CONFIG.model_selection_cache

PREPARED_TRANSACTIONS = CONFIG.prepared_transactions_path
CUSTOMER_KPIS = CONFIG.customer_kpis_path
PRODUCT_MASTER = CONFIG.product_master_path


def configure_mode(mode: str) -> PipelineConfig:
    """Make an explicit mode authoritative for notebooks and interactive runs."""

    global CONFIG
    global MODE
    global RUN_MODE
    global RANDOM_SEED
    global MIN_TICKETS_PER_CUSTOMER
    global DATA_PROCESSED
    global DATA_PROD
    global RAW_PARQUET
    global RAW_CSV
    global MODELS
    global OUTPUTS
    global REPORTS
    global ARTIFACTS
    global FIGURES
    global MODEL_SELECTION_CACHE
    global PREPARED_TRANSACTIONS
    global CUSTOMER_KPIS
    global PRODUCT_MASTER

    cfg = load_config(mode)
    os.environ[str(cfg.get("run.mode_env", "CARREFOUR_MODE"))] = cfg.mode

    CONFIG = cfg
    MODE = cfg.mode
    RUN_MODE = cfg.mode
    RANDOM_SEED = cfg.random_seed
    MIN_TICKETS_PER_CUSTOMER = int(cfg.get("data.min_tickets_per_customer", 3))

    DATA_PROCESSED = cfg.data_processed
    DATA_PROD = cfg.data_prod
    RAW_PARQUET = cfg.raw_parquet
    RAW_CSV = cfg.raw_csv
    MODELS = cfg.models
    OUTPUTS = cfg.outputs
    REPORTS = cfg.reports
    ARTIFACTS = cfg.artifacts
    FIGURES = cfg.figures
    MODEL_SELECTION_CACHE = cfg.model_selection_cache

    PREPARED_TRANSACTIONS = cfg.prepared_transactions_path
    CUSTOMER_KPIS = cfg.customer_kpis_path
    PRODUCT_MASTER = cfg.product_master_path
    return cfg
