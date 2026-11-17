"""Run configuration: which model the report is about, which baselines sit next to it, and
how each model's predictions are produced. Read from `dashboard/config.toml`."""
from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..core.schemas import MODEL_ID, SchemaError

DEFAULT_CONFIG = Path("dashboard") / "config.toml"
BUILTIN_BASELINES = ("baseline_close", "baseline_moneyline", "baseline_elo", "baseline_home")

Producer = Callable[[object, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ReportConfig:
    primary: str = "baseline_elo"
    baselines: tuple[str, ...] = BUILTIN_BASELINES
    producers: dict[str, str] = field(default_factory=dict)
    build_features: bool = True
    market_models: tuple[str, ...] = ("baseline_moneyline", "baseline_close")
    timezone: str = "America/New_York"
    # Model rankings vs polls (cfb.report.rankings): a built-in rating source ("power", "elo")
    # or "module:function"; an empty source turns the section off.
    rankings_source: str = "power"
    rankings_top: int = 50

    def __post_init__(self):
        for m in (self.primary, *self.baselines, *self.producers, *self.market_models):
            if not MODEL_ID.fullmatch(m):
                raise SchemaError(f"invalid model_id {m!r} in the run configuration")
        for m, spec in self.producers.items():
            if ":" not in spec:
                raise ValueError(f"producer for {m} must be 'module:function', got {spec!r}")
        if self.rankings_top < 1:
            raise ValueError(f"rankings top must be at least 1, got {self.rankings_top}")

    @property
    def models(self) -> list[str]:
        """Primary first, then baselines and market models, without repeats."""
        out: list[str] = []
        for m in (self.primary, *self.baselines, *self.market_models):
            if m not in out:
                out.append(m)
        return out

    @property
    def comparison(self) -> list[str]:
        """Models scored against the primary one."""
        return [m for m in self.baselines if m != self.primary]

    def producer(self, model_id: str) -> Producer | None:
        """The function that predicts `model_id`, or None to read stored predictions."""
        spec = self.producers.get(model_id)
        if spec:
            module, _, name = spec.partition(":")
            return getattr(importlib.import_module(module), name)
        if model_id in BUILTIN_BASELINES:
            from ..eval.baselines import run_baseline

            def produce(warehouse, games, _m=model_id):
                return run_baseline(warehouse, _m, sorted(games["season"].unique()), games=games)

            return produce
        return None


def load_config(path: str | Path | None = None, **overrides) -> ReportConfig:
    """Read the TOML file (missing file: defaults) and apply keyword overrides."""
    values: dict = {}
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if p.exists():
        doc = tomllib.loads(p.read_text(encoding="utf-8"))
        models, run = doc.get("models", {}), doc.get("run", {})
        if "primary" in models:
            values["primary"] = str(models["primary"])
        if "baselines" in models:
            values["baselines"] = tuple(models["baselines"])
        values["producers"] = {str(k): str(v) for k, v in doc.get("producers", {}).items()}
        for key in ("build_features", "timezone"):
            if key in run:
                values[key] = run[key]
        if "market_models" in run:
            values["market_models"] = tuple(run["market_models"])
        rk = doc.get("rankings", {})
        if "source" in rk:
            values["rankings_source"] = str(rk["source"])
        if "top" in rk:
            values["rankings_top"] = int(rk["top"])
    elif path is not None:
        raise FileNotFoundError(p)
    values.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("baselines", "market_models"):
        if key in values:
            values[key] = tuple(values[key])
    return ReportConfig(**values)
