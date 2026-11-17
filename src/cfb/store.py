"""Local storage: raw JSON cache (source of truth) plus flattened Parquet tables.

Layout under the data directory:
    raw/<dataset>/<key>.json.gz       exact API responses
    tables/<dataset>/<key>.parquet    pandas.json_normalize of the same response
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


class Store:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)

    def raw_path(self, dataset: str, key: str) -> Path:
        return self.root / "raw" / dataset / f"{key}.json.gz"

    def table_path(self, dataset: str, key: str) -> Path:
        return self.root / "tables" / dataset / f"{key}.parquet"

    def has(self, dataset: str, key: str) -> bool:
        return self.raw_path(dataset, key).exists()

    def write(self, dataset: str, key: str, payload: Any, extra: dict | None = None) -> int:
        """Save the raw payload and its flattened table. Returns the row count."""
        raw = self.raw_path(dataset, key)
        raw.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(raw)

        frame = flatten(payload, extra)
        out = self.table_path(dataset, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        return len(frame)

    def read_raw(self, dataset: str, key: str) -> Any:
        with gzip.open(self.raw_path(dataset, key), "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, dataset: str) -> pd.DataFrame:
        """Concatenate every Parquet partition of a dataset."""
        files = sorted((self.root / "tables" / dataset).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No tables for dataset '{dataset}' under {self.root}")
        frames = [f for f in (pd.read_parquet(p) for p in files) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flatten(payload: Any, extra: dict | None = None) -> pd.DataFrame:
    """Flatten a JSON response into one row per top-level record.

    Nested objects become dotted columns. Lists inside a record are kept as
    JSON strings so no information is dropped and Parquet types stay stable.
    `extra` adds constant columns (for example the season requested).
    """
    records = payload if isinstance(payload, list) else [payload]
    frame = pd.json_normalize(records) if records else pd.DataFrame()
    for col in frame.columns:
        if frame[col].map(lambda v: isinstance(v, (list, dict))).any():
            frame[col] = frame[col].map(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
            )
    for name, value in (extra or {}).items():
        if name not in frame.columns:
            frame[name] = value
    return frame
