"""File contracts for predictions and metrics.

Predictions: data/predictions/<model_id>/<season>_<season_type>_<week>.parquet
    game_id       int64
    model_id      string    lower-case letters, digits, "_", "-", "."
    p_home_win    float64   probability the home team wins, 0 to 1, never empty.
                            At a neutral site "home" is the team CFBD lists as home.
    pred_margin   float64   home points minus away points. Empty until phase 2.
    pred_total    float64   home plus away points. Empty until phase 2.
    cutoff_utc    timestamp the cutoff of the view the inputs were read through: nothing
                            later was used. A live run made before that moment used less.
    created_utc   timestamp when the row was written
    reasons       string    JSON list, strongest first; "[]" when there are none. Each item:
                            {"text": str, "feature": str?, "effect": number?, "value": number?}
                            `effect` is signed toward the home team.
One row per (game_id, model_id). Baselines use the same schema. The week in the file
name is two digits (2026_regular_12.parquet).

Metrics: data/eval/<model_id>.json
    {
      "schema_version": 1, "model_id": str, "created_utc": ISO time, "scope": str,
      "overall":    block,
      "by_season":  [{"season": int, ...block}],
      "by_week":    [{"season": int, "season_type": str, "week": int, ...block}],
      "by_segment": [{"segment": str, ...block}]
    }
    block = {"n": int, "log_loss": number|null, "brier": number|null,
             "accuracy": number|null, "calibration": [bin, ...]}
    bin   = {"p_lo": number, "p_hi": number, "n": int, "p_mean": number|null, "win_rate": number|null}
`scope` names the games behind `overall`, `by_season` and `by_week` (headline scope is
"fbs_vs_fbs"). `calibration` is required in `overall` and `by_season`, optional elsewhere.
Metrics are null when n is 0. Segment names in use: fbs_vs_fbs, fbs_vs_fcs, conference,
non_conference, neutral, regular, postseason.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from .game_index import SEASON_TYPES

PREDICTION_COLUMNS = [
    "game_id", "model_id", "p_home_win", "pred_margin", "pred_total", "cutoff_utc", "created_utc", "reasons",
]
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9_.-]*")
METRICS_SCHEMA_VERSION = 1
METRIC_FIELDS = ("log_loss", "brier", "accuracy")
REASON_KEYS = {"text", "feature", "effect", "value"}


class SchemaError(ValueError):
    pass


# ---- predictions -----------------------------------------------------------------------

def reasons_json(reasons: list[dict] | None) -> str:
    """Encode a list of reasons, checking each item."""
    items = list(reasons or [])
    for item in items:
        _check_reason(item)
    return json.dumps(items, separators=(",", ":"))


def _check_reason(item) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
        raise SchemaError(f"a reason needs a non-empty 'text': {item!r}")
    extra = set(item) - REASON_KEYS
    if extra:
        raise SchemaError(f"unknown reason keys {sorted(extra)}")
    for key in ("effect", "value"):
        if item.get(key) is not None and not _is_number(item[key]):
            raise SchemaError(f"reason '{key}' must be a number or null")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def parse_reasons(text: str) -> list[dict]:
    try:
        items = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"reasons is not valid JSON: {text!r}") from exc
    if not isinstance(items, list):
        raise SchemaError("reasons must be a JSON list")
    for item in items:
        _check_reason(item)
    return items


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Raise SchemaError unless `frame` meets the contract. Returns it with canonical dtypes."""
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"predictions are missing columns: {missing}")
    extra = [c for c in frame.columns if c not in PREDICTION_COLUMNS]
    if extra:
        raise SchemaError(f"predictions have unexpected columns: {extra}")
    out = frame[PREDICTION_COLUMNS].copy()
    try:
        out["game_id"] = out["game_id"].astype("int64")
    except (TypeError, ValueError) as exc:
        raise SchemaError("game_id must be an integer without gaps") from exc
    out["model_id"] = out["model_id"].astype("str")
    bad = sorted({m for m in out["model_id"].unique() if not MODEL_ID.fullmatch(m)})
    if bad:
        raise SchemaError(f"invalid model_id values: {bad}")
    if out.duplicated(["game_id", "model_id"]).any():
        raise SchemaError("more than one row for a (game_id, model_id)")
    for col in ("p_home_win", "pred_margin", "pred_total"):
        try:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{col} must be numeric") from exc
    p = out["p_home_win"]
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        raise SchemaError("p_home_win must be between 0 and 1 and never empty")
    if (out["pred_total"] < 0).any():
        raise SchemaError("pred_total cannot be negative")
    for col in ("cutoff_utc", "created_utc"):
        if not isinstance(out[col].dtype, pd.DatetimeTZDtype):
            raise SchemaError(f"{col} must be timezone-aware")
        if out[col].isna().any():
            raise SchemaError(f"{col} has empty values")
        out[col] = out[col].dt.tz_convert("UTC").dt.as_unit("ns")
    if out["reasons"].isna().any():
        raise SchemaError('reasons has empty values; use "[]"')
    out["reasons"] = out["reasons"].astype("str")
    for text in out["reasons"].unique():
        parse_reasons(text)
    return out


def prediction_path(data_dir: str | Path, model_id: str, season: int, season_type: str, week: int) -> Path:
    if not MODEL_ID.fullmatch(model_id):
        raise SchemaError(f"invalid model_id {model_id!r}")
    if season_type not in SEASON_TYPES:
        raise SchemaError(f"season_type must be one of {SEASON_TYPES}")
    return Path(data_dir) / "predictions" / model_id / f"{int(season)}_{season_type}_{int(week):02d}.parquet"


def write_predictions(frame: pd.DataFrame, data_dir: str | Path, season: int, season_type: str, week: int) -> Path:
    """Validate and write one model's predictions for one week. Returns the path."""
    out = validate_predictions(frame)
    models = out["model_id"].unique()
    if len(models) != 1:
        raise SchemaError(f"one file holds one model; got {sorted(models)}")
    path = prediction_path(data_dir, models[0], season, season_type, week)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def read_predictions(data_dir: str | Path, model_id: str | None = None) -> pd.DataFrame:
    """All stored predictions, optionally for one model, with season, season_type and week
    taken from the file names."""
    root = Path(data_dir) / "predictions"
    frames = []
    for path in sorted(root.glob(f"{model_id or '*'}/*.parquet")):
        m = re.fullmatch(r"(\d{4})_([a-z]+)_(\d+)", path.stem)
        if not m:
            continue
        frame = validate_predictions(pd.read_parquet(path))
        frame["season"], frame["season_type"], frame["week"] = int(m.group(1)), m.group(2), int(m.group(3))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS + ["season", "season_type", "week"])
    return pd.concat(frames, ignore_index=True)


# ---- metrics -----------------------------------------------------------------------------

def _check_block(block, where: str, need_calibration: bool) -> None:
    if not isinstance(block, dict):
        raise SchemaError(f"{where}: expected an object")
    n = block.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise SchemaError(f"{where}: n must be a non-negative integer")
    for name in METRIC_FIELDS:
        if name not in block:
            raise SchemaError(f"{where}: missing {name}")
        value = block[name]
        if value is None:
            if n > 0:
                raise SchemaError(f"{where}: {name} is null although n is {n}")
        elif not _is_number(value) or value < 0:
            raise SchemaError(f"{where}: {name} must be a non-negative number or null")
    if block["accuracy"] is not None and block["accuracy"] > 1:
        raise SchemaError(f"{where}: accuracy above 1")
    if block["brier"] is not None and block["brier"] > 1:
        raise SchemaError(f"{where}: brier above 1")
    if "calibration" not in block:
        if need_calibration:
            raise SchemaError(f"{where}: missing calibration")
        return
    bins = block["calibration"]
    if not isinstance(bins, list):
        raise SchemaError(f"{where}: calibration must be a list")
    total = 0
    for i, b in enumerate(bins):
        spot = f"{where}.calibration[{i}]"
        if not isinstance(b, dict) or not {"p_lo", "p_hi", "n", "p_mean", "win_rate"} <= set(b):
            raise SchemaError(f"{spot}: needs p_lo, p_hi, n, p_mean, win_rate")
        if not (_is_number(b["p_lo"]) and _is_number(b["p_hi"]) and 0 <= b["p_lo"] < b["p_hi"] <= 1):
            raise SchemaError(f"{spot}: bad bin edges")
        if not isinstance(b["n"], int) or b["n"] < 0:
            raise SchemaError(f"{spot}: n must be a non-negative integer")
        for name in ("p_mean", "win_rate"):
            v = b[name]
            if v is None:
                if b["n"] > 0:
                    raise SchemaError(f"{spot}: {name} is null although n is {b['n']}")
            elif not _is_number(v) or not 0 <= v <= 1:
                raise SchemaError(f"{spot}: {name} must be between 0 and 1")
        total += b["n"]
    if bins and total != n:
        raise SchemaError(f"{where}: calibration bins hold {total} games, n is {n}")


def validate_metrics(doc: dict) -> dict:
    """Raise SchemaError unless `doc` meets the metrics contract. Returns it unchanged."""
    if not isinstance(doc, dict):
        raise SchemaError("metrics must be a JSON object")
    if doc.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise SchemaError(f"schema_version must be {METRICS_SCHEMA_VERSION}")
    if not isinstance(doc.get("model_id"), str) or not MODEL_ID.fullmatch(doc["model_id"]):
        raise SchemaError("invalid or missing model_id")
    if not isinstance(doc.get("scope"), str) or not doc["scope"]:
        raise SchemaError("missing scope")
    try:
        created = pd.Timestamp(doc.get("created_utc"))
    except (TypeError, ValueError) as exc:
        raise SchemaError("created_utc is not a timestamp") from exc
    if created is pd.NaT or created.tzinfo is None:
        raise SchemaError("created_utc must carry a timezone")
    _check_block(doc.get("overall"), "overall", need_calibration=True)
    spec = {
        "by_season": (("season", int),),
        "by_week": (("season", int), ("season_type", str), ("week", int)),
        "by_segment": (("segment", str),),
    }
    for section, keys in spec.items():
        rows = doc.get(section)
        if not isinstance(rows, list):
            raise SchemaError(f"{section} must be a list")
        seen = set()
        for i, row in enumerate(rows):
            where = f"{section}[{i}]"
            _check_block(row, where, need_calibration=section == "by_season")
            for key, typ in keys:
                if not isinstance(row.get(key), typ) or isinstance(row.get(key), bool):
                    raise SchemaError(f"{where}: {key} must be {typ.__name__}")
            ident = tuple(row[k] for k, _ in keys)
            if ident in seen:
                raise SchemaError(f"{where}: duplicate entry {ident}")
            seen.add(ident)
            if section == "by_week" and row["season_type"] not in SEASON_TYPES:
                raise SchemaError(f"{where}: season_type must be one of {SEASON_TYPES}")
    return doc


def metrics_path(data_dir: str | Path, model_id: str) -> Path:
    if not MODEL_ID.fullmatch(model_id):
        raise SchemaError(f"invalid model_id {model_id!r}")
    return Path(data_dir) / "eval" / f"{model_id}.json"


def write_metrics(doc: dict, data_dir: str | Path) -> Path:
    validate_metrics(doc)
    path = metrics_path(data_dir, doc["model_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def read_metrics(data_dir: str | Path, model_id: str) -> dict:
    return validate_metrics(json.loads(metrics_path(data_dir, model_id).read_text(encoding="utf-8")))
