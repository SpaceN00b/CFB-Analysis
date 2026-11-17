"""Prediction and metrics file contracts, and the committed sample files."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb.core import schemas
from cfb.core.samples import SAMPLE_MODEL, sample_metrics, sample_predictions, write_samples
from cfb.core.schemas import (
    PREDICTION_COLUMNS, SchemaError, parse_reasons, prediction_path, read_metrics, read_predictions, reasons_json,
    validate_metrics, validate_predictions, write_metrics, write_predictions,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _predictions(n=3, model="m1"):
    now = pd.Timestamp("2025-09-02T12:00:00Z")
    return pd.DataFrame({
        "game_id": np.arange(1, n + 1), "model_id": model, "p_home_win": np.linspace(0.2, 0.8, n),
        "pred_margin": np.nan, "pred_total": np.nan, "cutoff_utc": pd.Timestamp("2025-09-01T07:00:00Z"),
        "created_utc": now, "reasons": reasons_json([{"text": "Home team rated higher", "feature": "str_gap", "effect": 0.4}]),
    })


def test_prediction_roundtrip(tmp_path):
    path = write_predictions(_predictions(), tmp_path, 2025, "regular", 2)
    assert path == tmp_path / "predictions" / "m1" / "2025_regular_02.parquet"
    assert path == prediction_path(tmp_path, "m1", 2025, "regular", 2)
    write_predictions(_predictions(2, "baseline_home"), tmp_path, 2025, "postseason", 1)
    back = read_predictions(tmp_path)
    assert list(back.columns) == PREDICTION_COLUMNS + ["season", "season_type", "week"]
    assert len(back) == 5 and set(back["model_id"]) == {"m1", "baseline_home"}
    one = read_predictions(tmp_path, "m1")
    assert (one["week"] == 2).all() and str(one["cutoff_utc"].dtype) == "datetime64[ns, UTC]"
    assert one["pred_margin"].isna().all()  # empty until phase 2
    assert parse_reasons(one["reasons"].iloc[0])[0]["feature"] == "str_gap"
    assert read_predictions(tmp_path / "nowhere").empty


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns="reasons"), "missing columns"),
    (lambda d: d.assign(extra=1), "unexpected columns"),
    (lambda d: d.assign(p_home_win=[0.2, 1.2, 0.5]), "between 0 and 1"),
    (lambda d: d.assign(p_home_win=[0.2, np.nan, 0.5]), "between 0 and 1"),
    (lambda d: d.assign(model_id="Has Spaces"), "invalid model_id"),
    (lambda d: d.assign(game_id=[1, 1, 2]), "more than one row"),
    (lambda d: d.assign(cutoff_utc=pd.Timestamp("2025-09-01")), "timezone-aware"),
    (lambda d: d.assign(reasons="not json"), "valid JSON"),
    (lambda d: d.assign(reasons='[{"feature": "x"}]'), "non-empty 'text'"),
    (lambda d: d.assign(reasons=None), "empty values"),
    (lambda d: d.assign(pred_total=-3.0), "negative"),
])
def test_prediction_validation_rejects(change, message):
    with pytest.raises(SchemaError, match=message):
        validate_predictions(change(_predictions()))


def test_one_file_holds_one_model(tmp_path):
    mixed = pd.concat([_predictions(2, "a"), _predictions(2, "b")])
    with pytest.raises(SchemaError, match="one model"):
        write_predictions(mixed, tmp_path, 2025, "regular", 1)
    with pytest.raises(SchemaError):
        prediction_path(tmp_path, "m1", 2025, "spring", 1)
    assert reasons_json(None) == "[]"
    with pytest.raises(SchemaError):
        reasons_json([{"text": "x", "weight": 1}])


def _metrics():
    cal = [{"p_lo": 0.0, "p_hi": 0.5, "n": 4, "p_mean": 0.3, "win_rate": 0.25},
           {"p_lo": 0.5, "p_hi": 1.0, "n": 6, "p_mean": 0.7, "win_rate": 0.67}]
    block = {"n": 10, "log_loss": 0.6, "brier": 0.2, "accuracy": 0.7}
    return {
        "schema_version": 1, "model_id": "m1", "created_utc": "2025-09-09T12:00:00+00:00", "scope": "fbs_vs_fbs",
        "overall": {**block, "calibration": cal},
        "by_season": [{"season": 2025, **block, "calibration": cal}],
        "by_week": [{"season": 2025, "season_type": "regular", "week": 1, **block},
                    {"season": 2025, "season_type": "regular", "week": 2, "n": 0, "log_loss": None, "brier": None,
                     "accuracy": None}],
        "by_segment": [{"segment": "conference", **block}],
    }


def test_metrics_roundtrip(tmp_path):
    path = write_metrics(_metrics(), tmp_path)
    assert path == tmp_path / "eval" / "m1.json"
    assert read_metrics(tmp_path, "m1") == _metrics()


def _set(doc, path, value):
    doc = copy.deepcopy(doc)
    node = doc
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return doc


@pytest.mark.parametrize("path, value, message", [
    (["schema_version"], 2, "schema_version"),
    (["model_id"], "Bad Id", "model_id"),
    (["created_utc"], "2025-09-09T12:00:00", "timezone"),
    (["scope"], "", "scope"),
    (["overall", "calibration"], KeyError, "missing calibration"),
    (["overall", "log_loss"], None, "null although n"),
    (["overall", "accuracy"], 1.4, "accuracy above 1"),
    (["overall", "n"], 11, "calibration bins hold"),
    (["by_season", 0, "season"], "2025", "season must be int"),
    (["by_week", 0, "season_type"], "spring", "season_type"),
    (["by_week", 1, "week"], 1, "duplicate entry"),
    (["by_segment"], None, "must be a list"),
    (["overall", "calibration", 0, "p_hi"], 0.0, "bin edges"),
    (["overall", "calibration", 0, "win_rate"], None, "null although n"),
])
def test_metrics_validation_rejects(path, value, message):
    with pytest.raises(SchemaError, match=message):
        validate_metrics(_set(_metrics(), path, value))


def test_samples_from_synthetic_data(synthetic, tmp_path):
    created = pd.Timestamp("2022-12-01T00:00:00Z")
    paths = write_samples(synthetic, tmp_path, 2022, [1, 2], created)
    assert [p.name for p in paths] == ["2022_regular_01.parquet", "2022_regular_02.parquet", f"{SAMPLE_MODEL}.json"]
    preds = read_predictions(tmp_path, SAMPLE_MODEL)
    games = synthetic.schedule([2022])
    fbs = games[games["home_fbs"] & games["away_fbs"] & games["week"].isin([1, 2]) & (games["season_type"] == "regular")]
    assert set(preds["game_id"]) == set(fbs["game_id"])  # games without Elo are left out
    doc = read_metrics(tmp_path, SAMPLE_MODEL)
    assert doc["overall"]["n"] == len(preds) and [w["week"] for w in doc["by_week"]] == [1, 2]
    # stronger Elo at home should mean a higher probability
    elo = synthetic.view(synthetic.cutoffs()[-1]).table("game_pregame")
    m = preds.merge(elo, on="game_id")
    assert np.corrcoef(m["p_home_win"], m["home_pregame_elo"] - m["away_pregame_elo"])[0, 1] > 0.9
    frame = sample_predictions(synthetic, 2022, "regular", 1, created)
    assert sample_metrics(frame, synthetic, created)["overall"]["n"] == len(frame)
    with pytest.raises(ValueError):
        sample_predictions(synthetic, 2022, "regular", 9, created)


def test_committed_sample_files_follow_the_contracts():
    files = sorted((SAMPLES / "predictions").glob("*/*.parquet"))
    assert files, "samples/predictions is empty"
    preds = read_predictions(SAMPLES)
    assert len(preds) and list(preds.columns[: len(PREDICTION_COLUMNS)]) == PREDICTION_COLUMNS
    for model in preds["model_id"].unique():
        doc = read_metrics(SAMPLES, model)
        assert doc["overall"]["n"] <= len(preds[preds["model_id"] == model])
    assert schemas.METRICS_SCHEMA_VERSION == json.loads((SAMPLES / "eval" / f"{SAMPLE_MODEL}.json").read_text())["schema_version"]
