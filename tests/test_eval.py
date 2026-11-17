"""cfb.eval: metrics, splits, walk-forward runner, paired bootstrap, baselines, CLI."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from cfb.core import RESULT_LAG, Warehouse
from cfb.core.schemas import SchemaError, read_metrics, read_predictions, validate_metrics
from cfb.eval import (
    common_games, compare, evaluate, log_loss, paired_bootstrap, run_baseline, score_block, season_forward,
    season_forward_predict, walk_forward, write_weekly,
)
from cfb.eval.__main__ import main as cli
from cfb.eval.baselines import logit_fit
from cfb.eval.metrics import accuracy, brier, calibration


# ---- metrics ------------------------------------------------------------------------------

def test_scores_on_known_values():
    y, p = np.array([1, 0, 1, 0]), np.array([0.8, 0.3, 0.5, 0.0])
    assert log_loss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.7) + np.log(0.5) + np.log(1 - 1e-6)) / 4)
    assert brier(y, p) == pytest.approx((0.04 + 0.09 + 0.25 + 0) / 4)
    assert accuracy(y, p) == pytest.approx(1.0)  # 0.5 counts as a home pick
    assert np.isnan(log_loss([], []))


def test_calibration_bins_hold_every_game():
    p = np.array([0.0, 0.05, 0.1, 0.55, 0.99, 1.0])
    bins = calibration(np.ones(6), p)
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 6
    assert bins[-1]["n"] == 2 and bins[1]["n"] == 1  # 0.1 opens the second bin, 1.0 closes the last
    assert bins[3]["p_mean"] is None and bins[3]["win_rate"] is None
    assert score_block([], [])["log_loss"] is None


def _fake_preds(wh: Warehouse, model_id: str, p) -> pd.DataFrame:
    games = wh.outcomes()
    return pd.DataFrame({"game_id": games["game_id"], "model_id": model_id,
                         "p_home_win": np.broadcast_to(p, len(games)).astype(float)})


def test_evaluate_writes_a_valid_document(synthetic, tmp_path):
    out = synthetic.outcomes()
    doc = evaluate(_fake_preds(synthetic, "flat", 0.6), out)
    validate_metrics(doc)
    fbs = out[out["home_fbs"] & out["away_fbs"]]
    assert doc["overall"]["n"] == len(fbs) and doc["scope"] == "fbs_vs_fbs"
    assert {r["season"] for r in doc["by_season"]} == set(fbs["season"])
    segs = {r["segment"]: r["n"] for r in doc["by_segment"]}
    assert segs["fbs_vs_fbs"] == len(fbs) and segs["fbs_vs_fcs"] == len(out) - len(fbs)
    assert segs["regular"] + segs["postseason"] == len(fbs)
    assert segs["conference"] + segs["non_conference"] == len(fbs)
    assert sum(r["n"] for r in doc["by_week"]) == len(fbs)
    assert evaluate(_fake_preds(synthetic, "flat", 0.6), out, scope="all")["overall"]["n"] == len(out)


def test_evaluate_drops_ties_and_unplayed_games(synthetic):
    out = synthetic.outcomes().copy()
    tie = out.index[0]
    out.loc[tie, "away_points"] = out.loc[tie, "home_points"]
    doc = evaluate(_fake_preds(synthetic, "flat", 0.6), out, scope="all")
    assert doc["overall"]["n"] == len(out) - 1
    sched = synthetic.schedule()
    unplayed = sched[~sched["game_id"].isin(out["game_id"])]
    assert len(unplayed)  # the last synthetic season is in progress
    extra = pd.DataFrame({"game_id": unplayed["game_id"], "model_id": "flat", "p_home_win": 0.6})
    both = pd.concat([_fake_preds(synthetic, "flat", 0.6), extra])
    assert evaluate(both, out, scope="all")["overall"]["n"] == len(out) - 1


def test_evaluate_rejects_two_rows_per_game(synthetic):
    p = _fake_preds(synthetic, "flat", 0.6)
    with pytest.raises(SchemaError):
        evaluate(pd.concat([p, p]), synthetic.outcomes())


# ---- splits and runner ----------------------------------------------------------------------

def test_season_forward_never_trains_on_the_test_season():
    frame = pd.DataFrame({"season": np.repeat([2016, 2017, 2018, 2019], 3), "x": np.arange(12)})
    splits = list(season_forward(frame, seasons=[2016, 2018, 2019, 2020]))
    assert [s.season for s in splits] == [2018, 2019]  # 2016 has no training rows, 2020 no test rows
    for s in splits:
        assert s.train["season"].max() < s.season and (s.test["season"] == s.season).all()
    pred = season_forward_predict(frame, fit=lambda tr: tr["x"].mean(), predict=lambda m, te: np.full(len(te), m),
                                  seasons=[2018, 2019])
    assert list(pred.index) == list(range(6, 12)) and pred.iloc[0] == pytest.approx(np.arange(6).mean())


def test_walk_forward_shows_only_visible_results(synthetic, tmp_path):
    calls = []

    def fit(view, train):
        assert (train["start_time_utc"] + RESULT_LAG <= view.cutoff).all()
        calls.append((view.cutoff, len(train)))
        return float((train["home_points"] > train["away_points"]).mean()) if len(train) else 0.5

    def predict(model, view, slate):
        return pd.DataFrame({"game_id": slate["game_id"], "p_home_win": model,
                             "reasons": [[{"text": "base rate", "value": model}]] * len(slate)})

    preds = walk_forward(synthetic, fit, predict, "toy", seasons=[2022, 2023], refit="cutoff")
    sched = synthetic.schedule([2022, 2023])
    assert set(preds["game_id"]) == set(sched["game_id"])
    assert len(calls) == sched["cutoff_utc"].nunique()
    assert [n for _, n in calls] == sorted(n for _, n in calls)  # more results visible each week
    merged = preds.merge(sched[["game_id", "cutoff_utc"]], on="game_id", suffixes=("", "_g"))
    assert (merged["cutoff_utc"] == merged["cutoff_utc_g"]).all()
    assert json.loads(preds["reasons"].iloc[0])[0]["text"] == "base rate"

    paths = write_weekly(preds, tmp_path)
    back = read_predictions(tmp_path, "toy")
    assert len(back) == len(preds) and len(paths) == preds.groupby(["season", "season_type", "week"]).ngroups

    once = []
    walk_forward(synthetic, lambda v, t: once.append(v.cutoff) or 0.5, predict, "toy", seasons=[2022, 2023])
    assert len(once) == 2  # refit="season": one fit per season


def test_walk_forward_rejects_bad_output(synthetic):
    fit = lambda v, t: None
    stray = lambda m, v, s: pd.DataFrame({"game_id": [-1], "p_home_win": [0.5]})
    with pytest.raises(SchemaError):
        walk_forward(synthetic, fit, stray, "toy", seasons=[2022])
    early = lambda m, v, s: pd.DataFrame({"game_id": s["game_id"], "p_home_win": 0.5,
                                          "cutoff_utc": s["cutoff_utc"] - pd.Timedelta(days=1)})
    with pytest.raises(SchemaError):
        walk_forward(synthetic, fit, early, "toy", seasons=[2022])
    with pytest.raises(SchemaError):
        walk_forward(synthetic, fit, early, "Bad Id", seasons=[2022])


def test_write_weekly_replace_removes_stale_weeks(synthetic, tmp_path):
    predict = lambda m, v, s: pd.DataFrame({"game_id": s["game_id"], "p_home_win": 0.5})
    write_weekly(walk_forward(synthetic, lambda v, t: None, predict, "toy", seasons=[2022, 2023]), tmp_path)
    write_weekly(walk_forward(synthetic, lambda v, t: None, predict, "toy", seasons=[2023]), tmp_path, replace=True)
    assert set(read_predictions(tmp_path, "toy")["season"]) == {2023}


# ---- paired bootstrap -------------------------------------------------------------------------

def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.1, 0.9, 3000)
    y = (rng.uniform(size=3000) < p_true).astype(float)
    same = paired_bootstrap(y, p_true, p_true, n_boot=200)
    assert same["diff"] == 0 and same["se"] == 0
    r = paired_bootstrap(y, p_true, np.full(3000, 0.5), n_boot=500)
    assert r["diff"] < 0 and r["ci_hi"] < 0 and r["p_a_better"] == 1.0
    assert r["a"] == pytest.approx(log_loss(y, p_true))
    acc = paired_bootstrap(y, p_true, np.full(3000, 0.5), metric="accuracy", n_boot=200)
    assert acc["diff"] > 0 and acc["p_a_better"] > 0.95
    weeks = np.repeat(np.arange(30), 100)
    cl = paired_bootstrap(y, p_true, np.full(3000, 0.5), n_boot=500, clusters=weeks)
    assert cl["n_clusters"] == 30 and cl["diff"] == pytest.approx(r["diff"])
    assert paired_bootstrap(y, p_true, p_true[::-1], n_boot=100, seed=3) == \
        paired_bootstrap(y, p_true, p_true[::-1], n_boot=100, seed=3)


def test_compare_uses_common_games(synthetic):
    a = _fake_preds(synthetic, "a", 0.6)
    b = _fake_preds(synthetic, "b", 0.5).iloc[::2]
    g = common_games({"a": a, "b": b}, synthetic.outcomes())
    assert set(g["game_id"]) <= set(b["game_id"]) and {"p_a", "p_b", "y"} <= set(g.columns)
    r = compare(a, b, synthetic.outcomes(), n_boot=100, by_week=True)
    assert r["n"] == len(g) and r["n_clusters"] < r["n"]


# ---- baselines -------------------------------------------------------------------------------

def test_logit_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20000), rng.normal(size=20000)])
    y = (rng.uniform(size=20000) < 1 / (1 + np.exp(-(X @ [0.3, -1.2])))).astype(float)
    assert logit_fit(y, X) == pytest.approx([0.3, -1.2], abs=0.06)


@pytest.mark.parametrize("model_id", ["baseline_close", "baseline_moneyline", "baseline_elo", "baseline_home"])
def test_baselines_on_synthetic(synthetic, model_id):
    preds = run_baseline(synthetic, model_id, seasons=[2022, 2023])
    assert len(preds) and preds["p_home_win"].between(0, 1).all()
    sched = synthetic.schedule([2022, 2023]).set_index("game_id")
    g = sched.loc[preds["game_id"]]
    if model_id in ("baseline_close", "baseline_moneyline"):
        assert (preds["cutoff_utc"].to_numpy() == g["start_time_utc"].to_numpy()).all()  # closing line: kickoff
    else:
        assert (preds["cutoff_utc"].to_numpy() == g["cutoff_utc"].to_numpy()).all()
    if model_id == "baseline_home":
        neutral = g["neutral"].to_numpy()
        assert np.allclose(preds["p_home_win"][neutral], 0.5)
        assert preds["p_home_win"][~neutral].nunique() <= 2  # one value per season
    if model_id == "baseline_elo":
        fcs = ~(g["home_fbs"] & g["away_fbs"]).to_numpy()
        assert not fcs.any()  # FBS vs FCS Elo from 2022 is not a pregame value
    for r in preds["reasons"]:
        json.loads(r)


def test_baseline_close_fits_only_earlier_seasons(synthetic):
    """Changing 2023 results must not change 2023 predictions."""
    base = run_baseline(synthetic, "baseline_close", seasons=[2023])
    gi = synthetic.game_index.copy()
    flip = gi["season"] == 2023
    gi.loc[flip, ["home_points", "away_points"]] = gi.loc[flip, ["away_points", "home_points"]].to_numpy()
    wh = Warehouse(synthetic.root)
    wh.meta["game_index"] = gi
    again = run_baseline(wh, "baseline_close", seasons=[2023])
    assert np.allclose(base["p_home_win"], again["p_home_win"])


# ---- CLI -------------------------------------------------------------------------------------

def test_cli_baselines_and_score(synthetic, tmp_path, capsys):
    root = tmp_path / "data"
    shutil.copytree(synthetic.root, root)
    assert cli(["--data-dir", str(root), "baselines", "--seasons", "2022-2023", "--score-seasons", "2022-2023"]) == 0
    text = capsys.readouterr().out
    assert "Season-forward" in text and "baseline_close" in text
    for m in ("baseline_close", "baseline_moneyline", "baseline_elo", "baseline_home"):
        assert read_metrics(root, m)["overall"]["n"] > 0
    assert cli(["--data-dir", str(root), "score", "baseline_elo", "--seasons", "2022-2023",
                "--common-with", "baseline_close", "--compare", "baseline_home", "--n-boot", "50"]) == 0
    text = capsys.readouterr().out
    assert "vs baseline_home" in text and "log_loss" in text
    assert "baseline_close" in read_metrics(root, "baseline_elo")["scope"]
    assert cli(["--data-dir", str(root), "score", "nothing_here"]) == 1


# ---- snapshot ----------------------------------------------------------------------------------

def test_baselines_reproduce_the_plan_numbers(snapshot):
    """Season-forward 2018-2025 on FBS vs FBS games with a spread and Elo: 0.520, 0.549, 0.679."""
    seasons = range(2018, 2026)
    preds = {m: run_baseline(snapshot, m, seasons=seasons) for m in ("baseline_close", "baseline_elo", "baseline_home")}
    g = common_games(preds, snapshot.outcomes())
    assert len(g) == 6024
    for m, want in (("baseline_close", 0.520), ("baseline_elo", 0.549), ("baseline_home", 0.679)):
        assert log_loss(g["y"], g[f"p_{m}"]) == pytest.approx(want, abs=0.002), m
