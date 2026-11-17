"""Phase 2 points models (spread and total) and their scoring."""
import numpy as np
import pandas as pd
import pytest

from cfb.core.leak import scramble_after
from cfb.core.schemas import validate_predictions
from cfb.eval.points import evaluate_points, format_points, hit_rate, points_frame
from cfb.eval.splits import visible_results, walk_forward
from cfb.features.base import run_group
from cfb.model import config as C
from cfb.model import points_config as P
from cfb.model.matrix import game_matrix, groups_by_name
from cfb.model.models import PointsModel
from cfb.predict import Predictor

TEST_GROUPS = ["str_power", "str_elo", "ctx_schedule", "ctx_season_phase"]


def _features(wh):
    games = wh.schedule()
    return {g.name: run_group(g, wh, games) for g in groups_by_name(TEST_GROUPS)}


def test_points_config():
    names = {g.name for g in groups_by_name()}
    for t in ("margin", "total"):
        assert set(P.GROUPS[t]) <= names
        assert "market_lines" not in P.GROUPS[t] and "ctx_weather" not in P.GROUPS[t]
        cols = P.model_columns(t)
        assert len(cols) == len(set(cols))
        assert not any(c.startswith("mkt_") for c in cols)
    assert all(c.startswith(("site", "fcs", "d_", "ctx_", "str_")) or c in P.EXTRA["margin"]
               for c in P.model_columns("margin"))


def test_clip_target():
    y = np.array([-50.0, -3.0, 0.0, 10.0, 60.0])
    old = dict(P.CAP)
    try:
        P.CAP["margin"], P.CAP["total"] = 35.0, 90.0
        assert P.clip_target("margin", y).tolist() == [-35.0, -3.0, 0.0, 10.0, 35.0]
        assert P.clip_target("total", np.array([20.0, 120.0])).tolist() == [20.0, 90.0]
        P.CAP["margin"] = None
        assert P.clip_target("margin", y).tolist() == y.tolist()
    finally:
        P.CAP.update(old)


def test_matrix_sum_form(synthetic):
    games = synthetic.schedule().head(12)
    groups = groups_by_name(["str_power"])
    X = game_matrix(games, {g.name: run_group(g, synthetic, games) for g in groups}, groups)
    ok = X["h_str_pwr_off"].notna() & X["a_str_pwr_off"].notna()
    assert ok.any()
    np.testing.assert_allclose(X.loc[ok, "s_str_pwr_off"], X.loc[ok, "h_str_pwr_off"] + X.loc[ok, "a_str_pwr_off"])


def test_points_model_learns():
    rng = np.random.default_rng(1)
    n = 4000
    X = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n),
                      "season": np.repeat(np.arange(2015, 2023), n // 8)})
    X.loc[rng.random(n) < 0.1, "noise"] = np.nan
    y = 55 + 8 * X["x"].to_numpy() + rng.normal(scale=14, size=n)
    m = PointsModel(["x", "noise"], lam=0.001).fit(X, y)
    pred = m.predict(X)
    assert np.mean(np.abs(y - pred)) < np.mean(np.abs(y - y.mean())) - 1
    assert m.resid_sd == pytest.approx(14, rel=0.05)
    c = m.contributions(X)
    np.testing.assert_allclose(c.sum(axis=1).to_numpy() + m.intercept, pred)


def test_hit_rate():
    pred = np.array([-3.0, 7.0, 10.0, 5.0, 2.0])
    line = np.array([0.0, 3.0, 3.0, 3.0, 2.0])     # last: zero edge, left out
    actual = np.array([-7.0, 10.0, 0.0, 3.0, 9.0])  # 4th: push, left out
    rate, n = hit_rate(pred, line, actual)
    assert n == 3 and rate == pytest.approx(2 / 3)
    rate, n = hit_rate(pred, line, actual, threshold=5.0)
    assert n == 1 and rate == 0.0


def test_evaluate_points():
    outcomes = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "season": 2025, "season_type": "regular", "week": 1,
        "home_fbs": [True, True, True, False], "away_fbs": True,
        "home_points": [30, 14, 21, 50], "away_points": [20, 17, 21, 0],
    })
    lines = pd.DataFrame({"game_id": [1, 2, 3, 4], "mkt_spread_close": [-7.0, 3.0, 0.0, -30.0],
                          "mkt_total_close": [45.0, 30.0, 40.0, 60.0]})
    preds = pd.DataFrame({"game_id": [1, 2, 3, 4], "pred_margin": [5.0, 1.0, 3.0, 20.0],
                          "pred_total": [52.0, 35.0, np.nan, 40.0]})
    doc = evaluate_points(preds, outcomes, lines, "m")
    mg = doc["margin"]["overall"]
    assert mg["n"] == 3                                   # FBS vs FBS only
    assert mg["mae"] == pytest.approx((5 + 4 + 3) / 3, abs=1e-4)
    assert mg["line_mae"] == pytest.approx((3 + 0 + 0) / 3, abs=1e-4)
    # game 1: edge 5 - 7 = -2 (away), result 10 - 7 = +3 (home covers) -> miss
    # game 2: edge 1 - (-3) = +4 (home), result -3 - (-3) = 0 -> push, left out
    # game 3: edge 3 - 0 = +3 (home), result 0 -> push, left out
    assert mg["vs_line"][0] == {"min_edge": 0.0, "n": 1, "hit_rate": 0.0}
    tt = doc["total"]["overall"]
    assert tt["n"] == 2 and tt["mae"] == pytest.approx((2 + 4) / 2, abs=1e-4)
    # over 45 with 50 actual -> hit; over 30 with 31 actual -> hit
    assert tt["vs_line"][0]["hit_rate"] == 1.0
    assert "Spread" in format_points(doc)
    assert len(points_frame(preds, outcomes, lines)) == 4


def test_predictor_fills_points(synthetic):
    pr = Predictor(synthetic, groups=TEST_GROUPS, features=_features(synthetic))
    seasons = sorted(synthetic.schedule()["season"].unique())[-1:]
    preds = walk_forward(synthetic, pr.fit, pr.predict, C.MODEL_ID, seasons=seasons, refit="cutoff")
    validate_predictions(preds.drop(columns=["season", "season_type", "week"]))
    assert preds["pred_margin"].notna().all() and preds["pred_total"].notna().all()
    assert (preds["pred_total"] >= 0).all()
    # the margin and the win probability move together (same inputs, both fitted on margins)
    logit = np.log(preds["p_home_win"] / (1 - preds["p_home_win"]))
    assert np.corrcoef(preds["pred_margin"], logit)[0, 1] > 0.7


def test_points_do_not_leak(synthetic):
    sched = synthetic.schedule()
    last = sorted(sched["season"].unique())[-1]
    cutoff = sorted(sched.loc[sched["season"] == last, "cutoff_utc"].unique())[2]
    slate = sched[sched["cutoff_utc"] == cutoff].reset_index(drop=True)
    out = []
    for wh in (synthetic, scramble_after(synthetic, cutoff, seed=5)):
        pr = Predictor(wh, groups=TEST_GROUPS, features=_features(wh))
        view = wh.view(cutoff)
        p = pr.predict(pr.fit(view, visible_results(wh, cutoff)), view, slate).sort_values("game_id")
        out.append(p[["pred_margin", "pred_total"]].to_numpy())
    np.testing.assert_allclose(out[0], out[1], rtol=1e-9, atol=1e-9)
