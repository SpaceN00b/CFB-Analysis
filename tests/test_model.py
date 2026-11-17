"""W8 model: matrix builder, model families, walk-forward predictions, leak test."""
import json

import numpy as np
import pandas as pd
import pytest

from cfb.core.leak import scramble_after
from cfb.core.schemas import validate_predictions
from cfb.eval.splits import visible_results, walk_forward
from cfb.features.base import run_group
from cfb.model import config as C
from cfb.model.matrix import feature_columns, game_matrix, groups_by_name
from cfb.model.models import LogitModel, MarginModel, log_loss
from cfb.predict import Predictor

TEST_GROUPS = ["str_power", "str_elo", "ctx_schedule", "ctx_season_phase"]


def _features(wh):
    games = wh.schedule()
    return {g.name: run_group(g, wh, games) for g in groups_by_name(TEST_GROUPS)}


def test_config_groups_exist():
    names = {g.name for g in groups_by_name()}
    assert set(C.MODEL_GROUPS) <= names
    assert "market_lines" not in C.MODEL_GROUPS
    cols = C.model_columns()
    assert len(cols) == len(set(cols))


def test_matrix_home_away_diff(synthetic):
    games = synthetic.schedule().head(12)
    groups = groups_by_name(["str_power", "ctx_season_phase"])
    feats = {g.name: run_group(g, synthetic, games) for g in groups}
    X = game_matrix(games, feats, groups)
    assert len(X) == len(games) and (X["game_id"].to_numpy() == games["game_id"].to_numpy()).all()
    f = feats["str_power"].set_index(["game_id", "team_id"])["str_pwr_net"]
    for _, r in X.iterrows():
        g = games[games["game_id"] == r["game_id"]].iloc[0]
        h, a = f.get((g["game_id"], g["home_id"])), f.get((g["game_id"], g["away_id"]))
        assert r["h_str_pwr_net"] == pytest.approx(h, nan_ok=True)
        assert r["a_str_pwr_net"] == pytest.approx(a, nan_ok=True)
        if pd.notna(h) and pd.notna(a):
            assert r["d_str_pwr_net"] == pytest.approx(h - a)
    assert "ctx_postseason" in X.columns
    assert feature_columns(groups, ("d",))[0].startswith("d_")


@pytest.mark.parametrize("cls", [LogitModel, MarginModel])
def test_models_learn_a_signal(cls):
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n),
                      "season": np.repeat(np.arange(2015, 2021), n // 6)})
    X.loc[rng.random(n) < 0.1, "noise"] = np.nan
    margin = 10 * X["x"] + rng.normal(scale=14, size=n)
    y = (margin > 0).astype(float).to_numpy()
    m = cls(["x", "noise"], lam=0.001).fit(X, y, margin.to_numpy())
    p = m.predict_proba(X)
    assert log_loss(y, p) < log_loss(y, np.full(n, y.mean())) - 0.1
    c = m.contributions(X)
    assert np.allclose(c.sum(axis=1).to_numpy() + (m.linear(X) - c.sum(axis=1).to_numpy()), m.linear(X))
    assert abs(np.corrcoef(c["x"], X["x"])[0, 1]) > 0.99


def test_walk_forward_predictions_are_valid(synthetic):
    pr = Predictor(synthetic, groups=TEST_GROUPS, features=_features(synthetic))
    seasons = sorted(synthetic.schedule()["season"].unique())[-1:]
    preds = walk_forward(synthetic, pr.fit, pr.predict, C.MODEL_ID, seasons=seasons, refit="cutoff")
    assert len(preds) > 0
    validate_predictions(preds.drop(columns=["season", "season_type", "week"]))
    assert preds["p_home_win"].between(0, 1).all()
    r = json.loads(preds["reasons"].iloc[0])
    assert isinstance(r, list) and all("text" in x for x in r)


def test_model_does_not_leak(synthetic):
    """Predictions at a cutoff do not change when every result after it is scrambled."""
    sched = synthetic.schedule()
    last = sorted(sched["season"].unique())[-1]
    cutoffs = sorted(sched.loc[sched["season"] == last, "cutoff_utc"].unique())
    for cutoff in (cutoffs[1], cutoffs[len(cutoffs) // 2]):
        slate = sched[sched["cutoff_utc"] == cutoff].reset_index(drop=True)
        out = []
        for wh in (synthetic, scramble_after(synthetic, cutoff, seed=3)):
            pr = Predictor(wh, groups=TEST_GROUPS, features=_features(wh))
            view = wh.view(cutoff)
            fitted = pr.fit(view, visible_results(wh, cutoff))
            out.append(pr.predict(fitted, view, slate).sort_values("game_id")["p_home_win"].to_numpy())
        np.testing.assert_allclose(out[0], out[1], rtol=1e-9, atol=1e-12)


def test_live_features_match_stored(synthetic):
    """An on-demand path (features built live at the cutoff) gives the stored values."""
    feats = _features(synthetic)
    sched = synthetic.schedule()
    cutoff = sorted(sched["cutoff_utc"].unique())[-3]
    slate = sched[sched["cutoff_utc"] == cutoff].reset_index(drop=True)
    pr_stored = Predictor(synthetic, groups=TEST_GROUPS, features=feats)
    view = synthetic.view(cutoff)
    a = pr_stored.matrix_for(view, slate)
    live = slate.copy()
    live["game_id"] = -live["game_id"]          # unknown ids force the live path
    b = pr_stored.matrix_for(view, live)
    cols = [c for c in a.columns if c.startswith(("h_", "a_", "d_", "ctx_"))]
    np.testing.assert_allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)
