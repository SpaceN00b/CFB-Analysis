"""Model rankings vs polls (cfb.report.rankings)."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from cfb.core.synthetic import SEASONS
from cfb.eval import run_baseline
from cfb.features.base import discover_groups
from cfb.features.strength.groups import Schedule
from cfb.report import build_bundle, load_config, render_dashboard, render_markdown, resolve_slate
from cfb.report import rankings as R
from cfb.report.config import ReportConfig

LIVE = SEASONS[-1]


def _slate(wh, week=4):
    return resolve_slate(wh, LIVE, "regular", week)


# ---- the ranking ------------------------------------------------------------------------------

def test_rankings_order_and_fields(synthetic):
    slate = _slate(synthetic)
    rk = R.build_rankings(synthetic, slate.asof, "power", top=5)
    json.dumps(rk, allow_nan=False)
    top = [t for t in rk["teams"] if t["in_top"]]
    assert [t["rank"] for t in top] == [1, 2, 3, 4, 5]
    assert all(a["rating"] >= b["rating"] for a, b in zip(top, top[1:]))
    assert rk["n_fbs"] == 8 and rk["season"] == LIVE and rk["top"] == 5
    assert rk["source"]["id"] == "power" and set(rk["source"]["columns"]) == {"off", "def", "se"}
    t = top[0]
    assert t["rating"] == pytest.approx(t["off"] - t["def"], abs=0.02)
    w, l = map(int, t["record"].split("-"))
    assert w + l > 0
    # Synthetic data have no polls.
    assert rk["polls"] == [] and rk["comparisons"] == {}
    # Movement against the previous week's ranking (same season).
    assert rk["previous_asof_utc"] is not None and any(x["prev_rank"] is not None for x in top)
    for x in top:
        if x["prev_rank"] is not None:
            assert x["move"] == x["prev_rank"] - x["rank"]


def test_rankings_use_only_visible_results(synthetic):
    """Same ranking whatever the warehouse holds after the cutoff."""
    slate = _slate(synthetic)
    a = R.build_rankings(synthetic, slate.asof, "power")
    b = R.build_rankings(synthetic, slate.asof + pd.Timedelta(hours=1), "power")  # no game in between
    assert [t["team_id"] for t in a["teams"]] == [t["team_id"] for t in b["teams"]]
    later = R.build_rankings(synthetic, _slate(synthetic, 5).asof, "power")
    assert later["asof_utc"] > a["asof_utc"]


def test_sor_matches_feature_group(synthetic):
    """strength_of_record repeats str_sor from the str_schedule group; keep them in step."""
    slate = _slate(synthetic)
    view = synthetic.view(slate.asof)
    feat = Schedule().build(view, slate.games).set_index("team_id")["str_sor"]
    mine = R.strength_of_record(view, feat.index.unique().to_numpy())
    feat = feat.groupby(level=0).first()
    np.testing.assert_allclose(mine.reindex(feat.index).to_numpy(), feat.to_numpy(), atol=1e-9)


def test_elo_source(synthetic):
    rk = R.build_rankings(synthetic, _slate(synthetic).asof, "elo", top=3)
    assert rk["source"]["unit"] == "Elo" and rk["source"]["columns"] == []
    assert [t["rank"] for t in rk["teams"]] == [1, 2, 3]


def test_custom_source(synthetic, monkeypatch):
    mod = types.ModuleType("fake_model_rating")

    def rate(view):
        ids = R._fbs_ids(view)
        return pd.DataFrame({"team_id": ids, "rating": -ids.astype(float)})  # lowest id is best

    rate.label, rate.unit = "Fake model", "p"
    mod.rate = rate
    monkeypatch.setitem(sys.modules, "fake_model_rating", mod)
    rk = R.build_rankings(synthetic, _slate(synthetic).asof, "fake_model_rating:rate", top=8)
    ids = [t["team_id"] for t in rk["teams"]]
    assert ids == sorted(ids) and rk["source"]["label"] == "Fake model"
    with pytest.raises(ValueError):
        R.resolve_source("no_such_source")
    bad = R.RatingSource("bad", "Bad", "", "", lambda view: pd.DataFrame({"team_id": [1]}))
    with pytest.raises(ValueError):
        R.build_rankings(synthetic, _slate(synthetic).asof, bad)


# ---- polls --------------------------------------------------------------------------------------

class _PollView:
    season = 2026

    def __init__(self, frame):
        self.frame = frame

    def has(self, name):
        return name == "rankings"

    def table(self, name, seasons=None):
        return self.frame


def _poll_rows(poll, week, order, season_type="regular"):
    return pd.DataFrame({"season": 2026, "seasonType": season_type, "week": week, "poll": poll,
                         "rank": range(1, len(order) + 1), "teamId": order, "points": 0})


def test_latest_polls_pick_latest_week_and_top_25():
    frame = pd.concat([
        _poll_rows("AP Top 25", 11, list(range(1, 27))),
        _poll_rows("AP Top 25", 12, list(range(26, 0, -1))),  # 26 rows: rank 26 is dropped
        _poll_rows("Playoff Committee Rankings", 11, [5, 4, 3]),
        _poll_rows("FCS Coaches Poll", 12, [900, 901]),
    ], ignore_index=True)
    polls = R.latest_polls(_PollView(frame))
    assert set(polls) == {"ap", "cfp"}
    assert polls["ap"]["week"] == 12 and polls["ap"]["ranks"][26] == 1 and 1 not in polls["ap"]["ranks"]
    assert polls["cfp"]["week"] == 11 and polls["cfp"]["ranks"] == {5: 1, 4: 2, 3: 3}


def test_compare_counts_and_disagreements():
    poll = {"short": "AP", "week": 12, "season_type": "regular",
            "ranks": {t: t for t in range(1, 26)}}            # poll: team t at rank t
    model = {t: t for t in range(1, 51)}
    model[3], model[40] = 40, 3                               # model: team 3 at 40, team 40 at 3
    c = R._compare(poll, model, {t: f"T{t}" for t in model})
    assert c["n"] == 25 and c["overlap"] == 24
    assert c["model_lower"][0] == {"team_id": 3, "team": "T3", "poll_rank": 3, "model_rank": 40, "diff": 37}
    assert [d["team_id"] for d in c["model_only"]] == [40]
    assert c["mean_abs_diff"] == pytest.approx(37 / 25, abs=0.05)
    assert -1 <= c["spearman"] < 1


# ---- report integration -------------------------------------------------------------------------

def _bundle(wh, cfg):
    slate = _slate(wh)
    sched = wh.schedule([slate.season])
    games = sched[sched["cutoff_utc"] <= slate.last_cutoff]
    preds = {m: run_baseline(wh, m, [slate.season], games=games) for m in cfg.models}
    return build_bundle(wh, slate, preds, cfg), preds


def test_bundle_markdown_and_dashboard(synthetic):
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",),
                       rankings_top=4)
    b, _ = _bundle(synthetic, cfg)
    assert b["rankings"]["top"] == 4 and sum(t["in_top"] for t in b["rankings"]["teams"]) == 4
    md = render_markdown(b)
    assert "## Model rankings vs polls" in md and "Power rating (pts)" in md
    page = render_dashboard(b)
    assert 'data-tab="rankings"' in page and '"rankings":{' in page


def test_rankings_off_and_failure(synthetic):
    off = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",),
                       rankings_source="")
    b, _ = _bundle(synthetic, off)
    assert b["rankings"] is None and "Model rankings" not in render_markdown(b)
    broken = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",),
                          rankings_source="no_such_module_xyz:rate")
    b, _ = _bundle(synthetic, broken)
    assert b["rankings"] is None and any("rankings could not be built" in n for n in b["meta"]["notes"])


def test_rankings_are_not_a_model_input(synthetic):
    """No feature group comes from the rankings, and predictions do not depend on them."""
    assert all(not type(g).__module__.startswith("cfb.report") for g in discover_groups())
    on = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",))
    off = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",),
                       rankings_source="")
    b_on, p_on = _bundle(synthetic, on)
    b_off, p_off = _bundle(synthetic, off)
    for m in p_on:
        cols = ["game_id", "p_home_win"]
        pd.testing.assert_frame_equal(p_on[m][cols], p_off[m][cols])
    assert [g["p"] for g in b_on["games"]] == [g["p"] for g in b_off["games"]]


def test_config_rankings_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[rankings]\nsource = "elo"\ntop = 25\n')
    cfg = load_config(p)
    assert cfg.rankings_source == "elo" and cfg.rankings_top == 25
    assert load_config("dashboard/config.toml").rankings_source == "power"
    with pytest.raises(ValueError):
        ReportConfig(rankings_top=0)


def test_cli_prints_markdown(synthetic, capsys):
    assert R.main(["--data-dir", str(synthetic.root), "--season", str(LIVE), "--week", "4", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert f"# {LIVE} week 4" in out and "| 3 |" in out
