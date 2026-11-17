"""W9: run configuration, slate resolution, report bundle, Markdown report, dashboard, run_week."""
import json
import re
import shutil
import sys
import types

import pandas as pd
import pytest

from cfb.core import Warehouse
from cfb.core.schemas import read_predictions, validate_metrics
from cfb.core.synthetic import SEASONS
from cfb.eval import run_baseline
from cfb.report import build_bundle, load_config, render_dashboard, render_markdown, resolve_slate
from cfb.report.config import ReportConfig
from cfb.report.slate import current_week
from cfb.run_week import main as run_week_main, run

LIVE = SEASONS[-1]  # synthetic: played through week 3, weeks 4-5 and the bowls scheduled


@pytest.fixture()
def store(synthetic, tmp_path):
    """A private copy of the synthetic warehouse; run_week writes into it."""
    root = tmp_path / "data"
    shutil.copytree(synthetic.root, root)
    return Warehouse(root)


def _predict(wh, slate, models):
    sched = wh.schedule([slate.season])
    games = sched[sched["cutoff_utc"] <= slate.last_cutoff]
    return {m: run_baseline(wh, m, [slate.season], games=games) for m in models}


# ---- configuration -------------------------------------------------------------------------

def test_config_file_and_overrides(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[models]\nprimary = "model_v1"\nbaselines = ["baseline_elo"]\n'
                 '[producers]\nmodel_v1 = "fake_w8:produce"\n[run]\nbuild_features = false\n')
    cfg = load_config(p)
    assert cfg.primary == "model_v1" and cfg.baselines == ("baseline_elo",) and not cfg.build_features
    assert cfg.models[0] == "model_v1" and "baseline_moneyline" in cfg.models  # market models are always added
    assert cfg.comparison == ["baseline_elo"]
    assert load_config(p, primary="baseline_home").primary == "baseline_home"
    assert load_config(None).primary  # repo default parses
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
    with pytest.raises(ValueError):
        ReportConfig(primary="Bad Id")
    with pytest.raises(ValueError):
        ReportConfig(producers={"model_v1": "no_colon"})


def test_repo_config_is_valid():
    cfg = load_config("dashboard/config.toml")
    assert cfg.primary in cfg.models
    assert all(cfg.producer(m) is not None for m in cfg.baselines)


def test_producer_lookup(monkeypatch):
    mod = types.ModuleType("fake_w8")
    mod.produce = lambda warehouse, games: None
    monkeypatch.setitem(sys.modules, "fake_w8", mod)
    cfg = ReportConfig(primary="model_v1", producers={"model_v1": "fake_w8:produce"})
    assert cfg.producer("model_v1") is mod.produce
    assert cfg.producer("baseline_elo") is not None
    assert cfg.producer("some_other_model") is None  # read from data/predictions


# ---- slate -----------------------------------------------------------------------------------

def test_current_week_from_calendar(synthetic):
    weeks = synthetic.meta["weeks"]
    live = weeks[weeks["season"] == LIVE].sort_values("start_utc")
    w4 = live[(live["season_type"] == "regular") & (live["week"] == 4)].iloc[0]
    assert current_week(weeks, w4["start_utc"] + pd.Timedelta(hours=30)) == (LIVE, "regular", 4)
    assert current_week(weeks, pd.Timestamp("2000-01-01", tz="UTC"))[0] == SEASONS[0]  # before: next week to start
    last = weeks.sort_values("start_utc").iloc[-1]
    assert current_week(weeks, last["end_utc"] + pd.Timedelta(days=400))[0] == int(last["season"])


def test_resolve_slate_regular_week(synthetic):
    s = resolve_slate(synthetic, LIVE, "regular", 4)
    gi = synthetic.game_index
    assert set(s.games["game_id"]) == set(gi[(gi["season"] == LIVE) & (gi["week"] == 4)
                                             & (gi["season_type"] == "regular")]["game_id"])
    assert (s.previous["week"] == 3).all() and len(s.previous)
    assert s.asof == s.games["cutoff_utc"].min()
    assert s.key == f"{LIVE}_regular_04" and s.label == f"{LIVE} week 4"
    assert "home_points" not in s.games.columns  # schedule rows, no results
    first = resolve_slate(synthetic, LIVE, "regular", 1)
    assert first.previous.empty


def test_resolve_slate_postseason(synthetic):
    gi = synthetic.game_index
    bowls = gi[(gi["season"] == SEASONS[1]) & (gi["season_type"] == "postseason")].sort_values("start_time_utc")
    whole = resolve_slate(synthetic, SEASONS[1], "postseason", 1)
    assert len(whole.games) == len(bowls) and (whole.previous["season_type"] == "regular").all()
    now = bowls["start_time_utc"].iloc[0] + pd.Timedelta(hours=1)  # the first bowl is under way
    live = resolve_slate(synthetic, SEASONS[1], "postseason", 1, now=now)
    assert set(live.games["game_id"]) <= set(bowls["game_id"].iloc[1:])
    assert bowls["game_id"].iloc[0] in set(live.previous["game_id"])
    assert live.asof == now


# ---- bundle and renderers --------------------------------------------------------------------

def test_bundle_contents(synthetic):
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_home", "baseline_close"),
                       market_models=("baseline_close",), build_features=False)
    slate = resolve_slate(synthetic, LIVE, "regular", 4)
    preds = _predict(synthetic, slate, cfg.models)
    b = build_bundle(synthetic, slate, preds, cfg, data_dir=synthetic.root)
    json.dumps(b, allow_nan=False)  # JSON-safe
    assert b["meta"]["primary"] == "baseline_elo" and b["meta"]["comparison"] == ["baseline_home", "baseline_close"]
    assert len(b["games"]) == len(slate.games)
    g = next(x for x in b["games"] if x["pick"])
    assert g["result"] is None  # no results for the target week
    assert g["pick"]["p"] >= 0.5 and set(g["p"]) >= {"baseline_elo", "baseline_home"}
    assert g["market"]["spread"] is not None and g["market"]["source"] == "baseline_close"
    assert g["edge"] == pytest.approx(g["p"]["baseline_elo"] - g["p"]["baseline_close"], abs=1e-3)
    prev = b["previous"]
    assert prev["games"] and all(x["result"] is not None for x in prev["games"])
    assert prev["summary"]["baseline_elo"]["n"] == sum(x["fbs_vs_fbs"] for x in prev["games"])
    season = b["season"]
    # Season scoring uses only results visible at the report's cutoff: weeks 1-3.
    weeks = {(w["season_type"], w["week"]) for w in season["weekly"]}
    assert weeks <= {("regular", 1), ("regular", 2), ("regular", 3)}
    assert season["common"]["models"][0] == "baseline_elo"
    assert {c["model"] for c in season["comparisons"]} == {"baseline_home", "baseline_close"}
    for bins in season["calibration"].values():
        assert len(bins) == 10
    assert len(b["teams"]) == 8  # FBS teams only
    t = b["teams"]["1"]
    assert t["games"] and all(x["opp"] for x in t["games"])
    played = [x for x in t["games"] if x["result"]]
    assert all(x["week"] <= 3 for x in played)
    assert b["meta"]["phase2"] is False


def test_markdown_report(synthetic):
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",))
    slate = resolve_slate(synthetic, LIVE, "regular", 4)
    b = build_bundle(synthetic, slate, _predict(synthetic, slate, cfg.models), cfg)
    md = render_markdown(b)
    assert md.startswith(f"# Weekly report: {LIVE} week 4")
    for section in ("## Season to date", f"## Last week: {LIVE} week 3", f"## This week: {LIVE} week 4",
                    "### Reasons per pick"):
        assert section in md
    assert "## Spread and total" not in md  # phase 2 only when the model predicts a margin
    # every game of the week has a table row
    for g in b["games"]:
        assert g["home"]["name"] in md


def test_markdown_phase2_section(synthetic):
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",))
    slate = resolve_slate(synthetic, LIVE, "regular", 4)
    preds = _predict(synthetic, slate, cfg.models)
    preds["baseline_elo"] = preds["baseline_elo"].assign(pred_margin=3.5, pred_total=51.0)
    b = build_bundle(synthetic, slate, preds, cfg)
    assert b["meta"]["phase2"] is True
    md = render_markdown(b)
    assert "## Spread and total" in md and "+3.5" in md


def test_dashboard_embeds_bundle_safely(synthetic):
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_home",), market_models=("baseline_close",))
    slate = resolve_slate(synthetic, LIVE, "regular", 4)
    b = build_bundle(synthetic, slate, _predict(synthetic, slate, cfg.models), cfg)
    b["games"][0]["home"]["name"] = "Evil </script><script>alert(1)</script> U"
    page = render_dashboard(b)
    assert page.lower().count("</script>") == 2  # the bundle's own and the app's
    data = re.search(r'<script id="bundle" type="application/json">(.*?)</script>', page, re.S).group(1)
    back = json.loads(data)
    assert back["games"][0]["home"]["name"].startswith("Evil </script>")
    assert back["meta"]["label"] == f"{LIVE} week 4"
    assert "__BUNDLE__" not in page and "__TITLE__" not in page
    assert "<title>CFB " in page


# ---- run_week --------------------------------------------------------------------------------

def test_run_week_end_to_end(store, tmp_path):
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_close", "baseline_home"),
                       market_models=("baseline_close",), build_features=False)
    pub = tmp_path / "published"
    res = run(store.root, cfg, season=LIVE, week=4, do_refresh=False, publish_dir=pub)
    for key in ("report", "dashboard", "bundle"):
        assert res.files[key].exists()
    assert res.out_dir == store.root / "reports" / f"{LIVE}_regular_04"
    metrics = json.loads(res.files["metrics_baseline_elo"].read_text())
    validate_metrics(metrics)
    assert {r["season"] for r in metrics["by_season"]} == {LIVE}
    assert not (store.root / "eval").exists()  # validation metrics in data/eval are left alone
    # predictions were written in the contract, and only through the target week
    p = read_predictions(store.root, "baseline_elo")
    assert set(p["season"]) == {LIVE} and p["week"].max() == 4
    assert (pub / f"{LIVE}_regular_04" / "dashboard.html").exists()
    assert (pub / "latest" / "KEY").read_text().strip() == f"{LIVE}_regular_04"
    assert any("baseline" in n for n in res.notes)


def test_switching_to_the_model_is_a_config_change(store, tmp_path, monkeypatch):
    """The plan's done-criterion: a producer entry and `primary` switch the report to the model."""
    calls = {}

    def produce(warehouse, games):
        calls["games"] = len(games)
        out = run_baseline(warehouse, "baseline_elo", sorted(games["season"].unique()), games=games)
        out = out.assign(model_id="model_v1", p_home_win=out["p_home_win"] * 0.9 + 0.05,
                         reasons='[{"text":"Power rating gap","feature":"str_pwr_net","effect":0.8}]')
        return out

    mod = types.ModuleType("fake_w8")
    mod.produce = produce
    monkeypatch.setitem(sys.modules, "fake_w8", mod)
    conf = tmp_path / "config.toml"
    conf.write_text('[models]\nprimary = "model_v1"\nbaselines = ["baseline_elo", "baseline_close"]\n'
                    '[producers]\nmodel_v1 = "fake_w8:produce"\n'
                    '[run]\nbuild_features = false\nmarket_models = ["baseline_close"]\n')
    rc = run_week_main(["--data-dir", str(store.root), "--config", str(conf), "--season", str(LIVE), "--week", "4",
                        "--no-refresh", "--out", str(tmp_path / "out")])
    assert rc == 0 and calls["games"] > 0
    b = json.loads((tmp_path / "out" / "bundle.json").read_text())
    assert b["meta"]["primary"] == "model_v1" and b["meta"]["comparison"] == ["baseline_elo", "baseline_close"]
    g = next(x for x in b["games"] if x["pick"])
    assert g["reasons"][0]["text"] == "Power rating gap"
    assert not any("baseline until" in n for n in b["meta"]["notes"])
    assert "Power rating gap" in (tmp_path / "out" / "report.md").read_text()
    assert read_predictions(store.root, "model_v1")["model_id"].eq("model_v1").all()


def test_stored_predictions_are_read_when_no_producer(store, tmp_path):
    from cfb.eval.splits import write_weekly
    sched = store.schedule([LIVE])
    pre = run_baseline(store, "baseline_home", [LIVE], games=sched).assign(model_id="model_v1")
    write_weekly(pre, store.root)
    cfg = ReportConfig(primary="model_v1", baselines=("baseline_home",), market_models=("baseline_close",),
                       build_features=False)
    res = run(store.root, cfg, season=LIVE, week=4, do_refresh=False)
    b = json.loads(res.files["bundle"].read_text())
    assert any(g["p"].get("model_v1") is not None for g in b["games"])
    # games after the target week are not part of the report
    assert all(g["week"] == 4 for g in b["games"])


def test_refresh_requires_the_archive(store):
    cfg = ReportConfig(build_features=False)
    with pytest.raises(SystemExit, match="archive"):
        run(store.root, cfg, season=LIVE, week=4, do_refresh=True, archive=None)


def test_missing_primary_predictions_stop_the_run(store):
    cfg = ReportConfig(primary="model_none", baselines=("baseline_home",), build_features=False)
    with pytest.raises(SystemExit, match="model_none"):
        run(store.root, cfg, season=LIVE, week=4, do_refresh=False)


# ---- real data (skipped in CI) -------------------------------------------------------------

def test_week12_2026_on_snapshot(snapshot, tmp_path):
    """The plan's done-criterion on real data: a week 12 2026 run from baseline predictions."""
    if 12 not in set(snapshot.game_index.query("season == 2026")["week"]):
        pytest.skip("snapshot has no 2026 week 12 schedule")
    cfg = load_config("dashboard/config.toml", build_features=False)
    res = run(snapshot.root, cfg, season=2026, week=12, do_refresh=False, out_dir=tmp_path / "r",
              write_predictions=False)
    b = json.loads(res.files["bundle"].read_text())
    assert b["meta"]["label"] == "2026 week 12" and len(b["games"]) == 70
    assert b["previous"]["label"] == "2026 week 11"
    elo = b["season"]["own"]["baseline_elo"]
    assert elo["n"] == 629  # 2026 FBS vs FBS completed games through week 11
    assert len(b["teams"]) >= 130
