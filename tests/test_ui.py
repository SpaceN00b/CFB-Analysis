"""Local app: data helpers on a synthetic bundle, and each page rendered headless."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfb.ui import data as D


def team(tid, name, conf="SEC", rank=None):
    return {"id": tid, "name": name, "conference": conf, "fbs": True, "rank": rank, "record": "5-5", "elo": 1500.0}


def game(gid, home, away, p_home, result=None):
    pick = {"side": "home", "team": home["name"], "p": p_home} if p_home >= 0.5 else \
        {"side": "away", "team": away["name"], "p": 1 - p_home}
    return {
        "game_id": gid, "season": 2026, "season_type": "regular", "week": 12,
        "kickoff_utc": "2026-11-21T17:00:00Z", "kickoff_local": "Sat Nov 21, 12:00 PM EST",
        "neutral": False, "conference_game": True, "fbs_vs_fbs": True, "home": home, "away": away,
        "p": {"model_v2": p_home, "baseline_close": 0.6}, "pick": pick,
        "reasons": [{"text": "Power rating gap", "effect": 0.2, "feature": "d_str_pwr_net", "value": 5.0}],
        "market": {"spread": -3.5, "total": 50.0, "p_home": 0.6}, "edge": p_home - 0.6,
        "pred_margin": 5.0, "pred_total": 52.0,
        "ats": {"side": "home", "team": home["name"], "line": -3.5, "edge": 1.5, "model_line": -5.0},
        "ou": {"side": "over", "line": 50.0, "edge": 2.0}, "result": result, "matchup": {},
    }


def make_bundle() -> dict:
    a, b, c = team(1, "Alpha", rank=3), team(2, "Beta"), team(3, "Gamma", conf="Big Ten")
    return {
        "bundle_version": 1,
        "meta": {"season": 2026, "season_type": "regular", "week": 12, "label": "2026 week 12",
                 "key": "2026_regular_12", "primary": "model_v2", "primary_label": "Model v2",
                 "data_through_utc": "2026-11-15T18:00:00Z", "lines_asof_local": "Tue"},
        "models": [{"id": "model_v2", "label": "Model v2"}, {"id": "baseline_close", "label": "Closing spread"}],
        "games": [game(10, a, b, 0.7), game(11, c, b, 0.4)],
        "previous": {"label": "2026 week 11", "games": [
            game(9, b, a, 0.3, {"home_points": 10, "away_points": 20, "correct": True})]},
        "season": {"scope": "FBS vs FBS",
                   "own": {"model_v2": {"label": "Model v2", "n": 10, "log_loss": 0.55, "brier": 0.19, "accuracy": 0.7}},
                   "weekly": [{"week": 1, "n": 5, "cumulative": {"model_v2": 0.5, "baseline_close": 0.45}},
                              {"week": 2, "n": 5, "cumulative": {"model_v2": 0.52, "baseline_close": 0.47}}],
                   "calibration": {"model_v2": [{"p_lo": 0.5, "p_hi": 0.6, "n": 4, "p_mean": 0.55, "win_rate": 0.5}]}},
        "season_points": {"scope": "FBS vs FBS", "margin": {"n": 10, "mae": 12.0}},
        "teams": {"1": {"id": 1, "name": "Alpha", "conference": "SEC", "record": "9-1",
                        "games": [{"week": 1, "date": "Sep 5", "opp": "Beta", "site": "home", "p_win": 0.7,
                                   "result": {"pts": 20, "opp_pts": 10, "outcome": "W"}}]},
                  "2": {"id": 2, "name": "Beta", "conference": "SEC", "record": "5-5", "games": []},
                  "3": {"id": 3, "name": "Gamma", "conference": "Big Ten", "record": "6-4", "games": []}},
        "rankings": {"source": {"label": "Power rating", "note": "test"},
                     "teams": [{"rank": 1, "move": 0, "team": "Alpha", "conference": "SEC", "record": "9-1",
                                "rating": 20.0, "off": 35.0, "def": 15.0, "sor_rank": 2,
                                "polls": {"ap": 3, "coaches": 2, "cfp": None}}]},
    }


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    for key in ("2026_regular_12", "2026_regular_3", "2025_postseason_1"):
        (d / "reports" / key).mkdir(parents=True)
        (d / "reports" / key / "bundle.json").write_text(json.dumps(make_bundle()), encoding="utf-8")
    (d / "reports" / "latest").mkdir()  # no bundle: ignored
    monkeypatch.setenv("CFB_DATA_DIR", str(d))
    monkeypatch.setenv("CFB_PROJECT_FILES", str(tmp_path / "no-mount"))
    monkeypatch.setenv("CFB_WAREHOUSE_DIR", str(tmp_path / "warehouse"))
    return d


def test_list_reports_newest_first(data_dir):
    assert [k for k, _ in D.list_reports(data_dir)] == ["2026_regular_12", "2026_regular_3", "2025_postseason_1"]
    assert D.list_reports(data_dir / "missing") == []


def test_frames():
    b = make_bundle()
    g = D.games_frame(b)
    assert list(g["game_id"]) == [10, 11]
    assert g.loc[0, "home"] == "#3 Alpha" and g.loc[0, "away"] == "Beta"
    assert g.loc[1, "pick"] == "Beta" and abs(g.loc[1, "p_pick"] - 0.6) < 1e-9
    assert g.loc[0, "conferences"] == ["SEC"]
    assert D.games_frame(b, b["previous"]["games"]).loc[0, "score"] == "Alpha 20 - Beta 10"
    assert D.game_by_id(b, 9)["game_id"] == 9
    assert list(D.weekly_frame(b).columns) == ["Model v2", "Closing spread"]
    assert D.rankings_frame(b).loc[0, "AP"] == 3
    assert D.teams_list(b)[0] == (1, "Alpha")
    assert D.team_games_frame(b, 1).loc[0, "result"] == "W 20-10"
    assert D.team_games_frame(b, 99).empty
    assert D.own_metrics_frame(b).loc[0, "model"] == "Model v2"


def test_resolve_data_dir_falls_back_to_settings(data_dir):
    assert D.resolve_data_dir() == data_dir


st_testing = pytest.importorskip("streamlit.testing.v1")


def run_page(name):
    def script(page_name):
        from cfb.ui import pages
        getattr(pages, page_name)()

    at = st_testing.AppTest.from_function(script, args=(name,), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_week_page(data_dir):
    at = run_page("page_week")
    assert at.title[0].value == "2026 week 12"
    assert len(at.dataframe) == 1


@pytest.mark.parametrize("name", ["page_teams", "page_performance", "page_data", "page_settings", "page_matchup",
                                  "page_results"])
def test_other_pages_render(data_dir, name):
    run_page(name)


def test_pages_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("CFB_WAREHOUSE_DIR", str(tmp_path / "warehouse"))
    at = run_page("page_week")
    assert at.warning


def test_launch_argv():
    from cfb.ui import launch

    argv = launch.streamlit_argv(8600, browser=False)
    assert argv[:3] == ["streamlit", "run", str(Path(launch.__file__).with_name("app.py"))]
    assert "--server.address=127.0.0.1" in argv and "--server.headless=true" in argv
    assert 0 < launch.free_port(0) < 65536


class FakeResp:
    def __init__(self, status, doc):
        self.status_code, self._doc = status, doc

    def json(self):
        return self._doc


class FakeSession:
    def __init__(self, resp):
        self.resp = resp

    def get(self, *a, **k):
        return self.resp


def test_update_check(monkeypatch):
    from cfb.ui import update

    doc = {"tag_name": "build-12", "html_url": "https://example.invalid/r",
           "assets": [{"name": "CFB-Predictor-Setup-build-12.exe", "browser_download_url": "https://example.invalid/s.exe"}]}
    rel = update.latest_release(session=FakeSession(FakeResp(200, doc)))
    assert rel == {"build": 12, "tag": "build-12", "url": "https://example.invalid/r",
                   "installer_url": "https://example.invalid/s.exe"}
    assert update.latest_release(session=FakeSession(FakeResp(404, {}))) is None
    assert update.latest_release(session=FakeSession(FakeResp(200, {"tag_name": "v1"}))) is None
    # A source checkout has no build number and never offers updates.
    monkeypatch.setattr(update, "current_build", lambda: None)
    assert update.update_available(session=FakeSession(FakeResp(200, doc))) is None
    monkeypatch.setattr(update, "current_build", lambda: 11)
    assert update.update_available(session=FakeSession(FakeResp(200, doc)))["build"] == 12
    monkeypatch.setattr(update, "current_build", lambda: 12)
    assert update.update_available(session=FakeSession(FakeResp(200, doc))) is None


def test_update_check_experimental_sees_only_its_own(monkeypatch):
    from cfb.ui import update

    monkeypatch.setenv("CFB_EDITION", "experimental")
    docs = [{"tag_name": "build-30", "assets": []},
            {"tag_name": "exp-build-21", "prerelease": True, "html_url": "u21", "assets": []},
            {"tag_name": "exp-build-25", "prerelease": True, "html_url": "u25", "draft": True, "assets": []},
            {"tag_name": "exp-build-19", "prerelease": True, "html_url": "u19", "assets": []}]
    rel = update.latest_release(session=FakeSession(FakeResp(200, docs)))
    assert rel["tag"] == "exp-build-21" and rel["build"] == 21
    assert update.latest_release(session=FakeSession(FakeResp(200, docs[:1]))) is None


def test_sync_job_runs_in_background_and_reports():
    import threading
    from cfb.ui.sync_job import SyncJob

    gate, logged, job = threading.Event(), threading.Event(), SyncJob()

    class R:
        changed, version = True, "v1"

    def run(log):
        log("syncing v1")
        logged.set()
        gate.wait(5)
        return R()

    assert job.start(run) and job.running
    assert not job.start(run)  # a second click while running starts nothing
    assert logged.wait(5) and job.snapshot().lines == ["syncing v1"]
    gate.set()
    job.wait(5)
    snap = job.snapshot()
    assert snap.state == "done" and snap.result.version == "v1"
    assert job.take_fresh() and not job.take_fresh()

    def boom(log):
        raise RuntimeError("bad key")

    assert job.start(boom)
    job.wait(5)
    assert job.snapshot().state == "failed" and str(job.snapshot().error) == "bad key"
    assert not job.take_fresh()
