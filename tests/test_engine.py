"""cfb.engine: bootstrap from a fake bucket, the weekly run on the synthetic store, backup, check,
status file, lock and off-season; cfb.edition and cfb.schedule; the engine parts of the app."""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from cfb import edition, engine as E, schedule as S, settings
from cfb import warehouse_sync as ws
from cfb.core.synthetic import SEASONS

LIVE = max(SEASONS)
CONFIG = ('[models]\nprimary = "baseline_elo"\nbaselines = ["baseline_close", "baseline_home"]\n'
          '[run]\nbuild_features = false\nmarket_models = ["baseline_close"]\n')


def quiet(_msg):
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Engine folders, data dir, dated-copy archive and a local bucket, all under tmp_path."""
    paths = {k: tmp_path / k for k in ("engine", "data", "asof", "bucket", "warehouse")}
    monkeypatch.setenv("CFB_ENGINE_DIR", str(paths["engine"]))
    monkeypatch.setenv("CFB_DATA_DIR", str(paths["data"]))
    monkeypatch.setenv("CFB_ASOF_ARCHIVE", str(paths["asof"]))
    monkeypatch.setenv("CFB_WAREHOUSE_LOCAL", str(paths["bucket"]))
    monkeypatch.setenv("CFB_WAREHOUSE_DIR", str(paths["warehouse"]))
    monkeypatch.setenv("CFB_PROJECT_FILES", str(tmp_path / "no-mount"))
    monkeypatch.setattr(ws, "cfbd_quota", lambda: {"tier": "Tier 3", "monthly_limit": 75000, "remaining": 74000})
    conf = tmp_path / "config.toml"
    conf.write_text(CONFIG, encoding="utf-8")
    paths["config"] = conf
    return paths


@pytest.fixture
def seeded(env, synthetic, tmp_path):
    """The cloud side: a data dir with a report and a dated copy, published to the full channel."""
    from cfb.report.config import load_config
    from cfb.run_week import run

    src = tmp_path / "cloud" / "data"
    shutil.copytree(synthetic.root, src)
    asof = src / "raw" / "ratings_sp_asof"
    asof.mkdir(parents=True)
    with gzip.open(asof / f"{LIVE}_{LIVE}1001.json.gz", "wt", encoding="utf-8") as fh:
        json.dump([{"team": "Team A", "rating": 10}], fh)
    run(src, load_config(env["config"]), season=LIVE, week=4, do_refresh=False)
    ws.publish(src, ws.LocalStore(env["bucket"]), channel="full", log=quiet)
    return src


def engine(*args) -> int:
    return E.main(list(args))


# ---- bootstrap, weekly run, backup, check ------------------------------------------------------

def test_bootstrap_from_the_full_channel(seeded, env):
    assert not E.is_bootstrapped()
    assert engine("bootstrap") == 0
    d = env["data"]
    assert E.is_bootstrapped(d) and (d / "raw" / "games").is_dir() and (d / "core" / "game_index.parquet").exists()
    assert not (d / "manifest.json").exists()
    boot = E.bootstrap_info()
    assert boot["source"] == "r2" and boot["files"] > 100
    assert (env["asof"] / "raw" / "ratings_sp_asof" / f"{LIVE}_{LIVE}1001.json.gz").exists()  # dated copy archived
    assert E.latest_reference().parent.name == f"{LIVE}_regular_04"
    assert not list(d.parent.glob(".bootstrap-*"))
    st = E.read_status()
    assert st["state"] == "ok" and st["command"] == "bootstrap" and Path(st["log"]).exists()
    assert "bootstrap: " in Path(st["log"]).read_text(encoding="utf-8")
    # a second bootstrap leaves the data alone
    assert engine("bootstrap") == 0 and E.read_status()["state"] == "ok"
    # --force keeps the old data dir aside
    assert engine("bootstrap", "--force") == 0
    assert len(list(d.parent.glob("data.old-*"))) == 1


def test_all_runs_end_to_end_and_backs_up(seeded, env):
    rc = engine("all", "--season", str(LIVE), "--week", "4", "--no-refresh", "--config", str(env["config"]),
                "--trigger", "scheduled")
    st = E.read_status()
    assert rc == 0, st
    assert st["state"] == "ok" and st["trigger"] == "scheduled" and st["report_key"] == f"{LIVE}_regular_04"
    assert [s["name"] for s in st["steps"]] == ["bootstrap", "weekly run", "backup"]
    assert all(s["state"] == "done" for s in st["steps"])
    assert (env["data"] / "reports" / f"{LIVE}_regular_04" / "bundle.json").exists()
    assert st["backup"]["state"] == "done" and st["cfbd_quota"]["remaining"] == 74000
    bucket = env["bucket"]
    slim = json.loads((bucket / "manifest.json").read_bytes())
    full = json.loads((bucket / "full" / "manifest.json").read_bytes())
    assert not any(k.startswith("raw/") for k in slim["files"]) and any(k.startswith("raw/") for k in full["files"])
    assert full["key"] == f"{LIVE}_regular_04"
    assert E.history()[0]["command"] == "all"
    # the app reads the engine's data and sees the run
    from cfb.ui import data as D
    assert D.resolve_data_dir() == env["data"]


def test_check_reproduces_the_reference(seeded, env):
    assert engine("bootstrap") == 0
    assert engine("check", "--config", str(env["config"])) == 0
    chk = E.read_status()["check"]
    assert chk["ok"] and chk["games"] > 0 and chk["values"] >= chk["games"]
    assert chk["max_diff"]["p"] < 1e-9


def test_compare_bundles_flags_differences():
    ref = {"games": [{"game_id": 1, "p": {"m": 0.6}, "pred_margin": 3.0, "pred_total": None},
                     {"game_id": 2, "p": {"m": 0.5}, "pred_margin": None, "pred_total": None}]}
    same = json.loads(json.dumps(ref))
    assert E.compare_bundles(ref, same)["ok"]
    off = json.loads(json.dumps(ref))
    off["games"][0]["p"]["m"] = 0.6003
    out = E.compare_bundles(ref, off)
    assert not out["ok"] and out["n_mismatches"] == 1 and abs(out["max_diff"]["p"] - 0.0003) < 1e-9
    assert E.compare_bundles(ref, {"games": off["games"][:1]})["missing_games"] == [2]


def test_all_without_data_or_keys_fails_with_a_message(env, monkeypatch):
    monkeypatch.delenv("CFB_WAREHOUSE_LOCAL")
    assert engine("all", "--no-refresh") == E.EXIT_FAILED
    st = E.read_status()
    assert st["state"] == "failed" and "R2 keys" in st["error"]


def test_backup_failure_is_a_warning(seeded, env, monkeypatch):
    assert engine("bootstrap") == 0

    def boom(*a, **k):
        raise ws.WarehouseError("HTTP 403 AccessDenied")

    monkeypatch.setattr(ws, "publish", boom)
    assert engine("backup") == E.EXIT_WARNING
    st = E.read_status()
    assert st["state"] == "warning" and "read-only" in st["backup"]["error"]


def test_backup_preference_off(seeded, env):
    E.save_prefs(backup=False)
    assert E.load_prefs()["backup"] is False
    assert engine("all", "--season", str(LIVE), "--week", "4", "--no-refresh", "--config", str(env["config"])) == 0
    assert E.read_status()["backup"] == {"state": "off"}
    assert not (env["bucket"] / "manifest.json").exists()


def test_light_pulls_only_the_light_datasets(seeded, env, monkeypatch):
    from cfb import run_week as RW

    assert engine("bootstrap") == 0
    calls = {}

    def fake_refresh(data_dir, season, archive, now, names=None, stats=None):
        calls.update(season=season, names=names, archive=archive)
        stats["api_calls"] = 7
        return []

    monkeypatch.setattr(RW, "refresh", fake_refresh)
    monkeypatch.setattr(E, "_now", lambda: datetime(2026, 11, 20, 12, tzinfo=timezone.utc))
    real_week = E.week
    monkeypatch.setattr(E, "week", lambda run, **kw: real_week(run, **{**kw, "season": LIVE, "week_no": 4}))
    assert engine("light", "--config", str(env["config"]), "--no-backup") == 0
    assert calls["names"] == list(E.LIGHT_DATASETS) and calls["season"] == 2026
    assert calls["archive"] == env["asof"]
    st = E.read_status()
    assert st["api_calls"] == 7 and [s["name"] for s in st["steps"]] == ["refresh (light)", "core", "weekly run"]


def test_off_season_skips(env, monkeypatch):
    monkeypatch.setattr(E, "_now", lambda: datetime(2027, 3, 2, 10, tzinfo=timezone.utc))
    monkeypatch.setattr(E, "bootstrap", lambda run, **kw: "present")
    assert engine("all") == 0
    st = E.read_status()
    assert st["state"] == "skipped" and "off-season" in st["notes"][0]


def test_lock_makes_a_second_run_wait(env):
    from cfb.warehouse_sync import _Lock

    with _Lock(E.engine_dir() / "engine.lock"):
        assert engine("bootstrap") == E.EXIT_BUSY


def test_stale_running_status_reads_as_interrupted(env):
    E._write_json(E.status_path(), {"state": "running", "pid": 999999999, "started_utc": "2026-11-20T05:30:00Z",
                                    "heartbeat_utc": "2026-11-20T05:30:00Z", "command": "all"})
    assert E.read_status()["state"] == "interrupted"
    import os
    E._write_json(E.status_path(), {"state": "running", "pid": os.getpid(), "command": "all",
                                    "heartbeat_utc": E._iso(E._now())})
    assert E.read_status()["state"] == "running"


def test_status_command(env, capsys):
    assert engine("status") == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["edition"] == "main" and doc["bootstrapped"] is False and doc["asof_archive"] == str(env["asof"])


def test_logs_are_pruned(env, monkeypatch):
    monkeypatch.setattr(E, "KEEP_LOGS", 3)
    E.logs_dir().mkdir(parents=True)
    for i in range(6):
        (E.logs_dir() / f"engine-2026010{i}T000000Z-all.log").write_text("x")
    E._prune_logs(3)
    assert len(list(E.logs_dir().glob("engine-*.log"))) == 3


# ---- editions -------------------------------------------------------------------------------------

def test_editions_never_share_names():
    m, x = edition.EDITIONS["main"], edition.EDITIONS["experimental"]
    for field in ("app_name", "folder", "keyring_service", "bucket", "tag_prefix"):
        assert getattr(m, field) != getattr(x, field)
    assert m.task_name("weekly") != x.task_name("weekly")
    assert edition.current().name == "main"


def test_experimental_bucket_is_its_own(monkeypatch, tmp_path):
    monkeypatch.setenv("CFB_EDITION", "experimental")
    cfg = tmp_path / "r2.toml"
    cfg.write_text('account_id = "acct"\nbucket = "cfb-warehouse"\n', encoding="utf-8")
    assert ws.r2_settings(cfg, env={}).bucket == "cfb-warehouse-exp"  # `bucket` (main's) is ignored
    cfg.write_text('account_id = "acct"\nbucket_experimental = "my-exp"\n', encoding="utf-8")
    assert ws.r2_settings(cfg, env={}).bucket == "my-exp"
    with pytest.raises(ws.WarehouseError, match="main bucket"):
        ws.r2_settings(cfg, env={"R2_BUCKET": "cfb-warehouse"})
    monkeypatch.setenv("CFB_EDITION", "main")
    assert ws.r2_settings(cfg, env={}).bucket == "cfb-warehouse"
    monkeypatch.setenv("CFB_EDITION", "nope")
    with pytest.raises(ValueError):
        edition.current()


def test_experimental_reads_only_the_cfbd_key_from_main(monkeypatch):
    store = {("cfb", "cfbd_api_key"): "K", ("cfb", "r2_access_key_id"): "MAIN-R2"}
    kr = type("KR", (), {"get_password": staticmethod(lambda s, n: store.get((s, n)))})
    monkeypatch.setattr(settings, "_keyring", lambda: kr)
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.setattr(settings, "KEYRING_SERVICE", "cfb-exp")
    assert settings.secret("cfbd_api_key") == "K" and settings.secret_source("cfbd_api_key") == "main keyring"
    assert settings.secret("r2_access_key_id") is None and settings.secret_source("r2_access_key_id") is None


def test_build_stamp_sets_the_edition(monkeypatch):
    import types
    monkeypatch.delenv("CFB_EDITION", raising=False)
    monkeypatch.setitem(sys.modules, "cfb._build", types.SimpleNamespace(EDITION="experimental", BUILD="7"))
    assert edition.current().name == "experimental"


# ---- Task Scheduler ---------------------------------------------------------------------------------

def test_task_xml_round_trip_and_settings():
    plan = S.default_plan("weekly")
    assert plan.days == ("Monday",) and plan.time == "05:30"
    xml = S.task_xml(plan, r"C:\Apps\CFB Engine.exe", "engine all --trigger scheduled", r"C:\Apps",
                     user="HOME\\josh", start=date(2026, 11, 20))
    assert xml.startswith('<?xml version="1.0" encoding="UTF-16"?>')
    for tag in ("<StartWhenAvailable>true", "<WakeToRun>true", "<DisallowStartIfOnBatteries>false",
                "<LogonType>InteractiveToken", "<MultipleInstancesPolicy>IgnoreNew", "<UserId>HOME\\josh",
                "<StartBoundary>2026-11-20T05:30:00"):
        assert tag in xml
    got = S.parse_xml(xml)
    assert got["days"] == ["Monday"] and got["time"] == "05:30" and got["wake"]
    assert got["command"] == r"C:\Apps\CFB Engine.exe" and got["arguments"] == "engine all --trigger scheduled"
    daily = S.default_plan("daily")
    assert daily.days == S.DAYS[1:] and daily.time == "06:00"
    assert S.parse_xml(S.task_xml(S.Plan("daily", daily.days, "06:00", wake=False), "x", "y", "z"))["wake"] is False


def test_experimental_schedule_is_later():
    x = edition.EDITIONS["experimental"]
    assert S.default_plan("weekly", x).time == "06:00" and S.default_plan("daily", x).time == "06:30"
    assert S.task_name("weekly", x) == "CFB Predictor Experimental weekly update"


def test_plan_validation_and_days():
    assert S.parse_day("mon") == "Monday" and S.parse_day("Sunday") == "Sunday"
    for bad in (dict(kind="hourly", days=("Monday",), time="05:00"), dict(kind="weekly", days=(), time="05:00"),
                dict(kind="weekly", days=("Monday",), time="25:00")):
        with pytest.raises(S.ScheduleError):
            S.Plan(**bad)


def test_install_and_remove_call_schtasks(monkeypatch):
    seen = []

    class R:
        def __init__(self, rc=0, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def fake_run(args, timeout=30):
        seen.append(args)
        if args[1] == "/Create":
            text = Path(args[args.index("/XML") + 1]).read_text(encoding="utf-16")
            assert "<WakeToRun>true" in text
        if args[1] == "/Query":
            return R(0, S.task_xml(S.default_plan("weekly"), "p", "a", "w"))
        return R()

    monkeypatch.setattr(S, "available", lambda: True)
    monkeypatch.setattr(S, "_run", fake_run)
    monkeypatch.setattr(S, "run_info", lambda name: {"next": "soon"})
    assert S.install(S.default_plan("weekly"), command=("p", "a", "w")) == "CFB Predictor weekly update"
    assert seen[0][:4] == ["schtasks", "/Create", "/TN", "CFB Predictor weekly update"]
    shown = S.show("weekly")
    assert shown["installed"] and shown["next"] == "soon" and shown["days"] == ["Monday"]
    assert S.remove("weekly") is True and seen[-1][1] == "/Delete"


def test_task_result_codes_and_run_info(monkeypatch):
    assert S.describe_result(0) == "success" and S.describe_result(267011) == "not run yet"
    assert S.describe_result(-2147216609) == "already running" and S.describe_result(5) == "code 0x5"

    class R:
        returncode, stderr = 0, ""
        stdout = '{"next":"2026-11-23 05:30","last":null,"result":267011}'

    monkeypatch.setattr(S, "_run", lambda args, timeout=30: R())
    assert S.run_info("x") == {"next": "2026-11-23 05:30", "last": None, "result": None}
    R.stdout = '{"next":"2026-11-30 05:30","last":"2026-11-23 05:30","result":0}'
    assert S.run_info("x")["result"] == "success"


def test_engine_command_in_the_packaged_app(monkeypatch, tmp_path):
    from cfb.ui import engine_job

    app = tmp_path / "CFB Predictor.exe"
    app.write_text("")
    monkeypatch.setattr(sys, "executable", str(app))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert S.engine_command("weekly")[:2] == (str(app), "engine all --trigger scheduled")
    assert engine_job.command_line("all")[:3] == [str(app), "engine", "all"]
    (tmp_path / "CFB Engine.exe").write_text("")
    assert S.engine_command("daily") == (str(tmp_path / "CFB Engine.exe"), "engine light --trigger scheduled",
                                         str(tmp_path))
    assert engine_job.command_line("check")[0] == str(tmp_path / "CFB Engine.exe")


def test_engine_job_follows_a_run(env, monkeypatch):
    from cfb.ui import engine_job

    job = engine_job.EngineJob()
    assert not job.running() and not job.take_fresh()
    E._write_json(E.status_path(), {"state": "ok", "finished_utc": "2026-11-20T06:00:00Z"})
    assert job.take_fresh() and not job.take_fresh()
    log = env["engine"] / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(30)), encoding="utf-8")
    assert engine_job.tail(log, 3) == ["line 27", "line 28", "line 29"]
    assert engine_job.tail(None) == []


# ---- app pages ------------------------------------------------------------------------------------

st_testing = pytest.importorskip("streamlit.testing.v1")


def _page(name):
    def script(page_name):
        from cfb.ui import pages
        getattr(pages, page_name)()

    at = st_testing.AppTest.from_function(script, args=(name,), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_data_page_shows_the_last_run(env):
    E._write_json(E.status_path(), {"state": "warning", "command": "all", "trigger": "scheduled",
                                    "started_utc": "2026-11-23T10:30:00Z", "finished_utc": "2026-11-23T10:40:00Z",
                                    "report_label": "2026 week 13", "report_games": 60, "api_calls": 220,
                                    "warnings": ["backup to R2 failed: x"], "cfbd_quota": {"remaining": 1, "monthly_limit": 2},
                                    "check": {"ok": True, "games": 1, "values": 2, "max_diff": {"p": 0, "pred_margin": 0}}})
    at = _page("page_data")
    text = " ".join(str(m.value) for m in at.warning) + " ".join(str(m.value) for m in at.info)
    assert "2026 week 13" in text and "backup to R2 failed" in text
    assert any("Bootstrap from the cloud" == b.label for b in at.button)


def test_settings_page_has_backup_and_schedule(env):
    at = _page("page_settings")
    assert any("Back up to R2" in t.label for t in at.toggle)
    assert [s.value for s in at.subheader][:3] == ["Keys", "Schedule", "Backup"]
