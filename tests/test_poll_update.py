"""cfb.poll_update: pull polls after the weekly run, refresh the report's rankings, publish."""
from __future__ import annotations

import json
import shutil
import tarfile
from datetime import datetime, timezone

import pandas as pd

from cfb import poll_update as pu
from cfb import warehouse_sync as ws
from cfb.core.synthetic import SEASONS
from cfb.report.config import ReportConfig

LIVE = max(SEASONS)
NOW = datetime(2026, 11, 25, 4, 0, tzinfo=timezone.utc)
CFG = ReportConfig(primary="baseline_elo", baselines=("baseline_close",), market_models=("baseline_close",),
                   build_features=False)


class FakeClient:
    def __init__(self, payload):
        self.payload, self.requests = payload, []

    def get(self, path, **params):
        self.requests.append((path, params))
        return self.payload


def polls_payload(team_ids, week, polls=("Playoff Committee Rankings",)):
    return [{"season": LIVE, "seasonType": "regular", "week": week, "polls": [
        {"poll": p, "isFinal": False, "ranks": [
            {"rank": i + 1, "teamId": int(t), "school": f"T{t}", "conference": "X", "firstPlaceVotes": 0,
             "points": 100 - i} for i, t in enumerate(team_ids)]} for p in polls]}]


def weekly_run(synthetic, tmp_path, monkeypatch):
    """Monday: run_week builds the week 4 report and publishes it to a folder bucket and to pub/."""
    from cfb.run_week import run

    root = tmp_path / "monday"
    shutil.copytree(synthetic.root, root)
    monkeypatch.setenv("CFB_WAREHOUSE_LOCAL", str(tmp_path / "bucket"))
    res = run(root, CFG, season=LIVE, week=4, do_refresh=False, publish_dir=tmp_path / "pub",
              publish_warehouse_to_r2=None)
    assert res.warehouse_error is None
    return res


def test_poll_update_refreshes_rankings_and_publishes(synthetic, tmp_path, monkeypatch):
    monday = weekly_run(synthetic, tmp_path, monkeypatch)
    key = monday.slate.key
    bucket = ws.LocalStore(tmp_path / "bucket")
    before = json.loads((tmp_path / "bucket" / "manifest.json").read_bytes())
    old_bundle = json.loads((tmp_path / "pub" / key / "bundle.json").read_text())
    assert not (old_bundle["rankings"] or {}).get("polls")  # synthetic data has no polls

    teams = pd.read_parquet(synthetic.root / "tables" / "teams_fbs" / f"{LIVE}.parquet")["id"].head(25).tolist()
    client = FakeClient(polls_payload(teams, week=4))
    res = pu.run(work=tmp_path / "work", season=LIVE, client=client, store=bucket, config=CFG,
                 publish_dir=tmp_path / "pub", archive_dir=tmp_path / "polls", now=NOW)

    assert client.requests == [("/rankings", {"year": LIVE})]
    assert res.key == key and res.new_polls == ["CFP week 4"] and res.report_updated
    assert [p["key"] for p in res.rankings["polls"]] == ["cfp"]

    # R2: only the rankings table and the report files changed.
    after = json.loads((tmp_path / "bucket" / "manifest.json").read_bytes())
    assert res.warehouse.changed and after["version"] != before["version"] and after["key"] == key
    changed = {k for k, f in after["files"].items() if before["files"].get(k) != f}
    assert changed == {f"tables/rankings/{LIVE}.parquet", f"reports/{key}/bundle.json",
                       f"reports/{key}/report.md", f"reports/{key}/dashboard.html"}

    # Project files: the published report and latest/ patched in place, metrics kept.
    for folder in (tmp_path / "pub" / key, tmp_path / "pub" / "latest"):
        b = json.loads((folder / "bundle.json").read_text())
        assert b["rankings"]["polls"][0]["week"] == 4
        assert b["games"] == old_bundle["games"] and b["season"] == old_bundle["season"]
        assert b["meta"]["notes"][-1].startswith("Polls refreshed 2026-11-25 04:00 UTC (CFP week 4)")
        assert "CFP (week 4)" in (folder / "report.md").read_text()
    assert (tmp_path / "pub" / key / "metrics").is_dir()

    # Polls archive: dated tar with the raw and table files, LATEST pointer.
    name = (tmp_path / "polls" / "LATEST").read_text().strip()
    with tarfile.open(tmp_path / "polls" / name) as tar:
        assert sorted(tar.getnames()) == [f"data/raw/rankings/{LIVE}.json.gz", f"data/tables/rankings/{LIVE}.parquet"]
    assert "CFP week 4" in pu.summary(res) and "CFP top 8 (model rank): 1 " in pu.summary(res)

    # Same pull again: no new poll week, nothing written or published.
    res2 = pu.run(work=tmp_path / "work", season=LIVE, client=client, store=bucket, config=CFG,
                  publish_dir=tmp_path / "pub", archive_dir=tmp_path / "polls", now=NOW)
    assert res2.new_polls == [] and res2.warehouse is None and res2.archive is None
    assert json.loads((tmp_path / "bucket" / "manifest.json").read_bytes())["version"] == after["version"]
    assert "No new poll" in pu.summary(res2)


def test_poll_labels():
    assert pu.poll_label("regular", 13, "Playoff Committee Rankings") == "CFP week 13"
    assert pu.poll_label("postseason", 1, "AP Top 25") == "AP final"
