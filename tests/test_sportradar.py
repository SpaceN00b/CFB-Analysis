"""Tests for cfb.sportradar. No network."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from cfb.sportradar import (
    SportradarClient,
    SportradarError,
    StopRun,
    fbs_team_ids,
    hierarchy_teams,
    play_rows,
    play_stat_rows,
    run,
    schedule_games,
    wanted,
)
from cfb.store import Store

NOV_18 = datetime(2026, 11, 18, 16, 0, tzinfo=timezone.utc)
FBS_A, FBS_B, FCS_C = "t-a", "t-b", "t-c"

HIERARCHY = {
    "league": {"alias": "NCAAFB"},
    "divisions": [
        {"alias": "FBS", "conferences": [
            {"id": "c1", "alias": "B1", "teams": [{"id": FBS_A, "name": "A"}]},
            {"id": "c2", "alias": "B2", "subdivisions": [{"alias": "EAST", "teams": [{"id": FBS_B, "name": "B"}]}]},
        ]},
        {"alias": "FCS", "conferences": [{"id": "c3", "alias": "F1", "teams": [{"id": FCS_C, "name": "C"}]}]},
    ],
}


def game(gid, home, away, status="closed", cov="full"):
    return {"id": gid, "status": status, "coverage": cov, "home": {"id": home}, "away": {"id": away}}


def schedule(year, stype, games):
    return {"year": year, "type": stype, "weeks": [{"sequence": 1, "title": "1", "games": games}]}


PBP = {
    "id": "g1",
    "periods": [
        {"period_type": "quarter", "number": 1, "pbp": [
            {"type": "drive", "id": "d1", "sequence": 1, "start_reason": "Kickoff", "end_reason": "Punt",
             "offensive_team": {"id": FBS_A}, "defensive_team": {"id": FBS_B}, "events": [
                {"type": "play", "id": "p1", "play_type": "pass", "clock": "14:55",
                 "start_situation": {"down": 1, "yfd": 10, "location": {"yardline": 25}},
                 "statistics": [
                     {"stat_type": "pass", "attempt": 1, "complete": 0,
                      "player": {"id": "qb1", "name": "Q", "position": "QB", "jersey": "01"}, "team": {"id": FBS_A, "alias": "A"}},
                     {"stat_type": "receive", "target": 1,
                      "player": {"id": "wr1", "name": "W", "position": "WR", "jersey": "02"}, "team": {"id": FBS_A, "alias": "A"}},
                 ],
                 "details": [{"category": "pass", "players": [{"id": "qb1", "role": "pass"}]}]},
                {"type": "event", "id": "e1", "event_type": "tv_timeout"},
            ]},
            {"type": "event", "id": "e2", "event_type": "period_end"},
        ]},
    ],
}


class Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.urls, self.headers = list(responses), [], []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        self.headers.append(headers)
        return self.responses.pop(0)


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def client(responses, **kw):
    clock = Clock()
    sess = FakeSession(responses)
    c = SportradarClient(api_key="", session=sess, sleep=clock.sleep, clock=clock, **kw)
    return c, sess, clock


def test_client_url_no_key_header_and_spacing():
    c, sess, clock = client([Resp(200, {"a": 1}), Resp(200, {"b": 2})], interval=1.2)
    assert c.get("league/hierarchy.json") == {"a": 1}
    assert c.get("/games/2025/REG/schedule.json") == {"b": 2}
    assert sess.urls == [
        "https://api.sportradar.com/ncaafb/trial/v7/en/league/hierarchy.json",
        "https://api.sportradar.com/ncaafb/trial/v7/en/games/2025/REG/schedule.json",
    ]
    assert all("x-api-key" not in h for h in sess.headers)
    assert clock.sleeps == [pytest.approx(1.2)]
    assert c.calls == 2


def test_client_sends_key_when_set():
    sess = FakeSession([Resp(200, {})])
    SportradarClient(api_key="k", session=sess, sleep=lambda s: None).get("x.json")
    assert sess.headers[0]["x-api-key"] == "k"


def test_client_retries_throttle_then_succeeds():
    c, sess, clock = client([Resp(429, text='{"message":"Too Many Requests"}'), Resp(200, {"ok": 1})])
    assert c.get("x.json") == {"ok": 1}
    assert c.calls == 2 and c.throttled == 1


def test_client_stops_on_quota_auth_and_budget():
    c, _, _ = client([Resp(429, text='{"message":"Limit Exceeded"}')])
    with pytest.raises(StopRun, match="quota"):
        c.get("x.json")
    c, _, _ = client([Resp(403, text='{"message":"Authentication Error"}')])
    with pytest.raises(StopRun, match="key rejected"):
        c.get("x.json")
    c, _, _ = client([Resp(200, {}), Resp(200, {})], max_calls=1)
    c.get("x.json")
    with pytest.raises(StopRun, match="budget"):
        c.get("x.json")


def test_schedule_404_is_skipped(tmp_path):
    r = routes()
    del r["games/2026/REG/schedule.json"]
    res = run(FakeClient(r), Store(tmp_path), [2026], estimate=True, now=NOV_18, log=lambda *a: None)
    assert res.pending == {} and not res.failures


def test_client_404_is_not_retried():
    c, _, _ = client([Resp(404, text="")])
    with pytest.raises(SportradarError) as err:
        c.get("games/2012/REG/schedule.json")
    assert err.value.status == 404 and c.calls == 1


def test_reshapers():
    teams = hierarchy_teams(HIERARCHY)
    assert {(t["id"], t["division"], t["subdivision"]) for t in teams} == {
        (FBS_A, "FBS", None), (FBS_B, "FBS", "EAST"), (FCS_C, "FCS", None)}
    assert fbs_team_ids(HIERARCHY) == {FBS_A, FBS_B}

    rows = schedule_games(schedule(2025, "REG", [game("g1", FBS_A, FBS_B)]))
    assert rows[0]["season"] == 2025 and rows[0]["season_type"] == "REG" and rows[0]["week"] == 1

    plays = play_rows(PBP)
    assert len(plays) == 1  # events that are not plays are left out
    assert plays[0]["drive_id"] == "d1" and plays[0]["drive_offense_id"] == FBS_A and "statistics" not in plays[0]
    stats = play_stat_rows(PBP)
    assert [(s["player_id"], s["stat_type"], s.get("target")) for s in stats] == [("qb1", "pass", None), ("wr1", "receive", 1)]


def test_wanted_filters_status_coverage_and_fbs():
    fbs = {FBS_A, FBS_B}
    assert wanted(game("1", FBS_A, FCS_C), fbs)
    assert not wanted(game("2", FCS_C, "x"), fbs)
    assert not wanted(game("3", FBS_A, FBS_B, status="scheduled"), fbs)
    assert not wanted(game("4", FBS_A, FBS_B, cov="extended_boxscore"), fbs)


class FakeClient:
    def __init__(self, routes):
        self.routes, self.paths, self.calls = routes, [], 0

    def get(self, path):
        self.calls += 1
        self.paths.append(path)
        hit = self.routes.get(path)
        if hit is None:
            raise SportradarError(f"GET {path} failed. HTTP 404", 404)
        if isinstance(hit, Exception):
            raise hit
        return hit


def routes():
    return {
        "league/hierarchy.json": HIERARCHY,
        "games/2025/REG/schedule.json": schedule(2025, "REG", [
            game("g1", FBS_A, FBS_B), game("g2", FBS_A, FCS_C, cov="boxscore"), game("g3", FCS_C, "x"),
            game("g4", FBS_B, FBS_A)]),
        "games/2025/PST/schedule.json": schedule(2025, "PST", [game("x", FBS_B, FBS_A)]),  # never asked
        "games/2026/REG/schedule.json": schedule(2026, "REG", [
            game("g5", FBS_A, FBS_B), game("g6", FBS_B, FBS_A, status="scheduled")]),
        "games/g1/pbp.json": PBP,
        "games/g4/pbp.json": {**PBP, "id": "g4"},
        "games/g5/pbp.json": {**PBP, "id": "g5"},
    }


def test_estimate_pulls_schedules_only(tmp_path):
    fc, store = FakeClient(routes()), Store(tmp_path)
    res = run(fc, store, [2025, 2026], estimate=True, now=NOV_18, log=lambda *a: None)
    assert not any(p.endswith("pbp.json") for p in fc.paths)
    assert "games/2025/PST/schedule.json" not in fc.paths
    assert res.pending == {"2025_REG": 2, "2026_REG": 1}
    assert res.pending_total == 3 and not res.failures and res.stopped is None


def test_run_pulls_caches_and_refreshes_current_season(tmp_path):
    store = Store(tmp_path)
    fc = FakeClient(routes())
    res = run(fc, store, [2025, 2026], now=NOV_18, log=lambda *a: None)
    assert res.pulled == 3 and res.pending_total == 0 and not res.failures
    plays = store.load("sr_pbp")
    assert set(plays["game_id"]) == {"g1", "g4", "g5"} and set(plays["season"]) == {2025, 2026}
    assert len(store.load("sr_play_stats")) == 6
    assert len(store.load("sr_schedule")) == 6

    fc2 = FakeClient(routes())
    run(fc2, store, [2025, 2026], now=NOV_18, log=lambda *a: None)
    assert fc2.paths == ["games/2026/REG/schedule.json"]  # cached: only the current season's schedule again


def test_run_stops_on_budget_and_resumes(tmp_path):
    store = Store(tmp_path)
    r = routes()
    calls = {"n": 0}

    class Budget(FakeClient):
        def get(self, path):
            calls["n"] += 1
            if calls["n"] > 3:  # hierarchy and two schedules, no play-by-play
                raise StopRun("call budget of 3 reached")
            return super().get(path)

    res = run(Budget(r), store, [2025, 2026], now=NOV_18, log=lambda *a: None)
    assert res.stopped and res.pulled == 0 and res.pending_total == 3
    res2 = run(FakeClient(r), store, [2025, 2026], now=NOV_18, log=lambda *a: None)
    assert res2.pulled == 3 and res2.pending_total == 0


def test_failed_game_is_reported_and_retried_later(tmp_path):
    store = Store(tmp_path)
    r = routes()
    r["games/g4/pbp.json"] = SportradarError("GET games/g4/pbp.json failed. HTTP 500", 500)
    res = run(FakeClient(r), store, [2025], now=NOV_18, log=lambda *a: None)
    assert res.pulled == 1 and len(res.failures) == 1 and res.pending == {"2025_REG": 1}
    assert not store.has("sr_pbp", "2025_REG_g4")
