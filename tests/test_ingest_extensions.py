"""Tests for the datasets and run rules added to cfb.ingest. No network."""
from datetime import datetime, timezone

import pandas as pd

from cfb.ingest import (
    BY_NAME,
    box_rows,
    coverage,
    current_season,
    main,
    poll_rows,
    run,
    sync_asof,
    tasks,
    to_markdown,
    years_for,
)
from cfb.store import Store

NOV_17 = datetime(2026, 11, 17, 6, 0, tzinfo=timezone.utc)


class FakeClient:
    """Answers every GET with `payload(path, params)` and records the calls."""

    def __init__(self, payload=lambda path, params: [{"team": "A", "rating": 1.0}]):
        self.payload, self.requests, self.calls = payload, [], 0

    def get(self, path, **params):
        self.calls += 1
        self.requests.append((path, params))
        return self.payload(path, params)


def calendar(year):
    """Three regular weeks for `year`: one ended, one in progress, one not started on NOV_17."""
    return [
        {"seasonType": "regular", "week": 1, "startDate": f"{year}-11-02T08:00:00.000Z", "endDate": f"{year}-11-09T07:59:00.000Z"},
        {"seasonType": "regular", "week": 2, "startDate": f"{year}-11-16T08:00:00.000Z", "endDate": f"{year}-11-23T07:59:00.000Z"},
        {"seasonType": "regular", "week": 3, "startDate": f"{year}-11-23T08:00:00.000Z", "endDate": f"{year}-11-30T07:59:00.000Z"},
        {"seasonType": "postseason", "week": 1, "startDate": f"{year}-12-12T08:00:00.000Z", "endDate": f"{year + 1}-01-28T07:59:00.000Z"},
    ]


def test_new_datasets_are_registered():
    new = {
        "weather", "media", "rankings", "ratings_elo_weekly", "ratings_elo_preseason",
        "recruiting_teams", "transfer_portal", "games_players", "coach_tenures",
        "stats_game_havoc", "punting_games", "ratings_core",
    }
    assert new <= set(BY_NAME)
    assert {n for n, d in BY_NAME.items() if d.asof} == {"ratings_sp", "ratings_fpi", "ratings_srs", "ratings_core"}


def test_current_season_turns_over_in_august():
    assert current_season(datetime(2026, 11, 17, tzinfo=timezone.utc)) == 2026
    assert current_season(datetime(2027, 1, 20, tzinfo=timezone.utc)) == 2026
    assert current_season(datetime(2027, 8, 1, tzinfo=timezone.utc)) == 2027


def test_years_for_adds_lag_and_lead():
    assert years_for(BY_NAME["recruiting_teams"], [2014, 2015]) == [2010, 2011, 2012, 2013, 2014, 2015]
    assert years_for(BY_NAME["transfer_portal"], [2025, 2026]) == [2025, 2026, 2027]
    assert years_for(BY_NAME["weather"], [2025, 2026]) == [2025, 2026]


def test_regular_week_tasks_omit_season_type_param():
    out = list(tasks(BY_NAME["ratings_elo_weekly"], 2025, {"regular": [1, 2], "postseason": [1]}))
    assert [k for k, _, _ in out] == ["2025_regular_01", "2025_regular_02"]
    assert out[0][1] == {"year": 2025, "week": 1}
    assert out[0][2] == {"season": 2025, "season_type": "regular", "week": 1}


def test_poll_rows_one_row_per_ranked_team():
    payload = [
        {
            "season": 2024, "seasonType": "regular", "week": 2,
            "polls": [
                {"poll": "AP Top 25", "isFinal": None, "ranks": [
                    {"rank": 1, "teamId": 61, "school": "Georgia", "conference": "SEC", "firstPlaceVotes": 57, "points": 1543},
                    {"rank": 2, "teamId": 194, "school": "Ohio State", "conference": "Big Ten", "firstPlaceVotes": 5, "points": 1477},
                ]},
                {"poll": "Coaches Poll", "ranks": []},
            ],
        }
    ]
    rows = poll_rows(payload)
    assert len(rows) == 2
    assert rows[1] == {
        "season": 2024, "seasonType": "regular", "week": 2, "poll": "AP Top 25", "isFinal": None,
        "rank": 2, "teamId": 194, "school": "Ohio State", "conference": "Big Ten", "firstPlaceVotes": 5, "points": 1477,
    }


def test_box_rows_one_row_per_athlete_stat():
    payload = [
        {"id": 7, "teams": [
            {"team": "A", "conference": "X", "homeAway": "home", "points": 21, "categories": [
                {"name": "passing", "types": [
                    {"name": "C/ATT", "athletes": [{"id": "11", "name": "Q One", "stat": "15/22"}]},
                    {"name": "YDS", "athletes": [{"id": 11, "name": "Q One", "stat": 201}]},
                ]},
            ]},
            {"team": "B", "conference": None, "homeAway": "away", "points": 3, "categories": []},
        ]}
    ]
    rows = box_rows(payload)
    assert rows == [
        {"gameId": 7, "team": "A", "conference": "X", "homeAway": "home", "category": "passing",
         "statType": "C/ATT", "athleteId": "11", "athleteName": "Q One", "stat": "15/22"},
        {"gameId": 7, "team": "A", "conference": "X", "homeAway": "home", "category": "passing",
         "statType": "YDS", "athleteId": "11", "athleteName": "Q One", "stat": "201"},
    ]


def test_reshaped_table_keeps_raw_payload(tmp_path):
    payload = [{"season": 2025, "seasonType": "regular", "week": 1, "polls": [
        {"poll": "AP Top 25", "ranks": [{"rank": 1, "teamId": 1, "school": "A"}, {"rank": 2, "teamId": 2, "school": "B"}]}
    ]}]
    s = Store(tmp_path)
    assert run(FakeClient(lambda p, q: payload), s, [2025], ["rankings"], now=NOV_17) == []
    assert s.read_raw("rankings", "2025") == payload
    table = s.load("rankings")
    assert list(table["school"]) == ["A", "B"] and list(table["week"]) == [1, 1]


def test_weeks_not_started_are_skipped_and_elo_waits_for_week_end(tmp_path):
    s = Store(tmp_path)
    s.write("calendar", "2026", calendar(2026))
    c = FakeClient()
    assert run(c, s, [2026], ["games_players", "ratings_elo_weekly"], now=NOV_17) == []
    pulled = {(path, params.get("seasonType"), params["week"]) for path, params in c.requests}
    assert pulled == {
        ("/games/players", "regular", 1),
        ("/games/players", "regular", 2),  # in progress: pulled
        ("/ratings/elo", None, 1),  # only the ended week
    }
    assert not s.has("games_players", "2026_regular_03") and not s.has("ratings_elo_weekly", "2026_regular_02")


def test_force_does_not_pull_unended_elo_weeks(tmp_path):
    s = Store(tmp_path)
    s.write("calendar", "2026", calendar(2026))
    c = FakeClient(lambda path, params: calendar(2026) if path == "/calendar" else [{"team": "A", "elo": 1500}])
    assert run(c, s, [2026], ["ratings_elo_weekly"], force=True, now=NOV_17) == []
    assert [params.get("week") for path, params in c.requests if path == "/ratings/elo"] == [1]


def test_elo_weeks_without_calendar_dates_pull_only_for_past_seasons(tmp_path):
    s = Store(tmp_path)
    for year in (2025, 2026):
        s.write("calendar", str(year), [{"seasonType": "regular", "week": 1}])
    c = FakeClient()
    run(c, s, [2025, 2026], ["ratings_elo_weekly"], now=NOV_17)
    assert [params["year"] for _, params in c.requests] == [2025]


def test_current_season_ratings_get_a_dated_copy_on_every_run(tmp_path):
    s = Store(tmp_path)
    c = FakeClient()
    assert run(c, s, [2025, 2026], ["ratings_sp", "ratings_core", "records"], now=NOV_17) == []
    assert c.calls == 6
    assert s.has("ratings_sp_asof", "2026_20261117") and s.has("ratings_core_asof", "2026_20261117")
    assert not s.has("ratings_sp_asof", "2025_20261117")  # finished season: no dated copy
    assert not (tmp_path / "raw" / "records_asof").exists()
    copy = s.load("ratings_sp_asof")
    assert list(copy["asof_date"]) == ["2026-11-17"] and list(copy["asof_utc"]) == ["2026-11-17T06:00:00Z"]
    assert list(copy["season"]) == [2026]

    # Next run, a week later and without --force: only the current-season ratings are pulled again.
    c2 = FakeClient(lambda path, params: [{"team": "A", "rating": 2.0}])
    later = datetime(2026, 11, 24, 6, 0, tzinfo=timezone.utc)
    run(c2, s, [2025, 2026], ["ratings_sp", "ratings_core", "records"], now=later)
    assert sorted((path, params["year"]) for path, params in c2.requests) == [("/ratings/core", 2026), ("/ratings/sp", 2026)]
    copies = s.load("ratings_sp_asof").sort_values("asof_date")
    assert list(copies["asof_date"]) == ["2026-11-17", "2026-11-24"] and list(copies["rating"]) == [1.0, 2.0]
    assert s.read_raw("ratings_sp", "2026") == [{"team": "A", "rating": 2.0}]


def test_empty_rating_response_makes_no_dated_copy(tmp_path):
    s = Store(tmp_path)
    run(FakeClient(lambda path, params: []), s, [2026], ["ratings_fpi"], now=NOV_17)
    assert s.has("ratings_fpi", "2026") and not s.has("ratings_fpi_asof", "2026_20261117")


def test_asof_archive_round_trip(tmp_path):
    archive = tmp_path / "archive"
    first = Store(tmp_path / "run1")
    run(FakeClient(), first, [2026], ["ratings_sp"], now=NOV_17, asof_archive=archive)
    assert (archive / "raw" / "ratings_sp_asof" / "2026_20261117.json.gz").exists()
    assert (archive / "tables" / "ratings_sp_asof" / "2026_20261117.parquet").exists()

    # A fresh data dir gets the earlier copy back, adds its own, and archives only the new one.
    second = Store(tmp_path / "run2")
    later = datetime(2026, 11, 24, 6, 0, tzinfo=timezone.utc)
    run(FakeClient(), second, [2026], ["ratings_sp"], now=later, asof_archive=archive)
    assert sorted(second.load("ratings_sp_asof")["asof_date"]) == ["2026-11-17", "2026-11-24"]
    assert len(list((archive / "raw" / "ratings_sp_asof").iterdir())) == 2
    assert sync_asof(second.root, archive) == 0  # nothing missing


def test_same_day_rerun_replaces_the_archived_copy(tmp_path):
    archive = tmp_path / "archive"
    s = Store(tmp_path / "data")
    run(FakeClient(), s, [2026], ["ratings_sp"], now=NOV_17, asof_archive=archive)
    run(FakeClient(lambda p, q: [{"team": "A", "rating": 9.0}]), s, [2026], ["ratings_sp"], now=NOV_17, asof_archive=archive)
    restored = Store(tmp_path / "restored")
    assert sync_asof(archive, restored.root) == 2
    assert list(restored.load("ratings_sp_asof")["rating"]) == [9.0]


def test_coach_tenures_one_request_per_fbs_team(tmp_path):
    def payload(path, params):
        if path == "/teams/fbs":
            teams = [{"id": 2, "school": "Texas A&M"}, {"id": 1, "school": "Akron"}]
            return teams + ([{"id": 3, "school": "Delaware"}] if params["year"] == 2026 else [])
        return [{"id": 10, "coach": {"id": 5}, "team": {"id": 1, "school": params["team"]}, "isInterim": False}]

    s = Store(tmp_path)
    c = FakeClient(payload)
    assert run(c, s, [2025, 2026], ["coach_tenures"], now=NOV_17) == []
    tenure_calls = [params for path, params in c.requests if path == "/coaches/tenures"]
    assert tenure_calls == [{"team": "Akron"}, {"team": "Delaware"}, {"team": "Texas A&M"}]
    assert s.has("coach_tenures", "team_3") and s.has("teams_fbs", "2025")
    assert set(s.load("coach_tenures")["team.school"]) == {"Akron", "Delaware", "Texas A&M"}

    c2 = FakeClient(payload)
    run(c2, s, [2025, 2026], ["coach_tenures"], now=NOV_17)
    assert c2.calls == 0  # all cached


def test_coverage_counts_rows_by_season(tmp_path):
    s = Store(tmp_path)
    s.write("games", "2025_regular", [{"id": 1}, {"id": 2}])
    s.write("games", "2025_postseason", [{"id": 3}])
    s.write("games", "2026_regular", [{"id": 4}])
    s.write("coach_tenures", "team_1", [{"id": 1}])
    table = coverage(s)
    assert table.loc["games", "2025"] == 3 and table.loc["games", "2026"] == 1
    assert table.loc["coach_tenures", "all"] == 1 and table.loc["coach_tenures", "2025"] == 0
    assert isinstance(table, pd.DataFrame)
    text = to_markdown(table)
    assert text.splitlines()[0] == "| dataset | 2025 | 2026 | all |"
    assert "| games | 3 | 1 | 0 |" in text


def test_cli_coverage_prints_table(tmp_path, capsys):
    Store(tmp_path).write("games", "2025_regular", [{"id": 1}])
    assert main(["--coverage", "--data-dir", str(tmp_path)]) == 0
    assert "| games | 1 |" in capsys.readouterr().out
