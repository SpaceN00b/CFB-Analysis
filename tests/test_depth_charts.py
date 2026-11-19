"""cfb.depth_charts: CSV parsing, team and player matching, dated partitions, access policy."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from cfb import depth_charts as dc
from cfb.core.policies import DATED, POLICIES
from cfb.store import Store

CSV = """Team,Unit,Scheme,Position,Depth,Number,Player
Georgia Bulldogs,Offense,Pro Spread,QB,First Team,14,"Stockton, Gunner"
Georgia Bulldogs,Offense,Pro Spread,QB,Second Team,12,"Puglisi, Ryan"
Georgia Bulldogs,Offense,Pro Spread,LT,First Team,71,"Greene III, Earnest"
Georgia Bulldogs,Defense,3-4,JACK,First Team,0,"Harris Jr., Gabe"
Georgia Bulldogs,Defense,3-4,STAR,First Team,3,"Doe, John"
Georgia Bulldogs,Special Teams,,PK,First Team,98,"Woodring, Peyton"
Georgia Bulldogs,Bench,,Bench,Bench,47,"Taylor, Will"
"""

OLE_MISS = """Team,Unit,Scheme,Position,Depth,Number,Player
Mississippi Rebels,Offense,Up-Tempo Spread,QB,First Team,2,"Chambliss, Trinidad"
"""


@pytest.fixture
def data_dir(tmp_path):
    store = Store(tmp_path / "data")
    teams = [
        {"id": 61, "school": "Georgia", "mascot": "Bulldogs", "alternateNames": json.dumps(["UGA"]), "season": 2026},
        {"id": 145, "school": "Ole Miss", "mascot": "Rebels", "alternateNames": json.dumps(["MISS"]), "season": 2026},
    ]
    roster = [
        # jersey + last name
        {"id": 1, "team": "Georgia", "firstName": "Gunner", "lastName": "Stockton", "jersey": 14},
        # last name with suffix in CFBD, same jersey
        {"id": 2, "team": "Georgia", "firstName": "Earnest", "lastName": "Greene III", "jersey": 71},
        # different jersey: full name
        {"id": 3, "team": "Georgia", "firstName": "Gabe", "lastName": "Harris", "jersey": 9, "position": "LB"},
        # unique last name, first name spelled differently
        {"id": 4, "team": "Georgia", "firstName": "Ry", "lastName": "Puglisi", "jersey": 99},
        # two Taylors, neither with #47: no match
        {"id": 5, "team": "Georgia", "firstName": "Talyn", "lastName": "Taylor", "jersey": 1},
        {"id": 6, "team": "Georgia", "firstName": "William", "lastName": "Taylor", "jersey": 48},
        {"id": 7, "team": "Ole Miss", "firstName": "Trinidad", "lastName": "Chambliss", "jersey": 2},
    ]
    for name, rows in (("teams_fbs", teams), ("roster", roster)):
        path = store.table_path(name, "2026")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    return store.root


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_split_name():
    assert dc.split_name("Greene III, Earnest") == ("Earnest", "Greene", "III")
    assert dc.split_name("Flynn Jr., Waltclaire") == ("Waltclaire", "Flynn", "Jr.")
    assert dc.split_name("Williams, Tyler J.") == ("Tyler J.", "Williams", "")
    assert dc.split_name("White-Helton, Sacovie") == ("Sacovie", "White-Helton", "")
    assert dc.split_name("Gunner Stockton") == ("Gunner", "Stockton", "")


def test_import_parses_matches_and_partitions(tmp_path, data_dir):
    src = write(tmp_path, "Georgia_Bulldogs_depth_chart.csv", CSV)
    rows = dc.import_files([src], data_dir=data_dir, asof="2026-11-19T04:01:16Z", log=lambda *_: None)
    assert len(rows) == 7 and set(rows["team"]) == {"Georgia"}
    by = rows.set_index("player_raw")
    assert by.loc["Stockton, Gunner", ["athlete_id", "match"]].tolist() == ["1", "jersey_last"]
    assert by.loc["Greene III, Earnest", ["athlete_id", "match", "player"]].tolist() == ["2", "jersey_last", "Earnest Greene III"]
    assert by.loc["Harris Jr., Gabe", ["athlete_id", "match"]].tolist() == ["3", "name"]
    assert by.loc["Puglisi, Ryan", ["athlete_id", "match"]].tolist() == ["4", "last_initial"]
    assert by.loc["Taylor, Will", "match"] == "none" and by.loc["Taylor, Will", "athlete_id"] is None
    assert by.loc["Harris Jr., Gabe", "pos_group"] == "EDGE" and by.loc["Doe, John", "pos_group"] == "DB"
    assert by.loc["Woodring, Peyton", "unit"] == "special_teams"
    assert pd.isna(by.loc["Taylor, Will", "depth"]) and by.loc["Puglisi, Ryan", "depth"] == 2
    # starters: depth 1 on offense or defense only (the kicker is not counted)
    assert by["starter"].sum() == 4 and not by.loc["Woodring, Peyton", "starter"]
    assert set(rows["asof_utc"]) == {"2026-11-19T04:01:16Z"} and set(rows["season"]) == {2026}

    store = Store(data_dir)
    assert store.table_path("depth_charts", "2026-11-19_georgia").exists()
    assert (data_dir / "raw" / "depth_charts" / "2026-11-19" / src.name).read_text(encoding="utf-8") == CSV


def test_reimport_replaces_same_date_and_adds_new_dates(tmp_path, data_dir):
    src = write(tmp_path, "Georgia_Bulldogs_depth_chart.csv", CSV)
    quiet = dict(data_dir=data_dir, log=lambda *_: None)
    dc.import_files([src], asof="2026-11-19T04:01:16Z", **quiet)
    dc.import_files([src], asof="2026-11-19T05:00:00Z", **quiet)
    dc.import_files([src], asof="2026-11-26T05:00:00Z", **quiet)
    dc.import_files([write(tmp_path, "om.csv", OLE_MISS)], asof="2026-11-26T05:00:00Z", **quiet)
    table = Store(data_dir).load("depth_charts")
    counts = table.groupby(["asof_date", "team"]).size().to_dict()
    assert counts == {("2026-11-19", "Georgia"): 7, ("2026-11-26", "Georgia"): 7, ("2026-11-26", "Ole Miss"): 1}
    assert set(table[table["asof_date"] == "2026-11-19"]["asof_utc"]) == {"2026-11-19T05:00:00Z"}
    assert dc.summary(data_dir)["rows"].sum() == 15


def test_unknown_team_and_columns_fail_loudly(tmp_path, data_dir):
    bad_team = write(tmp_path, "x.csv", CSV.replace("Georgia Bulldogs", "Nowhere Owls"))
    with pytest.raises(ValueError, match="Nowhere Owls"):
        dc.import_files([bad_team], data_dir=data_dir, asof="2026-11-19", log=lambda *_: None)
    good = write(tmp_path, "Georgia_Bulldogs_depth_chart.csv", CSV)
    with pytest.raises(ValueError, match="Nowhere Owls"):  # nothing is written when one file fails
        dc.import_files([good, bad_team], data_dir=data_dir, asof="2026-11-19", log=lambda *_: None)
    assert not (data_dir / "tables" / "depth_charts").exists()
    bad_cols = write(tmp_path, "y.csv", "Team,Pos,Player\nGeorgia Bulldogs,QB,x\n")
    with pytest.raises(ValueError, match="missing columns"):
        dc.import_files([bad_cols], data_dir=data_dir, asof="2026-11-19", log=lambda *_: None)


def test_unknown_position_is_kept_as_other(tmp_path, data_dir):
    src = write(tmp_path, "z.csv", CSV.replace(",STAR,", ",SPECTRE,").replace(",JACK,", ",GHOST,"))
    notes = []
    rows = dc.import_files([src], data_dir=data_dir, asof="2026-11-19", log=notes.append)
    by = rows.set_index("player_raw")
    assert by.loc["Doe, John", "pos_group"] == "OTHER"  # not on the roster
    assert by.loc["Harris Jr., Gabe", ["pos_group", "roster_position"]].tolist() == ["LB", "LB"]
    assert sum("SPECTRE" in n for n in notes) == 1 and sum("GHOST" in n for n in notes) == 1


def test_full_name_match_across_the_first_last_split():
    frame = pd.DataFrame([{"team": "TCU", "first_name": "John", "last_name": "Hoyet Chance", "jersey": 48}])
    roster = pd.DataFrame([{"id": 9, "team": "TCU", "firstName": "John Hoyet", "lastName": "Chance",
                            "jersey": 47, "position": "P"}])
    out = dc.match_players(frame, roster)
    assert out.loc[0, ["athlete_id", "match", "roster_position"]].tolist() == ["9", "name", "P"]


def test_policy_is_dated_on_download_time():
    policy = POLICIES["depth_charts"]
    assert policy.kind == DATED and policy.date_col == "asof_utc"
    assert {"athlete_id", "jersey", "player_raw"} <= set(policy.keys)


def test_manual_override_settles_duplicate_roster_rows(monkeypatch):
    frame = pd.DataFrame([{"team": "Utah", "first_name": "Nick", "last_name": "Brown", "jersey": 13,
                           "player_raw": "Brown, Nick"}])
    roster = pd.DataFrame([{"id": i, "team": "Utah", "firstName": "Nick", "lastName": "Brown", "jersey": 13,
                            "position": "CB"} for i in (518520, 5226164)])
    monkeypatch.setattr(dc, "PLAYER_OVERRIDES", {})
    assert dc.match_players(frame, roster).loc[0, "match"] == "none"
    monkeypatch.setattr(dc, "PLAYER_OVERRIDES", {("Utah", 13, "Brown, Nick"): "5226164"})
    out = dc.match_players(frame, roster)
    assert out.loc[0, ["athlete_id", "match", "roster_position"]].tolist() == ["5226164", "manual", "CB"]
