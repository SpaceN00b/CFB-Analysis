"""Play-text parser: both text formats, penalties, names, and the table build."""
import pandas as pd
import pytest

from cfb.core import playtext as pt


def test_text_format():
    assert pt.text_format("(13:59) Shotgun #20 J.Roberts rush right for 6 yards gain to the MIZ31") == "detailed"
    assert pt.text_format("#7 D.Scott rush middle for 3 yards gain to the NDSU28 (#28 C.Johnson)") == "detailed"
    assert pt.text_format("Shotgun Sayin,Julian rush right for 7 yards gain to the MINN41") == "detailed"
    assert pt.text_format("Darius Taylor run for 1 yd to the MINN 36") == "condensed"
    assert pt.text_format(None) == "none"


def test_detailed_rush_with_two_tacklers():
    r = pt.parse_detailed("(13:59) Shotgun #20 J.Roberts rush right for 6 yards gain to the MIZ31 (#14 R.Dinkins; #4 K.Bolden)")
    assert r["action"] == "rush" and r["run_dir"] == "right" and r["text_yards"] == 6
    assert r["shotgun"] and not r["no_huddle"]
    assert r["players"] == [("rusher", "#20 J.Roberts"), ("tackler", "#14 R.Dinkins"), ("tackler", "#4 K.Bolden")]
    assert r["spots"]["end"] == "MIZ31"


def test_detailed_completion_depth_direction_and_loss():
    r = pt.parse_detailed(
        "(13:24) No Huddle-Shotgun #13 A.Simmons pass complete short middle to #0 N.Montgomery "
        "caught at MIZ27, for 4 yards loss to the MIZ27 (#19 J.Williams)")
    assert (r["action"], r["completed"], r["pass_depth"], r["pass_dir"]) == ("pass", True, "short", "middle")
    assert r["no_huddle"] and r["shotgun"]
    assert r["text_yards"] == -4
    assert r["spots"]["pass"] == "MIZ27"
    assert ("target", "#0 N.Montgomery") in r["players"] and ("tackler", "#19 J.Williams") in r["players"]


def test_detailed_incompletion_breakup_hurry():
    r = pt.parse_detailed(
        "(08:57) No Huddle-Shotgun #17 D.Mestemaker pass incomplete short right to #0 J.Bowick thrown to OSU43 "
        "broken up by #4 D.Martin QB hurried by #92 B.Butler and #15 A.Romaine PENALTY OSU Ineligible Downfield declined")
    assert r["completed"] is False and r["pass_depth"] == "short" and r["pass_dir"] == "right"
    roles = dict((role, name) for role, name in r["players"] if role != "hurry")
    assert roles == {"passer": "#17 D.Mestemaker", "target": "#0 J.Bowick", "breakup": "#4 D.Martin"}
    assert [n for role, n in r["players"] if role == "hurry"] == ["#92 B.Butler", "#15 A.Romaine"]
    assert r["spots"]["pass"] == "OSU43"


def test_detailed_names_with_spaces_suffixes_accents():
    r = pt.parse_detailed(
        "(01:46) No Huddle-Shotgun #7 C.Del Rio-Wilson pass complete short right to #0 R.Vick, Jr. "
        "caught at MAR45, for 15 yards to the MAR48 (#13 T.Brown), 1ST DOWN")
    assert r["players"][:2] == [("passer", "#7 C.Del Rio-Wilson"), ("target", "#0 R.Vick, Jr.")]
    r = pt.parse_detailed("No Huddle-Shotgun #12 K.Colón pass incomplete deep left to #1 J.Rae thrown to BUF30")
    assert r["players"] == [("passer", "#12 K.Colón"), ("target", "#1 J.Rae")] and r["pass_depth"] == "deep"


def test_detailed_sack_kneel_spike_interception_fumble():
    r = pt.parse_detailed("(11:49) No Huddle-Shotgun #8 C.Hurst sacked for loss of 6 yards to the UWG12 (#92 E.Sherman, #54 R.Reese)")
    assert r["action"] == "sack" and r["text_yards"] == -6
    assert [n for role, n in r["players"] if role == "sacker"] == ["#92 E.Sherman", "#54 R.Reese"]
    assert pt.parse_detailed("(00:12) Kneel down by #2 T.Reynolds at CLM15 for loss of 2 yards")["kneel"]
    assert pt.parse_detailed("(00:13) Kneel down at ISU09 for loss of 2 yards")["action"] == "kneel"
    assert pt.parse_detailed("(00:13) Shotgun #13 M.Reynolds pass incomplete, Spike")["action"] == "spike"
    r = pt.parse_detailed("(00:53) Shotgun #8 R.Slone pass intercepted by #22 T.Lee at UCLA20 #22 T.Lee return 27 yards to the UCLA47 (#2 J.Jackson)")
    assert r["intercepted"] and ("interceptor", "#22 T.Lee") in r["players"] and r["spots"]["pass"] == "UCLA20"
    assert not any(role == "tackler" for role, _ in r["players"])  # the return tackler is not a defender
    r = pt.parse_detailed(
        "(12:33) Shotgun #3 N.Frazier rush right for 15 yards gain to the MIZ45 fumbled by #3 N.Frazier at MIZ45 "
        "forced by #0 R.Woodyard Jr. (#0 R.Woodyard Jr.; #22 E.Dotson), out of bounds, 1ST DOWN")
    assert r["fumbled"] and ("forced_fumble", "#0 R.Woodyard Jr.") in r["players"]
    assert [n for role, n in r["players"] if role == "tackler"] == ["#0 R.Woodyard Jr.", "#22 E.Dotson"]


def test_detailed_team_actor_and_missing_actor():
    r = pt.parse_detailed("No Huddle-Shotgun Colorado pass incomplete short middle thrown to GT 20")
    assert r["action"] == "pass" and r["pass_dir"] == "middle" and r["players"] == []
    r = pt.parse_detailed("Shotgun  rush left for 10 yards gain to the SACST35 (#4 E.Tierney), 1ST DOWN")
    assert r["run_dir"] == "left" and r["players"] == [("tackler", "#4 E.Tierney")]


def test_condensed():
    r = pt.parse_condensed("Drake Lindsey pass complete to Frank Bierman for 5 yds to the MINN 30")
    assert r["completed"] and r["text_yards"] == 5
    assert r["players"] == [("passer", "Drake Lindsey"), ("target", "Frank Bierman")]
    r = pt.parse_condensed("Chad Chalich pass incomplete to Deon Watson, broken up by Jerrion Burton")
    assert r["completed"] is False and ("breakup", "Jerrion Burton") in r["players"]
    assert pt.parse_condensed("Drake Lindsey pass incomplete")["players"] == [("passer", "Drake Lindsey")]
    r = pt.parse_condensed("Brett Hundley sacked by Mustafa Jalil and Michael Barton for a loss of 1 yard to the Cal 6")
    assert r["action"] == "sack" and r["text_yards"] == -1 and len(r["players"]) == 3
    r = pt.parse_condensed("Cooper Barkate 9 Yd pass from Darian Mensah (Jake Weinberg Kick)")
    assert r["players"] == [("passer", "Darian Mensah"), ("target", "Cooper Barkate")] and r["text_yards"] == 9
    assert pt.parse_condensed("TEAM run for a loss of 3 yards to the NMex 26")["players"] == []
    assert pt.parse_condensed("(00:17) [NH] Kneel down by DUK at MIA6 (team loss of 0).")["kneel"]


@pytest.mark.parametrize("text, expect", [
    ("PENALTY OSU Holding (#71 J.Milon) 10 yards from ISU40 to ISU50. NO PLAY",
     [("OSU", "Holding", "accepted", 10, True, "#71 J.Milon")]),
    ("PENALTY Texas UNR: Unnecessary Roughness (#93 H.Kanu) 15 yards from Mich45 to Texas40",
     [("Texas", "Unnecessary Roughness", "accepted", 15, False, "#93 H.Kanu")]),
    ("PENALTY COLO Illegal Formation declined", [("COLO", "Illegal Formation", "declined", None, False, None)]),
    ("x PENALTY KSU Pass Interference offsetting SAM Ineligible Downfield on Pass offsetting. NO PLAY",
     [("KSU", "Pass Interference", "offsetting", None, True, None),
      ("SAM", "Ineligible Downfield on Pass", "offsetting", None, True, None)]),
    ("x PENALTY NC ST Holding (#71 J.Milon) 10 yards from ISU40 to ISU50", [("NC ST", "Holding", "accepted", 10, False, "#71 J.Milon")]),
    ("NOTRE DAME Penalty, Defensive Offside (Jarron Jones) to the FlaSt 36 for a 1ST down",
     [("NOTRE DAME", "Defensive Offside", "accepted", None, False, "Jarron Jones")]),
    ("GEORGIA TECH Penalty, false start (-5 Yards) to the NCaro 10", [("GEORGIA TECH", "false start", "accepted", None, False, None)]),
    ("x PENALTY on CMC-R.Blum, Offensive Holding, 1 yards, accepted. No Play.",
     [("CMC", "Offensive Holding", "accepted", 1, True, "R.Blum")]),
    ("x PENALTY Before the snap, UCO Illegal Motion on BELL, Skyler enforced 5 yards from the UCO20 to the UCO15. NO PLAY",
     [("UCO", "Illegal Motion", "accepted", 5, True, "BELL, Skyler")]),
])
def test_penalties(text, expect):
    got = [(p["team_text"], p["penalty"], p["status"], p["yards"], p["no_play"], p["name"]) for p in pt.parse_penalties(text)]
    assert got == expect


def test_split_name():
    assert pt.split_name("#0 R.Vick, Jr.") == (0, "r", "vick")
    assert pt.split_name("#7 C.Del Rio-Wilson") == (7, "c", "delriowilson")
    assert pt.split_name("Logan Jr.,Chauncey") == (None, "chauncey", "logan")
    assert pt.split_name("Reginald Bell Jr.") == (None, "reginald", "bell")
    assert pt.split_name("#12 K.Colón") == (12, "k", "colon")


def _game_index():
    return pd.DataFrame({"game_id": [1], "home_id": [10], "away_id": [20],
                         "home_team": ["Georgia"], "away_team": ["Missouri"]})


def _plays(rows):
    base = {"gameId": 1, "season": 2026, "season_type": "regular", "week": 12}
    return pd.DataFrame([{**base, "id": i, **r} for i, r in enumerate(rows)])


def test_build_tables_air_yards_and_roster_ids():
    plays = _plays([
        # Missouri (offense) at its own 25: end spot MIZ31 after 6 yards -> "MIZ" is Missouri
        {"offense": "Missouri", "defense": "Georgia", "yardsToGoal": 75, "playType": "Rush",
         "playText": "(13:59) Shotgun #20 J.Roberts rush right for 6 yards gain to the MIZ31 (#14 R.Dinkins; #4 K.Bolden)"},
        # Georgia at the MIZ 40: end spot MIZ27 after 13 yards -> "MIZ" is the defense
        {"offense": "Georgia", "defense": "Missouri", "yardsToGoal": 40, "playType": "Rush",
         "playText": "(12:00) #3 N.Frazier rush middle for 13 yards gain to the MIZ27 (#7 K.Barnes)"},
        # Missouri at own 31, caught at MIZ37 (6 air yards), 9 total yards
        {"offense": "Missouri", "defense": "Georgia", "yardsToGoal": 69, "playType": "Pass Reception",
         "playText": "(11:00) #13 A.Simmons pass complete short left to #87 B.Norfleet caught at MIZ37, for 9 yards to the MIZ40 (#14 R.Dinkins)"},
        {"offense": "Missouri", "defense": "Georgia", "yardsToGoal": 60, "playType": "Penalty",
         "playText": "(10:00) PENALTY MIZ False Start (#75 H.Creel) 5 yards from MIZ40 to MIZ35. NO PLAY"},
    ])
    roster = pd.DataFrame({
        "id": [100, 101, 102, 103, 104, 105],
        "firstName": ["Jamal", "Rasheem", "Kendall", "Aidan", "Brett", "Harold"],
        "lastName": ["Roberts", "Dinkins", "Bolden", "Simmons", "Norfleet", "Creel"],
        "team": ["Missouri", "Georgia", "Georgia", "Missouri", "Missouri", "Missouri"],
        "jersey": [20, 14, 4, 13, 87, 75], "season": [2026] * 6,
    })
    out = pt.build_play_text(plays, _game_index(), roster)
    d = out["play_detail"].set_index("play_id")
    assert d.loc["0", "run_dir"] == "right" and d.loc["0", "n_tacklers"] == 2
    assert d.loc["2", "air_yards"] == 6 and d.loc["2", "yac"] == 3
    assert d.loc["3", "action"] == "penalty" and bool(d.loc["3", "no_play"])
    pl = out["play_players"]
    assert dict(zip(pl["name"], pl["athlete_id"]))["#14 R.Dinkins"] == 101
    assert set(pl.loc[pl["role"] == "tackler", "team_id"]) == {10, 20}
    assert pl["match"].eq("jersey").sum() == len(pl) - 2  # #3 N.Frazier, #7 K.Barnes are not on the roster
    pen = out["play_penalties"].iloc[0]
    assert (pen["side"], pen["team_id"], pen["athlete_id"], pen["penalty"]) == ("offense", 20, 105, "False Start")
    for name, cols in (("play_detail", pt.DETAIL_COLUMNS), ("play_players", pt.PLAYER_COLUMNS),
                       ("play_penalties", pt.PENALTY_COLUMNS)):
        assert list(out[name].columns) == cols


def test_build_empty():
    out = pt.build_play_text(pd.DataFrame(), _game_index())
    assert all(len(v) == 0 for v in out.values())
