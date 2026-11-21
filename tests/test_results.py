"""Published picks vs results: grading a stored bundle against final scores."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from cfb.report import results as R
from test_ui import game, make_bundle, team


def graded_bundle():
    b = make_bundle()
    a, c = team(1, "Alpha"), team(3, "Gamma", conf="Big Ten")
    g = game(12, a, c, 0.55)
    g["fbs_vs_fbs"] = False
    b["games"].append(g)
    b["meta"]["phase2"] = True
    return b


def test_grade_game_rows():
    b = graded_bundle()
    # 10: Alpha (home, p 0.7, -3.5) wins 30-20: pick right, covers (W), total 50 hit exactly (P).
    # 11: Gamma home p 0.4 -> pick Beta away; Gamma wins 24-21: wrong; home -3.5 -> 24-21-3.5 < 0, home ATS L.
    g = R.grade(b, {10: (30.0, 20.0), 11: (24.0, 21.0), 12: (14.0, 14.0)})
    rows = {r["game_id"]: r for r in g["games"]}
    assert rows[10]["pick_correct"] is True and rows[10]["ats_result"] == "W" and rows[10]["ou_result"] == "P"
    assert rows[10]["log_loss"] == round(-math.log(0.7), 4)
    assert rows[11]["pick"] == "Beta" and rows[11]["pick_correct"] is False
    assert rows[11]["ats_result"] == "L" and rows[11]["ou_result"] == "L"   # over 50, scored 45
    assert rows[11]["log_loss"] == round(-math.log(0.4), 4)
    assert rows[12]["final"] is False and rows[12]["pick_correct"] is None   # a tie is not graded
    assert g["counts"] == {"games": 3, "final": 2, "pending": 1, "fbs_vs_fbs_final": 2}
    assert g["picks"] == {"n": 2, "correct": 1}
    v2 = g["models"]["model_v2"]
    assert v2["primary"] and v2["n"] == 2 and v2["correct"] == 1
    assert v2["log_loss"] == pytest.approx((-math.log(0.7) - math.log(0.4)) / 2, abs=1e-4)
    assert g["models"]["baseline_close"]["correct"] == 2     # 0.6 home; home won both
    assert g["ats"] == {"won": 1, "lost": 1, "push": 0, "n": 2, "hit_rate": 0.5}
    assert g["ou"]["push"] == 1 and g["ou"]["hit_rate"] == 0.0


def test_stored_values_are_used_not_recomputed():
    b = graded_bundle()
    b["games"][0]["p"]["model_v2"] = 0.123          # whatever the run stored is what gets graded
    b["games"][0]["pick"] = {"side": "away", "team": "Beta", "p": 0.877}
    g = R.grade(b, {10: (30.0, 20.0)})
    r = next(r for r in g["games"] if r["game_id"] == 10)
    assert r["pick"] == "Beta" and r["pick_correct"] is False and r["log_loss"] == round(-math.log(0.123), 4)


def test_pending_week_and_frame():
    g = R.grade(graded_bundle(), {})
    assert g["counts"]["final"] == 0 and g["models"] == {} and g["ats"] is None
    df = R.frame(g)
    assert list(df["game_id"]) == [10, 11, 12] and df["score"].isna().all()
    assert "pending" in R.render_results(g)


def test_results_sources():
    b = make_bundle()
    assert R.results_from_bundles([b]) == {9: (10.0, 20.0)}
    out = pd.DataFrame({"game_id": [1, 2], "home_points": [21, None], "away_points": [14, None]})
    assert R.results_from_outcomes(out) == {1: (21.0, 14.0)}


def test_later_bundles_and_cli(tmp_path, capsys):
    reports = tmp_path / "data" / "reports"
    week12 = graded_bundle()
    week13 = make_bundle()
    week13["previous"] = {"label": "2026 week 12", "games": [
        {**week12["games"][0], "result": {"home_points": 30, "away_points": 20}}]}
    for key, b in (("2026_regular_12", week12), ("2026_regular_13", week13), ("2026_regular_11", make_bundle())):
        (reports / key).mkdir(parents=True)
        (reports / key / "bundle.json").write_text(json.dumps(b), encoding="utf-8")
    assert len(R.later_bundles(reports, "2026_regular_12")) == 1
    out = tmp_path / "r.json"
    assert R.main(["--season", "2026", "--week", "12", "--data", str(tmp_path / "data"), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2026 week 12: published picks vs results" in text and "Final: 1 of 3" in text
    assert json.loads(out.read_text())["picks"] == {"n": 1, "correct": 1}
    assert R.main(["--season", "2026", "--week", "9", "--data", str(tmp_path / "data")]) == 2


def test_away_spread_pick_uses_its_own_line():
    a, b = team(1, "Alpha"), team(2, "Beta")
    g = game(20, a, b, 0.3)                       # away Beta favoured by 9: home line +9, Beta -9
    g["market"].update(spread=9.0)
    g["ats"] = {"side": "away", "team": "Beta", "line": -9.0, "model_line": 11.0}
    assert R.grade_game(g, (3.0, 20.0), "model_v2")["ats_result"] == "W"    # won by 17
    assert R.grade_game(g, (14.0, 20.0), "model_v2")["ats_result"] == "L"   # won by 6
    assert R.grade_game(g, (11.0, 20.0), "model_v2")["ats_result"] == "P"
