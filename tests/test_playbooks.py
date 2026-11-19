"""Team playbooks test bench (cfb.lab.playbooks): play-type labels, the stat-crew adjustment and
the pass-rate-over-expected fit on small synthetic inputs."""
import numpy as np
import pandas as pd

from cfb.lab import playbooks as pb


def _bundle():
    n = 6
    plays = pd.DataFrame({
        "play_id": [str(i) for i in range(n)], "game_id": [10] * n, "season": [2026] * n,
        "offense_id": [1] * n, "defense_id": [2] * n,
        "action": ["rush", "rush", "pass", "pass", "sack", "pass"],
        "shotgun": pd.array([False, True, True, True, True, pd.NA], dtype="boolean"),
        "no_huddle": pd.array([False, True, False, True, True, False], dtype="boolean"),
        "run_dir": pd.array(["middle", "left", pd.NA, pd.NA, pd.NA, pd.NA], dtype="string"),
        "pass_dir": pd.array([pd.NA, pd.NA, "right", "left", pd.NA, pd.NA], dtype="string"),
        "air_yards": pd.array([pd.NA, pd.NA, -2, 24, pd.NA, pd.NA], dtype="Float64"),
    })
    players = pd.DataFrame({
        "play_id": ["0", "1", "2", "3", "2", "3"],
        "role": ["rusher", "rusher", "target", "target", "passer", "passer"],
        "athlete_id": pd.array([100, 200, 300, 400, 200, 200], dtype="Int64"),
        "team_id": [1] * 6, "side": ["offense"] * 6,
    })
    info = pd.DataFrame({"season": [2026] * 4, "athlete_id": pd.array([100, 200, 300, 400], dtype="Int64"),
                         "player": ["Back", "Quarterback", "Wideout", "Tight End"],
                         "position": ["RB", "QB", "WR", "TE"]})
    return {"plays": plays, "players": players, "info": info}


def test_label_plays_groups_and_plays():
    p = pb.label_plays(_bundle()).set_index("play_id")
    assert p.loc["0", ["formation", "group", "play", "ball_player"]].tolist() == [
        "Under center", "RB run middle", "RB run middle", "Back"]
    assert p.loc["1", ["formation", "tempo", "group", "play"]].tolist() == ["Shotgun", "No huddle", "QB run", "QB run left"]
    assert p.loc["2", ["group", "play", "ball_player"]].tolist() == [
        "Screen / behind LOS", "Screen / behind LOS right to WR", "Wideout"]
    assert p.loc["3", ["group", "play"]].tolist() == ["Deep 20+", "Deep 20+ left to TE"]
    assert p.loc["4", ["group", "play"]].tolist() == ["Sack", "Sack"]
    assert p.loc["5", ["group", "play", "formation"]].tolist() == ["Pass, no target", "Pass, no target", "Under center"]
    assert p["dropback"].tolist() == [False, False, True, True, True, True]


def test_crew_adjustment_recovers_team_share():
    rng = np.random.default_rng(0)
    team = {1: 0.2, 2: 0.5, 3: 0.8}
    crew = {11: 0.0, 12: 0.15, 13: -0.15}
    rows, games, gid = [], [], 0
    for _ in range(40):
        for venue in crew:
            off = int(rng.choice(list(team)))
            gid += 1
            games.append({"game_id": gid, "venue_id": venue})
            share = np.clip(team[off] + crew[venue], 0, 1)
            for k in range(60):
                rows.append({"game_id": gid, "offense_id": off, "no_huddle": bool(rng.random() < share)})
    est = pb.crew_adjusted(pd.DataFrame(rows), "no_huddle", pd.DataFrame(games))
    for t, v in team.items():
        assert abs(est[t] - v) < 0.05


def test_expected_pass_follows_down_and_distance():
    rng = np.random.default_rng(1)
    n = 4000
    p = pd.DataFrame({
        "down": rng.integers(1, 4, n), "distance": rng.integers(1, 15, n),
        "yards_to_goal": rng.integers(5, 95, n), "score_diff": rng.integers(-14, 15, n),
        "period": rng.integers(1, 5, n), "clock_seconds": rng.integers(0, 900, n),
    })
    truth = 1 / (1 + np.exp(-(-1.0 + 0.9 * (p["down"] == 3) + 0.12 * p["distance"])))
    p["dropback"] = rng.random(n) < truth
    q = pb.expected_pass(p)
    third_long = (p["down"] == 3) & (p["distance"] >= 10)
    first_short = (p["down"] == 1) & (p["distance"] <= 3)
    assert q[third_long].mean() > 0.7 > 0.4 > q[first_short].mean()
    assert abs(q.mean() - p["dropback"].mean()) < 0.01


def test_scope_ids_p4_includes_notre_dame():
    ts = pd.DataFrame({"team_id": [1, 2, 3, 4, 5], "team": ["A", "B", "Notre Dame", "C", "D"],
                       "conference": ["SEC", "Big Ten", "FBS Independents", "Sun Belt", "FBS Independents"]})
    assert pb.scope_ids(ts, "P4") == [1, 2, 3]
    assert pb.scope_ids(ts, "SEC") == [1]
