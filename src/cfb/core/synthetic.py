"""A small fake CFBD store, shaped like the real one, for tests that run without the snapshot.

    from cfb.core.synthetic import synthetic_warehouse
    wh = synthetic_warehouse(tmp_path)        # raw store + core tables + unit_ratings

Eight FBS teams in two conferences and two FCS teams play three seasons: five regular
weeks and three bowls each. The last season stops after week 3, like a snapshot taken
in season: later games are scheduled but not played. Stronger teams score more, gain
more per play and carry higher Elo, so features built on it behave plausibly. Year-level
tables (ratings_sp, records, coaches) contain same-season results, as the real ones do.

Tables written: calendar, venues, teams_fbs, games, games_teams, plays, drives,
stats_game_advanced, ppa_games, lines, talent, returning_production, ratings_sp,
ratings_elo, records, coaches. For any other table, build a frame in the test and
call `warehouse.put(name, frame)`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..store import Store
from .asof import Warehouse
from .build import build_core
from .unit_ratings import build_unit_ratings

FBS = [(i + 1, f"Team {c}", "East" if i < 4 else "West") for i, c in enumerate("ABCDEFGH")]
FCS = [(101, "FCS One", "Lower"), (102, "FCS Two", "Lower")]
SEASONS = (2021, 2022, 2023)
REGULAR_WEEKS = 5
PLAYED_WEEKS_LAST_SEASON = 3
_PLAY_TYPES = [
    ("Rush", 0.44), ("Pass Reception", 0.24), ("Pass Incompletion", 0.14), ("Sack", 0.03),
    ("Punt", 0.05), ("Kickoff", 0.03), ("Penalty", 0.02), ("Rushing Touchdown", 0.02),
    ("Passing Touchdown", 0.02), ("Fumble Recovery (Opponent)", 0.01),
]
_SCRIMMAGE = {"Rush", "Pass Reception", "Pass Incompletion", "Sack", "Rushing Touchdown",
              "Passing Touchdown", "Fumble Recovery (Opponent)"}


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _round_robin(ids: list[int], rnd: int) -> list[tuple[int, int]]:
    rot = [ids[0]] + ids[1:][-(rnd % (len(ids) - 1)):] + ids[1:][: -(rnd % (len(ids) - 1)) or None]
    half = len(rot) // 2
    pairs = list(zip(rot[:half], reversed(rot[half:])))
    return [(a, b) if (rnd + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(pairs)]


def write_synthetic_store(root: str | Path, seed: int = 7) -> Store:
    """Write the raw datasets under `root` with the same keys ingest uses."""
    rng = np.random.default_rng(seed)
    store = Store(root)
    names = {tid: name for tid, name, _ in FBS + FCS}
    conf = {tid: c for tid, _, c in FBS + FCS}
    strength = {tid: float(s) for (tid, _, _), s in zip(FBS, rng.normal(0, 1, len(FBS)))}
    strength.update({tid: -2.5 for tid, _, _ in FCS})
    elo = {tid: 1500.0 + 150 * strength[tid] for tid in strength}

    store.write("venues", "all", [
        {"id": 9000 + tid, "name": f"{names[tid]} Stadium", "capacity": 40000 + 1000 * tid, "grass": bool(tid % 2),
         "dome": False, "city": f"City {tid}", "state": "TX", "zip": "70000", "countryCode": "US",
         "timezone": "America/Chicago", "latitude": 30.0 + tid, "longitude": -95.0 - tid,
         "elevation": str(100.0 * tid), "constructionYear": 1950 + tid}
        for tid, _, _ in FBS
    ] + [{"id": 9900, "name": "Neutral Bowl", "capacity": 65000, "grass": True, "dome": True, "city": "Bowl City",
          "state": "FL", "zip": "33000", "countryCode": "US", "timezone": "America/New_York",
          "latitude": 28.0, "longitude": -81.0, "elevation": "10.0", "constructionYear": 1990}])

    game_id = 400000000
    for season in SEASONS:
        base = pd.Timestamp(f"{season}-08-30T07:00:00Z")
        base -= pd.Timedelta(days=base.dayofweek)  # Monday
        calendar = []
        for wk in range(1, REGULAR_WEEKS + 1):
            start = base + pd.Timedelta(days=7 * (wk - 1))
            end = start + pd.Timedelta(days=7) - pd.Timedelta(minutes=1)
            calendar.append({"season": season, "week": wk, "seasonType": "regular", "startDate": _iso(start),
                             "endDate": _iso(end), "firstGameStart": _iso(start), "lastGameStart": _iso(end)})
        post_start = base + pd.Timedelta(days=7 * REGULAR_WEEKS)
        post_end = post_start + pd.Timedelta(days=30) - pd.Timedelta(minutes=1)
        calendar.append({"season": season, "week": 1, "seasonType": "postseason", "startDate": _iso(post_start),
                         "endDate": _iso(post_end), "firstGameStart": _iso(post_start), "lastGameStart": _iso(post_end)})
        store.write("calendar", str(season), calendar, {"season": season})

        # schedule: (season_type, week, home, away, kickoff, neutral)
        slate = []
        fbs_ids = [t[0] for t in FBS]
        for wk in range(1, REGULAR_WEEKS + 1):
            saturday = base + pd.Timedelta(days=7 * (wk - 1) + 5)
            pairs = _round_robin(fbs_ids, wk + season)
            if wk == 1:  # two FBS teams open against FCS opponents instead
                (h1, a1), (h2, a2) = pairs[0], pairs[1]
                pairs = [(h1, FCS[0][0]), (a1, FCS[1][0])] + pairs[2:] + [(h2, a2)]
            for i, (home, away) in enumerate(pairs):
                slate.append(("regular", wk, home, away, saturday + pd.Timedelta(hours=10 + 3 * i), False))
        bowl_day = post_start + pd.Timedelta(days=12)
        # The second bowl kicks off 26 hours after the first, so the first is still
        # being played 24 hours before the second.
        slate.append(("postseason", 1, fbs_ids[0], fbs_ids[5], bowl_day + pd.Timedelta(hours=13), True))
        slate.append(("postseason", 1, fbs_ids[2], fbs_ids[7], bowl_day + pd.Timedelta(hours=39), True))
        slate.append(("postseason", 1, fbs_ids[1], fbs_ids[4], bowl_day + pd.Timedelta(days=9, hours=17), True))

        games, lines, wins, margins = [], [], {t: 0 for t in names}, {t: [] for t in names}
        played_n = {t: 0 for t in names}
        by_week: dict[tuple[str, int], dict[str, list]] = {}
        for stype, wk, home, away, kickoff, neutral in slate:
            game_id += 1
            played = not (season == SEASONS[-1] and (stype == "postseason" or wk > PLAYED_WEEKS_LAST_SEASON))
            pre_h, pre_a = elo[home], elo[away]
            edge = 7.0 * (strength[home] - strength[away]) + (0.0 if neutral else 2.5)
            hp = ap = None
            post_h = post_a = wp = None
            if played:
                hp = int(max(0, round(27 + edge / 2 + rng.normal(0, 9))))
                ap = int(max(0, round(27 - edge / 2 + rng.normal(0, 9))))
                if hp == ap:
                    hp += 3
                expect = 1 / (1 + 10 ** (-(pre_h - pre_a + (0 if neutral else 55)) / 400))
                delta = 25 * ((hp > ap) - expect)
                elo[home], elo[away] = pre_h + delta, pre_a - delta
                post_h, post_a = round(elo[home]), round(elo[away])
                wp = float(np.clip(0.5 + (hp - ap) / 40 + rng.normal(0, 0.05), 0.01, 0.99))
                wins[home] += hp > ap
                wins[away] += ap > hp
                margins[home].append(hp - ap)
                margins[away].append(ap - hp)
                played_n[home] += 1
                played_n[away] += 1
            fbs_game = home < 100 and away < 100
            row = {
                "id": game_id, "season": season, "week": wk, "seasonType": stype, "startDate": _iso(kickoff),
                "startTimeTBD": False, "completed": played, "neutralSite": neutral,
                "conferenceGame": conf[home] == conf[away] and not neutral,
                "attendance": int(30000 + rng.integers(0, 20000)) if played else None,
                "venueId": 9900 if neutral else 9000 + home, "venue": "Neutral Bowl" if neutral else f"{names[home]} Stadium",
                "homeId": home, "homeTeam": names[home], "homeClassification": "fbs" if home < 100 else "fcs",
                "homeConference": conf[home], "homePoints": hp,
                "homeLineScores": [hp // 4] * 4 if played else None,
                "homePostgameWinProbability": wp, "homePregameElo": round(pre_h) if fbs_game else None,
                "homePostgameElo": post_h if fbs_game else None,
                "awayId": away, "awayTeam": names[away], "awayClassification": "fbs" if away < 100 else "fcs",
                "awayConference": conf[away], "awayPoints": ap,
                "awayLineScores": [ap // 4] * 4 if played else None,
                "awayPostgameWinProbability": (1 - wp) if played else None,
                "awayPregameElo": round(pre_a) if fbs_game else None, "awayPostgameElo": post_a if fbs_game else None,
                "excitementIndex": float(rng.uniform(1, 9)) if played else None, "highlights": "", "notes": None,
            }
            games.append(row)
            if fbs_game:
                spread = -round(((pre_h - pre_a) / 25 + (0 if neutral else 2.5)) * 2) / 2
                lines.append({
                    "id": game_id, "season": season, "seasonType": stype, "week": wk, "startDate": _iso(kickoff),
                    "homeTeamId": home, "homeTeam": names[home], "homeConference": conf[home],
                    "homeClassification": "fbs", "homeScore": hp, "awayTeamId": away, "awayTeam": names[away],
                    "awayConference": conf[away], "awayClassification": "fbs", "awayScore": ap,
                    "lines": [{"provider": "consensus", "spread": spread, "formattedSpread": f"{names[home]} {spread}",
                               "spreadOpen": spread + 0.5, "overUnder": 54.5, "overUnderOpen": 55.0,
                               "homeMoneyline": -150 if spread < 0 else 130, "awayMoneyline": 130 if spread < 0 else -150}],
                })
            bucket = by_week.setdefault((stype, wk), {"teams": [], "plays": [], "drives": [], "adv": [], "ppa": []})
            if played:
                _play_game(rng, bucket, row, strength, names, conf)

        for stype in ("regular", "postseason"):
            extra = {"season": season, "season_type": stype}
            key = f"{season}_{stype}"
            store.write("games", key, [g for g in games if g["seasonType"] == stype], extra)
            store.write("lines", key, [ln for ln in lines if ln["seasonType"] == stype], extra)
            for name, part in (("drives", "drives"), ("stats_game_advanced", "adv"), ("ppa_games", "ppa")):
                rows = [r for (st, _), b in by_week.items() if st == stype for r in b[part]]
                store.write(name, key, rows, extra)
        for (stype, wk), bucket in by_week.items():
            extra = {"season": season, "season_type": stype, "week": wk}
            key = f"{season}_{stype}_{wk:02d}"
            store.write("games_teams", key, bucket["teams"], extra)
            store.write("plays", key, bucket["plays"], extra)

        # Year-level tables. Like the real ones they are "as of the pull": they include
        # every result of the season, which is why same-season access is banned.
        year = {"season": season}
        store.write("teams_fbs", str(season), [
            {"id": tid, "school": name, "mascot": "Mascots", "abbreviation": name[-1], "conference": c,
             "classification": "fbs", "location": {"id": 9000 + tid, "name": f"{name} Stadium", "city": f"City {tid}",
                                                    "state": "TX", "timezone": "America/Chicago", "latitude": 30.0 + tid,
                                                    "longitude": -95.0 - tid, "elevation": str(100.0 * tid),
                                                    "capacity": 40000 + 1000 * tid, "grass": bool(tid % 2), "dome": False}}
            for tid, name, c in FBS], year)
        store.write("talent", str(season), [
            {"year": season, "team": name, "talent": round(700 + 80 * strength[tid] + rng.normal(0, 10), 2)}
            for tid, name, _ in FBS], year)
        store.write("returning_production", str(season), [
            {"season": season, "team": name, "conference": c, "percentPPA": float(rng.uniform(0.3, 0.9)),
             "totalPPA": float(rng.uniform(100, 400)), "usage": float(rng.uniform(0.3, 0.9))}
            for tid, name, c in FBS], year)
        mean_margin = {t: (float(np.mean(m)) if m else 0.0) for t, m in margins.items()}
        store.write("ratings_sp", str(season), [
            {"year": season, "team": name, "conference": c, "rating": round(mean_margin[tid], 2), "ranking": None,
             "offense": {"rating": round(28 + mean_margin[tid] / 2, 2)}, "defense": {"rating": round(28 - mean_margin[tid] / 2, 2)}}
            for tid, name, c in FBS], year)
        store.write("ratings_elo", str(season), [
            {"year": season, "team": name, "conference": c, "elo": int(round(elo[tid]))} for tid, name, c in FBS], year)
        store.write("records", str(season), [
            {"year": season, "teamId": tid, "team": name, "classification": "fbs", "conference": c,
             "total": {"games": played_n[tid], "wins": int(wins[tid]), "losses": played_n[tid] - int(wins[tid]), "ties": 0}}
            for tid, name, c in FBS], year)
        store.write("coaches", str(season), [
            {"id": 500 + tid, "firstName": "Coach", "lastName": name[-1], "hireDate": f"{SEASONS[0] - 2}-12-01T00:00:00.000Z",
             "seasons": [{"teamId": tid, "school": name, "conference": c, "year": season, "games": played_n[tid],
                          "wins": int(wins[tid]), "losses": played_n[tid] - int(wins[tid]), "ties": 0}]}
            for tid, name, c in FBS], year)
    return store


def _play_game(rng, bucket: dict, game: dict, strength: dict, names: dict, conf: dict) -> None:
    """Append one played game's box score, plays, drives and advanced stats to `bucket`."""
    gid, home, away = game["id"], game["homeId"], game["awayId"]
    types = [t for t, _ in _PLAY_TYPES]
    probs = np.array([p for _, p in _PLAY_TYPES])
    probs = probs / probs.sum()
    box, summaries = [], {}
    play_no = 0
    for side, (off, dfn) in (("home", (home, away)), ("away", (away, home))):
        n = int(rng.integers(55, 75))
        kinds = rng.choice(types, size=n, p=probs)
        edge = 0.12 * (strength[off] - strength[dfn])
        off_pts, def_pts = game[f"{side}Points"], game["awayPoints" if side == "home" else "homePoints"]
        rows = []
        for i, kind in enumerate(kinds):
            play_no += 1
            scrimmage = kind in _SCRIMMAGE
            ppa = float(rng.normal(0.02 + edge, 0.9)) if scrimmage else None
            down = int(rng.integers(1, 5))
            distance = int(rng.integers(1, 11))
            gained = int(round((ppa or 0) * 6 + rng.normal(3, 3))) if scrimmage else 0
            period = min(4, 1 + (4 * i) // n)
            drive = 1 + i // 6
            rows.append({
                "gameId": gid, "driveId": f"{gid}{'1' if side == 'home' else '2'}{drive:02d}",
                "id": f"{gid}{play_no:05d}", "driveNumber": drive, "playNumber": 1 + i % 6,
                "offense": names[off], "offenseConference": conf[off], "offenseScore": int(off_pts * (i + 1) / n),
                "defense": names[dfn], "defenseConference": conf[dfn], "defenseScore": int(def_pts * (i + 1) / n),
                "home": names[home], "away": names[away], "period": period, "offenseTimeouts": 3.0,
                "defenseTimeouts": 3.0, "yardline": 50, "yardsToGoal": int(rng.integers(1, 99)), "down": down,
                "distance": distance, "yardsGained": gained, "scoring": kind.endswith("Touchdown"),
                "playType": kind, "playText": f"{kind} for {gained} yds", "ppa": ppa, "wallclock": None,
                "clock": {"minutes": int(14 - (i % 15)), "seconds": int(rng.integers(0, 60))},
            })
        bucket["plays"] += rows
        scr = [r for r in rows if r["ppa"] is not None]
        rush = [r for r in scr if r["playType"] in ("Rush", "Rushing Touchdown")]
        pas = [r for r in scr if r not in rush]
        mean = lambda xs: float(np.mean([x["ppa"] for x in xs])) if xs else 0.0
        summaries[off] = {"plays": len(scr), "ppa": mean(scr), "rush": mean(rush), "pass": mean(pas),
                          "rush_n": len(rush), "pass_n": len(pas)}
        comp = sum(r["playType"] in ("Pass Reception", "Passing Touchdown") for r in rows)
        box.append({"teamId": off, "team": names[off], "conference": conf[off], "homeAway": side, "points": off_pts, "stats": [
            {"category": "rushingAttempts", "stat": str(len(rush))},
            {"category": "rushingYards", "stat": str(sum(r["yardsGained"] for r in rush))},
            {"category": "completionAttempts", "stat": f"{comp}-{len(pas)}"},
            {"category": "netPassingYards", "stat": str(sum(r["yardsGained"] for r in pas))},
            {"category": "totalYards", "stat": str(sum(r["yardsGained"] for r in scr))},
            {"category": "turnovers", "stat": str(int(rng.integers(0, 4)))},
            {"category": "thirdDownEff", "stat": f"{int(rng.integers(2, 8))}-{int(rng.integers(9, 16))}"},
            {"category": "totalPenaltiesYards", "stat": f"{int(rng.integers(2, 9))}-{int(rng.integers(20, 90))}"},
            {"category": "possessionTime", "stat": f"{int(rng.integers(24, 36))}:{int(rng.integers(0, 60)):02d}"},
            {"category": "firstDowns", "stat": str(int(rng.integers(12, 30)))},
        ]})
        for d in range(1, 1 + n // 6):
            bucket["drives"].append({
                "id": f"{gid}{'1' if side == 'home' else '2'}{d:02d}", "gameId": gid, "offense": names[off],
                "offenseConference": conf[off], "defense": names[dfn], "defenseConference": conf[dfn],
                "driveNumber": d, "scoring": bool(d % 4 == 0), "startPeriod": min(4, 1 + d // 3), "startYardline": 25,
                "startYardsToGoal": 75, "endPeriod": min(4, 1 + d // 3), "endYardline": 50, "endYardsToGoal": 50,
                "plays": 6, "yards": 25, "driveResult": "TD" if d % 4 == 0 else "PUNT", "isHomeOffense": side == "home",
                "startOffenseScore": 0, "startDefenseScore": 0, "endOffenseScore": 7 if d % 4 == 0 else 0,
                "endDefenseScore": 0, "startTime": {"minutes": 15, "seconds": 0}, "endTime": {"minutes": 12, "seconds": 30},
                "elapsed": {"minutes": 2, "seconds": 30},
            })
    bucket["teams"].append({"id": gid, "teams": box})
    for team, opp in ((home, away), (away, home)):
        mine, theirs = summaries[team], summaries[opp]
        side_stats = lambda s: {
            "plays": s["plays"], "drives": 10, "ppa": s["ppa"], "totalPPA": s["ppa"] * s["plays"],
            "successRate": 0.42, "explosiveness": 1.2,
            "rushingPlays": {"ppa": s["rush"], "totalPPA": s["rush"] * s["rush_n"], "successRate": 0.4, "explosiveness": 1.0},
            "passingPlays": {"ppa": s["pass"], "totalPPA": s["pass"] * s["pass_n"], "successRate": 0.44, "explosiveness": 1.4},
        }
        bucket["adv"].append({"gameId": gid, "season": game["season"], "week": game["week"], "seasonType": game["seasonType"],
                              "team": names[team], "opponent": names[opp], "offense": side_stats(mine), "defense": side_stats(theirs)})
        if team < 100:
            ppa_side = lambda s: {"overall": s["ppa"], "passing": s["pass"], "rushing": s["rush"],
                                  "firstDown": s["ppa"], "secondDown": s["ppa"], "thirdDown": s["ppa"]}
            bucket["ppa"].append({"gameId": gid, "season": game["season"], "week": game["week"],
                                  "seasonType": game["seasonType"], "team": names[team], "conference": conf[team],
                                  "opponent": names[opp], "offense": ppa_side(mine), "defense": ppa_side(theirs)})


def synthetic_warehouse(root: str | Path, seed: int = 7, unit_ratings: bool = True) -> Warehouse:
    """Write the fake store under `root`, build core tables and unit_ratings, return a Warehouse."""
    write_synthetic_store(root, seed)
    build_core(root)
    warehouse = Warehouse(root)
    if unit_ratings:
        warehouse.put("unit_ratings", build_unit_ratings(warehouse), persist=True)
    return warehouse


