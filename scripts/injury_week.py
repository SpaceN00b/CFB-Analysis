"""This week's games with the injury-adjusted test model (forward use of the CFBDepth list).

model_v2 margin (fit 2015-2025, the same fit as the test in scripts/injury_select.py) plus the fixed
injury term: -(QB value over replacement x p_out) + per-share points for RB / WR / TE out
(PR 26: RB -1.94, WR -3.24, TE -3.77 per 100% of carries / targets). p_out comes from the CFBDepth
list only (current status for the team's next game, official-report entries for that game); a
player who is not listed counts as available (the `fixed_listed` version of the test). Players
listed for the game are shown with their usage. Nebraska is not on the CFBDepth list.

    python scripts/injury_week.py --week 12 [--season 2026] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_select  # noqa: E402

from cfb.lab import availability as A  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.models import MarginModel, sigmoid  # noqa: E402

DATA = Path("data")
EXP = DATA / "exploration"
CFBDEPTH = Path("/mnt/project-files/injury-reports/cfbdepth_2026-11-18/cfbdepth-player-updates.csv")
CHECKS = Path("/mnt/project-files/injury-reports/checks_2026-11-19")
DOWNLOAD = pd.Timestamp("2026-11-18")
FIXED = {"rb_out": -1.94, "wr_out": -3.24, "te_out": -3.77}
P4 = {"ACC", "Big Ten", "Big 12", "SEC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("/mnt/project-files/reports/injury-test"))
    a = ap.parse_args()
    S = a.season

    # model_v2 margins for the week
    model_select.LAST_SEASON = S
    X, _ = model_select.load()
    cols = list(C.MODEL_V2_COLUMNS)
    tr = X[(X["season"] >= C.TRAIN_START) & (X["season"] < S) & X["y"].notna()]
    wk = X[(X["season"] == S) & (X["week"] == a.week)].copy()
    m = MarginModel(cols, lam=C.LAM).fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
    wk["pred"] = m.predict_margin(wk)
    wk["lin"] = m.linear(wk)
    wk["calib_s"] = m.calib[1]

    gi = pd.read_parquet(DATA / "core" / "game_index.parquet")
    wg = gi[(gi["season"] == S) & (gi["week"] == a.week)]
    until = wg["start_time_utc"].max() + pd.Timedelta(hours=1)
    sched = A.team_schedule(DATA, [S], upcoming_until=until)
    u = pd.read_parquet(EXP / "avail_player_games.parquet")
    u = u[u["season"] == S]
    ro = A.roles(u, sched)
    ro_all = pd.concat([pd.read_parquet(EXP / "avail_roles.parquet").query("season < @S"), ro], ignore_index=True)
    teams = pd.read_parquet(DATA / "core" / "teams.parquet")
    checks = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(CHECKS / "out_batch_*.csv")))])
    checks["row_id"] = pd.to_numeric(checks["row_id"], errors="coerce").astype("Int64")
    lst = A.cfbdepth_listings(DATA, sched, teams, CFBDEPTH, checks, DOWNLOAD)
    hdi = A.hdi_listings(DATA, A.team_schedule(DATA, [S - 1, S]), teams,
                         Path("/mnt/project-files/injury-reports/hdi_raw_2026-11-18"))
    rates = A.status_rates(pd.read_parquet(EXP / "avail_roles.parquet"), hdi[hdi["source"] == f"hdi{S - 1}"])
    rates.setdefault("Out", (0.99, 0))
    rates["Available"] = (0.0, 0)
    po = A.streak_p_out(ro_all)
    po = A.overlay_listings(po, lst, rates)
    po = po[(po["season"] == S) & po["game_id"].isin(wg["game_id"])]
    # recommended version (fixed_listed): only listed players move the line; the streak is not used
    po["p_out"] = np.where(po["src"].isin(["none", "streak"]), 0.0, po["p_out"])
    mu = {S: A.qb_league_means(ro_all, S)}
    tf = A.team_features(po, mu)

    def side(df, pre, key):
        return df.rename(columns={c: f"{pre}{c}" for c in df.columns if c not in ("game_id", "team_id")}).rename(
            columns={"team_id": key})
    g = wk[["game_id", "home_id", "away_id", "pred", "lin", "calib_s"]].merge(
        side(tf, "h_", "home_id"), on=["game_id", "home_id"], how="left").merge(
        side(tf, "a_", "away_id"), on=["game_id", "away_id"], how="left")
    for c in ("qb_out",) + tuple(FIXED):
        g[f"h_{c}"] = g[f"h_{c}"].fillna(0.0)
        g[f"a_{c}"] = g[f"a_{c}"].fillna(0.0)
    inj = lambda pre: -g[f"{pre}qb_out"] + sum(g[f"{pre}{c}"] * v for c, v in FIXED.items())  # noqa: E731
    g["home_inj"] = inj("h_")
    g["away_inj"] = inj("a_")
    g["adj"] = g["home_inj"] - g["away_inj"]
    g["p_v2"] = sigmoid(g["lin"])
    g["p_test"] = sigmoid(g["lin"] + g["calib_s"] * g["adj"])
    g = g.merge(wg[["game_id", "home_team", "away_team", "start_time_utc", "neutral"]], on="game_id")
    t = teams[teams["season"] == S].set_index("team_id")
    g["home_conf"] = g["home_id"].map(t["conference"])
    g["away_conf"] = g["away_id"].map(t["conference"])
    g["p4"] = g["home_conf"].isin(P4) | g["away_conf"].isin(P4) | g["home_team"].eq("Notre Dame") | g["away_team"].eq("Notre Dame")

    # who is listed: role players with p_out > 0 and their value
    r = pd.read_parquet(DATA / "tables" / "roster" / f"{S}.parquet")
    r["athlete_id"] = pd.to_numeric(r["id"], errors="coerce")
    nm = (r["firstName"].fillna("") + " " + r["lastName"].fillna("")).groupby(r["athlete_id"]).first()
    lp = po[po["p_out"] > 0].copy()
    lp["name"] = lp["athlete_id"].map(nm)
    lp["usage"] = np.select([lp["grp"] == "QB", lp["grp"] == "RB", lp["grp"].isin(["WR", "TE"])],
                            [lp["sh_drop"], lp["sh_carry"], lp["sh_tgt"]], lp["sh_tkl"])
    lp["team"] = lp["team_id"].map(t["team"])
    lp = lp[(lp["usage"] >= 0.10) | (lp["grp"] == "QB")]
    lp["label"] = lp.apply(lambda x: f"{x['name']} {x['grp']} {x['usage']:.0%} p_out {x['p_out']:.2f} ({x['src']})", axis=1)
    who = lp.sort_values("usage", ascending=False).groupby(["game_id", "team_id"])["label"].apply(lambda s: "; ".join(s))
    g["home_out"] = [who.get((gid, h), "") for gid, h in zip(g["game_id"], g["home_id"])]
    g["away_out"] = [who.get((gid, aw), "") for gid, aw in zip(g["game_id"], g["away_id"])]

    g = g.sort_values(["p4", "start_time_utc"], ascending=[False, True])
    a.out.mkdir(parents=True, exist_ok=True)
    keep = ["game_id", "start_time_utc", "home_team", "away_team", "neutral", "pred", "adj", "p_v2", "p_test",
            "home_inj", "away_inj", "home_out", "away_out", "p4"]
    g[keep].to_csv(a.out / f"week{a.week}_{S}.csv", index=False)
    lines = [f"# Week {a.week} {S}: model_v2 vs injury-adjusted test model", "",
             f"CFBDepth list downloaded {DOWNLOAD.date()}; p_out from its current status / official-report entries "
             "for the game (players not listed count as available; Nebraska is not on the list). Adjustment = home injury points minus away injury points "
             "(negative favors the away team). Players shown: QBs and role players with 10%+ usage and p_out > 0.", "",
             "| Kickoff (ET) | Away @ Home | v2 margin | injury adj | v2 P(home) | test P(home) | home out | away out |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for x in g.itertuples(index=False):
        ko = x.start_time_utc.tz_convert("America/New_York").strftime("%a %m-%d %I:%M %p")
        lines.append(f"| {ko} | {x.away_team} @ {x.home_team}{' (N)' if x.neutral else ''} | {x.pred:+.1f} | {x.adj:+.1f} | "
                     f"{x.p_v2:.3f} | {x.p_test:.3f} | {x.home_out or '-'} | {x.away_out or '-'} |")
    (a.out / f"week{a.week}_{S}.md").write_text("\n".join(lines) + "\n")
    flips = g[(g["p_v2"] >= 0.5) != (g["p_test"] >= 0.5)]
    print(f"{len(g)} games; |adj| >= 1 pt: {(g['adj'].abs() >= 1).sum()}; pick flips: {len(flips)}")
    print(g.loc[g["adj"].abs() >= 1.5, ["away_team", "home_team", "pred", "adj", "p_v2", "p_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
