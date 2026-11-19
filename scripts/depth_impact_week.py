"""This week's picks with the depth-chart injury term (2026-11-19, test bench).

Josiah (2026-11-19): the adjusted charts are for backtesting; this week uses the current team charts
less every player listed on the injury reports. So each listed-out full-strength starter (most touches
so far this season, `dc_total`) costs his value over the next man up on this week's chart
(scripts/depth_impact_select.py writes team_costs_week<N>_2026.csv and the impact scores). Teams
without a chart use the usage replacement. model_v2 margin (fit 2015-2025, as in PR 34's
injury_week.py) minus home cost plus away cost, through model_v2's calibration slope. PR 34's
fixed_listed picks are shown for comparison when its week file exists.

    python scripts/depth_impact_select.py --week 12
    python scripts/depth_impact_week.py --week 12 [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_select  # noqa: E402

from cfb.model import config as C  # noqa: E402
from cfb.model.models import MarginModel, sigmoid  # noqa: E402

DATA = Path("data")
EXP = DATA / "exploration" / "depth_impact"
PR34 = Path("/mnt/project-files/reports/injury-test")
P4 = {"ACC", "Big Ten", "Big 12", "SEC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("/mnt/project-files/reports/depth-impact"))
    a = ap.parse_args()
    S = a.season

    model_select.LAST_SEASON = S
    X, _ = model_select.load()
    cols = list(C.MODEL_V2_COLUMNS)
    tr = X[(X["season"] >= C.TRAIN_START) & (X["season"] < S) & X["y"].notna()]
    wk = X[(X["season"] == S) & (X["week"] == a.week)].copy()
    m = MarginModel(cols, lam=C.LAM).fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
    wk["pred"] = m.predict_margin(wk)
    wk["lin"] = m.linear(wk)
    wk["calib_s"] = m.calib[1]

    cost = pd.read_csv(EXP / f"team_costs_week{a.week}_{S}.csv").set_index(["game_id", "team_id"])["cost"]
    sc = pd.read_csv(EXP / f"impact_scores_week{a.week}_{S}.csv")
    gi = pd.read_parquet(DATA / "core" / "game_index.parquet")
    teams = pd.read_parquet(DATA / "core" / "teams.parquet")
    t = teams[teams["season"] == S].set_index("team_id")
    g = wk[["game_id", "home_id", "away_id", "pred", "lin", "calib_s"]].merge(
        gi[["game_id", "home_team", "away_team", "start_time_utc", "neutral"]], on="game_id")
    g["home_cost"] = [cost.get((x, h), 0.0) for x, h in zip(g["game_id"], g["home_id"])]
    g["away_cost"] = [cost.get((x, w), 0.0) for x, w in zip(g["game_id"], g["away_id"])]
    g["adj"] = -(g["home_cost"] - g["away_cost"])
    g["p_v2"] = sigmoid(g["lin"])
    g["p_dc"] = sigmoid(g["lin"] + g["calib_s"] * g["adj"])
    g["p4"] = (g["home_id"].map(t["conference"]).isin(P4) | g["away_id"].map(t["conference"]).isin(P4)
               | g["home_team"].eq("Notre Dame") | g["away_team"].eq("Notre Dame"))
    f34 = PR34 / f"week{a.week}_{S}.csv"
    if f34.exists():
        p34 = pd.read_csv(f34)[["game_id", "adj", "p_test"]].rename(columns={"adj": "adj_pr34", "p_test": "p_pr34"})
        g = g.merge(p34, on="game_id", how="left")
    else:
        g["adj_pr34"] = g["p_pr34"] = np.nan

    sc = sc[sc["cost"] >= 0.25].copy()
    sc["label"] = [f"{p} {gr} out p {po:.2f}, next {r if isinstance(r, str) else 'league backup'} ({c:.1f})"
                   for p, gr, po, r, c in zip(sc["player"], sc["grp"], sc["p_out"], sc["replacement"], sc["cost"])]
    sc["team_id"] = sc["team"].map({v: k for k, v in t["team"].items()})
    who = sc.sort_values("cost", ascending=False).groupby(["game_id", "team_id"])["label"].apply("; ".join)
    g["home_out"] = [who.get((x, h), "") for x, h in zip(g["game_id"], g["home_id"])]
    g["away_out"] = [who.get((x, w), "") for x, w in zip(g["game_id"], g["away_id"])]
    g = g.sort_values(["p4", "start_time_utc"], ascending=[False, True])
    a.out.mkdir(parents=True, exist_ok=True)
    keep = ["game_id", "start_time_utc", "home_team", "away_team", "neutral", "pred", "adj", "p_v2", "p_dc",
            "adj_pr34", "p_pr34", "home_cost", "away_cost", "home_out", "away_out", "p4"]
    g[keep].to_csv(a.out / f"week{a.week}_{S}.csv", index=False)
    lines = [f"# Week {a.week} {S}: model_v2 with the depth-chart injury term", "",
             "This week's charts = current team charts less every player listed out on the CFBDepth list "
             "(Out, Out for Season, Questionable). Cost of a listed-out starter = his value over the next man up "
             "on that chart, times P(out). Adjustment = away cost minus home cost (negative favors the away team). "
             "PR 34 = injury-adjusted test model (fixed_listed). Players shown: cost >= 0.25 points.", "",
             "| Kickoff (ET) | Away @ Home | v2 margin | adj | PR 34 adj | v2 P(home) | depth-chart P(home) | PR 34 P(home) | home out | away out |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for x in g.itertuples(index=False):
        ko = pd.Timestamp(x.start_time_utc).tz_convert("America/New_York").strftime("%a %m-%d %I:%M %p")
        a34 = f"{x.adj_pr34:+.1f}" if pd.notna(x.adj_pr34) else "-"
        p34 = f"{x.p_pr34:.3f}" if pd.notna(x.p_pr34) else "-"
        lines.append(f"| {ko} | {x.away_team} @ {x.home_team}{' (N)' if x.neutral else ''} | {x.pred:+.1f} | {x.adj:+.1f} | "
                     f"{a34} | {x.p_v2:.3f} | {x.p_dc:.3f} | {p34} | {x.home_out or '-'} | {x.away_out or '-'} |")
    (a.out / f"week{a.week}_{S}.md").write_text("\n".join(lines) + "\n")
    flips = g[(g["p_v2"] >= 0.5) != (g["p_dc"] >= 0.5)]
    print(f"{len(g)} games; |adj| >= 1 pt: {int((g['adj'].abs() >= 1).sum())}; pick flips vs model_v2: {len(flips)}")
    print(g.loc[g["adj"].abs() >= 1.5, ["away_team", "home_team", "pred", "adj", "adj_pr34", "p_v2", "p_dc", "p_pr34"]]
          .to_string(index=False))
    if len(flips):
        print("flips:\n" + flips[["away_team", "home_team", "pred", "adj", "p_v2", "p_dc"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
