"""2026 games where a team's regular starters were missing, used less than usual, or listed
injured, how teammates absorbed their share, and how model_v2 did in those games.

    python scripts/starter_response.py [--data-dir data] [--preds DIR ...] [--out DIR]

Usage comes from CFBD /plays/stats (passer, rusher, target rows) and the parsed play text
(defensive credits: tackles, sacks, hurries, breakups, interceptions), on plays outside garbage
time. A team's regulars are set from its last 4 earlier 2026 games (at least 2):

    QB1   most dropbacks; usual share of team dropbacks in games he played >= 70%
    RB1   most carries among RB/FB; usual share of team carries >= 30%
    WR1, WR2   most targets among WR; usual target share >= 15% / >= 10%
    TE1   most targets among TE; usual target share >= 8%
    DEF   five defenders with the most credits; usual share of team credits >= 6%

In a game a regular is "out" with no stat line at all (any play, garbage time included) and
"reduced" when his non-garbage share is under 40% of his usual share. "Listed" = Out or
Questionable on the conference availability report's last status before kickoff (ACC, Big Ten,
Big 12 and SEC conference games only; HDI archive pulled 2026-11-18, matched on team, jersey and
last name). In-game usage is known only after the game, so it sorts games and explains misses;
it is not a pregame input. The availability report is pregame.

Response: for each affected regular, the teammates at the same position who gained share
(change against their own usual share), and the team's non-garbage offensive EPA per play and
success against its own earlier games, adjusted by the opponent's earlier EPA and success allowed.

Model: margin residual (home margin minus pred_margin) and log loss of each --preds folder's
model_v2 predictions (default data/predictions/model_v2), on affected and other games.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = 2026
PASSER = ("Completion", "Incompletion", "Sack Taken", "Interception Thrown")
TARGET = ("Reception", "Target")
DEF_ROLES = ("tackler", "sacker", "hurry", "breakup", "interceptor", "forced_fumble")
ROLE_RULES = {"QB1": 0.70, "RB1": 0.30, "WR1": 0.15, "WR2": 0.10, "TE1": 0.08}
DEF_N, DEF_MIN = 5, 0.06
REDUCED = 0.40
WINDOW = 4          # regulars come from the team's last 4 earlier games
HDI_TEAM_FIX = {"Miami (FL)": "Miami", "Southern California": "USC", "Pitt": "Pittsburgh",
                "NC State": "NC State", "Ole Miss": "Ole Miss"}


def load(data: Path):
    pbp = pd.read_parquet(data / "core" / "pbp.parquet", columns=[
        "play_id", "game_id", "season", "week", "offense_id", "defense_id", "garbage_time", "ppa", "success",
        "is_scrimmage"])
    pbp = pbp[pbp["season"] == SEASON]
    files = [f for f in sorted(glob.glob(str(data / "tables" / "plays_stats" / "*.parquet"))) if re.search(rf"[/\\]{SEASON}_", f)]
    st = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    st["play_id"] = st["playId"].astype(str)
    st["athlete_id"] = pd.to_numeric(st["athleteId"], errors="coerce").astype("Int64")
    st = st.drop_duplicates(["play_id", "athlete_id", "statType"])
    teams = pd.read_parquet(data / "core" / "teams.parquet")
    teams = teams[teams["season"] == SEASON]
    tid = dict(zip(teams["team"], teams["team_id"]))
    st["team_id"] = st["team"].map(tid)
    st = st.merge(pbp[["play_id", "game_id", "offense_id", "defense_id", "garbage_time", "ppa", "success"]], on="play_id")
    pp = pd.read_parquet(data / "core" / "play_players.parquet")
    pp = pp[(pp["season"] == SEASON) & pp["role"].isin(DEF_ROLES) & (pp["side"] == "defense") & pp["athlete_id"].notna()]
    pp = pp.merge(pbp[["play_id", "garbage_time"]], on="play_id")
    roster = pd.read_parquet(data / "tables" / "roster" / f"{SEASON}.parquet")
    roster["athlete_id"] = pd.to_numeric(roster["id"], errors="coerce").astype("Int64")
    roster["player"] = (roster["firstName"].fillna("") + " " + roster["lastName"].fillna("")).str.strip()
    gi = pd.read_parquet(data / "core" / "game_index.parquet")
    gi = gi[gi["season"] == SEASON]
    return pbp, st, pp, roster, gi, teams


def usage(st, pp, roster, gi):
    """Per (game, team, athlete): non-garbage counts and any-play presence."""
    pos = roster.drop_duplicates("athlete_id").set_index("athlete_id")["position"]
    ng = st[~st["garbage_time"].fillna(False).astype(bool)]
    def count(df, types, name):
        return df[df["statType"].isin(types)].groupby(["game_id", "team_id", "athlete_id"]).size().rename(name)
    u = pd.concat([count(ng, PASSER, "drop"), count(ng, ("Rush",), "carry"), count(ng, TARGET, "tgt")], axis=1).fillna(0)
    d = (pp[~pp["garbage_time"].fillna(False).astype(bool)].groupby(["game_id", "team_id", "athlete_id"]).size()
         .rename("dcred"))
    u = u.join(d, how="outer").fillna(0).reset_index()
    present = pd.concat([st[["game_id", "team_id", "athlete_id"]], pp[["game_id", "team_id", "athlete_id"]]]).drop_duplicates()
    present["present"] = True
    u = u.merge(present, on=["game_id", "team_id", "athlete_id"], how="outer")
    u["present"] = u["present"].fillna(False)
    for c in ("drop", "carry", "tgt", "dcred"):
        u[c] = u[c].fillna(0)
    u["pos"] = u["athlete_id"].map(pos)
    tot = u.groupby(["game_id", "team_id"])[["drop", "carry", "tgt", "dcred"]].transform("sum")
    for c in ("drop", "carry", "tgt", "dcred"):
        u[f"s_{c}"] = np.where(tot[c] > 0, u[c] / tot[c].where(tot[c] > 0, 1), np.nan)
    u = u.merge(gi[["game_id", "start_time_utc", "week"]], on="game_id")
    return u


ROLE_STAT = {"QB1": ("drop", None), "RB1": ("carry", ("RB", "FB")), "WR1": ("tgt", ("WR",)),
             "WR2": ("tgt", ("WR",)), "TE1": ("tgt", ("TE",))}


def regulars(u: pd.DataFrame, team: int, before) -> list[dict]:
    """Regulars of `team` from its games before `before`."""
    prev = u[(u["team_id"] == team) & (u["start_time_utc"] < before)]
    if prev["game_id"].nunique() < 2:
        return []
    # the team's last WINDOW games, so a starter who lost the job stops counting as missing
    last = prev.drop_duplicates("game_id").nlargest(WINDOW, "start_time_utc")["game_id"]
    prev = prev[prev["game_id"].isin(last)]
    out = []
    for role, (stat, posset) in ROLE_STAT.items():
        cand = prev if posset is None else prev[prev["pos"].isin(posset)]
        tot = cand.groupby("athlete_id")[stat].sum().sort_values(ascending=False)
        rank = 1 if role != "WR2" else 2
        if len(tot) < rank or tot.iloc[rank - 1] <= 0:
            continue
        aid = tot.index[rank - 1]
        played = prev[(prev["athlete_id"] == aid) & prev["present"]]
        share = played[f"s_{stat}"].mean()
        if share >= ROLE_RULES[role] and len(played) >= 2:
            out.append({"role": role, "athlete_id": aid, "stat": stat, "usual": float(share), "pos": posset})
    dtot = prev.groupby("athlete_id")["dcred"].sum().sort_values(ascending=False).head(DEF_N)
    for aid in dtot.index:
        played = prev[(prev["athlete_id"] == aid) & prev["present"]]
        share = played["s_dcred"].mean()
        if share >= DEF_MIN and len(played) >= 2:
            out.append({"role": "DEF", "athlete_id": aid, "stat": "dcred", "usual": float(share), "pos": None})
    return out


def hdi_status(roster: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Last pregame availability status per (team_id, week, athlete_id)."""
    rows = []
    for f in glob.glob("/mnt/project-files/injury-reports/hdi_raw_2026-11-18/arch_*.json.gz"):
        d = json.load(gzip.open(f))
        rows += d.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["team_id", "week", "athlete_id", "listed"])
    h = pd.DataFrame(rows)
    def last(r):
        for c in ("Game Day", "Update 2", "Update 1", "Initial"):
            v = r.get(c)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    h["status"] = h.apply(last, axis=1)
    h["team"] = h["Team"].map(lambda t: HDI_TEAM_FIX.get(t, re.sub(r" St\.$", " State", t)))
    tid = dict(zip(teams["team"], teams["team_id"]))
    h["team_id"] = h["team"].map(tid)
    h["jersey"] = pd.to_numeric(h["Number"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    h["last"] = h["Player"].astype(str).str.split().str[-1].str.lower()
    r = roster.assign(last=roster["lastName"].astype(str).str.split().str[-1].str.lower(),
                      team_id=roster["team"].map(tid))
    m = h.merge(r[["team_id", "jersey", "last", "athlete_id"]], on=["team_id", "jersey", "last"], how="left")
    m = m.rename(columns={"WeekNum": "week"})
    m["listed"] = m["status"].isin(["Out", "Questionable", "Doubtful"])
    m = m[m["athlete_id"].notna()]
    return m[["team_id", "week", "athlete_id", "status", "listed"]].drop_duplicates(["team_id", "week", "athlete_id"])


def team_eff(pbp: pd.DataFrame, gi: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["is_scrimmage"] & ~pbp["garbage_time"].fillna(False).astype(bool)]
    o = p.groupby(["game_id", "offense_id"]).agg(epa=("ppa", "mean"), sr=("success", "mean"), n=("ppa", "size")).reset_index()
    o = o.merge(gi[["game_id", "start_time_utc"]], on="game_id")
    o = o.merge(p.groupby(["game_id", "offense_id"])["defense_id"].first().reset_index(), on=["game_id", "offense_id"])
    o["sr"] = o["sr"].astype(float)
    lg_epa, lg_sr = o["epa"].mean(), o["sr"].mean()
    rows = []
    for r in o.itertuples(index=False):
        prev_o = o[(o["offense_id"] == r.offense_id) & (o["start_time_utc"] < r.start_time_utc)]
        prev_d = o[(o["defense_id"] == r.defense_id) & (o["start_time_utc"] < r.start_time_utc)]
        if len(prev_o) < 2 or len(prev_d) < 2:
            continue
        exp_epa = np.average(prev_o["epa"], weights=prev_o["n"]) + np.average(prev_d["epa"], weights=prev_d["n"]) - lg_epa
        exp_sr = np.average(prev_o["sr"], weights=prev_o["n"]) + np.average(prev_d["sr"], weights=prev_d["n"]) - lg_sr
        rows.append({"game_id": r.game_id, "team_id": r.offense_id, "d_epa": r.epa - exp_epa, "d_sr": r.sr - exp_sr})
    return pd.DataFrame(rows)


def load_preds(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "*.parquet")))
    p = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    p = p.merge(pd.Series(dtype=object, name="x"), how="cross") if False else p
    return p.sort_values("cutoff_utc").drop_duplicates("game_id", keep="last")[["game_id", "p_home_win", "pred_margin"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--preds", nargs="*", default=None, help="label=folder of model_v2 prediction parquet files")
    ap.add_argument("--out", default=None, help="folder for the CSV outputs")
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    pbp, st, pp, roster, gi, teams = load(data)
    u = usage(st, pp, roster, gi)
    names = roster.drop_duplicates("athlete_id").set_index("athlete_id")["player"]
    tname = dict(zip(teams["team_id"], teams["team"]))
    sec = set(teams.loc[teams["conference"] == "SEC", "team_id"])
    fbs = set(teams.loc[teams["fbs"].astype(bool), "team_id"])
    hdi = hdi_status(roster, teams)
    hdi_idx = {(int(r.team_id), int(r.week), int(r.athlete_id)): r.status for r in hdi.itertuples() if r.listed}
    done = gi[gi["completed"].astype(bool) & gi["home_fbs"].astype(bool) & gi["away_fbs"].astype(bool)]

    flags, resp = [], []
    for g in done.itertuples(index=False):
        for team in (g.home_id, g.away_id):
            regs = regulars(u, team, g.start_time_utc)
            cur = u[(u["game_id"] == g.game_id) & (u["team_id"] == team)].set_index("athlete_id")
            if cur.empty:
                continue
            for rg in regs:
                aid = rg["athlete_id"]
                present = aid in cur.index and bool(cur.loc[aid, "present"])
                share = float(cur.loc[aid, f"s_{rg['stat']}"]) if present and pd.notna(cur.loc[aid, f"s_{rg['stat']}"]) else 0.0
                listed = hdi_idx.get((int(team), int(g.week), int(aid)))
                status = "out" if not present else ("reduced" if share < REDUCED * rg["usual"] else "normal")
                if status == "normal" and not listed:
                    continue
                flags.append({"game_id": g.game_id, "week": g.week, "team_id": team, "team": tname.get(team),
                              "role": rg["role"], "player": names.get(aid, str(aid)), "athlete_id": aid,
                              "usual": round(rg["usual"], 3), "share": round(share, 3), "status": status,
                              "listed": listed})
                if status == "normal" or rg["role"] == "DEF":
                    continue
                # who absorbed the share: same position group, change vs own usual share this season
                stat = rg["stat"]
                prev = u[(u["team_id"] == team) & (u["start_time_utc"] < g.start_time_utc)]
                mates = cur.drop(index=aid, errors="ignore")
                if rg["pos"] is not None:
                    mates = mates[mates["pos"].isin(rg["pos"])]
                elif rg["role"] == "QB1":
                    mates = mates[mates["drop"] > 0]
                usual_m = prev.groupby("athlete_id")[f"s_{stat}"].mean()
                gain = (mates[f"s_{stat}"].fillna(0) - usual_m.reindex(mates.index).fillna(0)).sort_values(ascending=False)
                top = gain.index[0] if len(gain) else None
                if top is not None:
                    plays = st[(st["game_id"] == g.game_id) & (st["athlete_id"] == top)
                               & ~st["garbage_time"].fillna(False).astype(bool)]
                    plays = plays[plays["statType"].isin(PASSER if stat == "drop" else (("Rush",) if stat == "carry" else TARGET))]
                    prevp = st[(st["athlete_id"] == aid) & st["game_id"].isin(prev["game_id"])
                               & ~st["garbage_time"].fillna(False).astype(bool)]
                    prevp = prevp[prevp["statType"].isin(PASSER if stat == "drop" else (("Rush",) if stat == "carry" else TARGET))]
                    resp.append({"game_id": g.game_id, "team_id": team, "team": tname.get(team), "role": rg["role"],
                                 "status": status, "starter": names.get(aid, str(aid)),
                                 "starter_usual_share": rg["usual"], "starter_share": share,
                                 "starter_usual_epa": prevp["ppa"].mean(),
                                 "replacement": names.get(top, str(top)), "repl_share": float(mates.loc[top, f"s_{stat}"]),
                                 "repl_usual_share": float(usual_m.get(top, 0.0)), "repl_gain": float(gain.iloc[0]),
                                 "repl_epa": plays["ppa"].mean(), "repl_success": plays["success"].astype(float).mean(),
                                 "repl_n": len(plays), "others_gain": float(gain.iloc[1:].clip(lower=0).sum())})
    fl = pd.DataFrame(flags)
    rs = pd.DataFrame(resp)
    eff = team_eff(pbp, gi)

    tg = fl.groupby(["game_id", "team_id"]).agg(
        qb=("role", lambda r: bool(((r == "QB1") & fl.loc[r.index, "status"].isin(["out", "reduced"])).any())),
        n_off=("role", lambda r: int(((r != "DEF") & fl.loc[r.index, "status"].isin(["out", "reduced"])).sum())),
        n_def=("role", lambda r: int(((r == "DEF") & fl.loc[r.index, "status"].isin(["out", "reduced"])).sum())),
        listed=("listed", lambda s: int(s.notna().sum()))).reset_index()
    tg["affected"] = tg["qb"] | (tg["n_off"] >= 2) | (tg["listed"] > 0)
    tg["kind"] = np.select([tg["qb"], tg["n_off"] >= 2, tg["listed"] > 0], ["QB1 out or reduced", "2+ offensive regulars",
                           "listed on availability report"], default="other")
    tg = tg.merge(eff, on=["game_id", "team_id"], how="left")

    games = done[["game_id", "week", "home_id", "away_id", "home_points", "away_points"]].copy()
    games["margin"] = games["home_points"] - games["away_points"]
    aff = tg[tg["affected"]]
    games["affected"] = games["game_id"].isin(aff["game_id"])
    games["sec"] = games["home_id"].isin(sec) | games["away_id"].isin(sec)

    print(f"2026 FBS vs FBS games {len(games)}; with a starter-affected team {int(games.affected.sum())} "
          f"(SEC games {int((games.affected & games.sec).sum())} of {int(games.sec.sum())})")
    print("\nAffected team-games by kind (first matching):")
    print(aff["kind"].value_counts().to_string())
    print("\nFlags by role and status (team-games):")
    print(pd.crosstab(fl["role"], fl["status"]).to_string())
    print(f"availability-report listings matched to a regular: {int(fl['listed'].notna().sum())}")

    # team-level effect: offense EPA and success vs expectation, margin vs model (team's side)
    base = Path(a.preds[0].split("=", 1)[1]) if a.preds else data / "predictions" / "model_v2"
    pr = load_preds(base)
    tgm = tg.merge(games[["game_id", "home_id", "margin"]], on="game_id").merge(pr, on="game_id", how="left")
    side = np.where(tgm["team_id"] == tgm["home_id"], 1, -1)
    tgm["resid_team"] = side * (tgm["margin"] - tgm["pred_margin"])
    print("\nTeam result in affected games (team's point of view): margin minus model_v2 pred_margin, "
          "offense EPA/play and success vs expectation (own earlier games + opponent's earlier defense)")
    for kind, s in [("QB1 out or reduced", tgm[tgm.kind == "QB1 out or reduced"]),
                    ("2+ offensive regulars", tgm[tgm.kind == "2+ offensive regulars"]),
                    ("listed on availability report", tgm[tgm.kind == "listed on availability report"]),
                    ("not affected", tgm[~tgm.affected])]:
        r = s["resid_team"].dropna()
        se = r.std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
        print(f"  {kind:<31} n {len(r):>4}  margin vs model {r.mean():+6.2f} (SE {se:.2f})  "
              f"off EPA {s['d_epa'].mean():+.3f}  success {s['d_sr'].mean():+.3f}")

    if len(rs):
        print("\nWho absorbed the share (affected regulars, out or reduced; replacement = same-position teammate "
              "with the largest share gain over his own usual share)")
        g2 = rs.groupby(["role", "status"]).agg(
            n=("game_id", "size"), starter_usual=("starter_usual_share", "mean"), starter_share=("starter_share", "mean"),
            repl_usual=("repl_usual_share", "mean"), repl_share=("repl_share", "mean"), others_gain=("others_gain", "mean"),
            starter_usual_epa=("starter_usual_epa", "mean"), repl_epa=("repl_epa", "mean"), repl_success=("repl_success", "mean"))
        print(g2.round(3).to_string())

    # model comparison across prediction folders
    labels = a.preds or [f"model_v2={base}"]
    res = []
    for lab in labels:
        name, folder = lab.split("=", 1)
        p = load_preds(Path(folder))
        m = games.merge(p, on="game_id")
        y = (m["margin"] > 0).astype(float)
        q = m["p_home_win"].clip(1e-6, 1 - 1e-6)
        m["ll"] = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        m["br"] = (q - y) ** 2
        m["ae"] = (m["margin"] - m["pred_margin"]).abs()
        m["label"] = name
        res.append(m[["game_id", "week", "affected", "sec", "label", "ll", "br", "ae"]])
    res = pd.concat(res)
    print("\nmodel_v2 by prediction set (log loss / Brier / margin MAE)")
    ref = labels[0].split("=", 1)[0]
    for sub_name, mask in [("starter-affected games", res.affected), ("SEC starter-affected", res.affected & res.sec),
                           ("other games", ~res.affected), ("all 2026 games", res.affected | ~res.affected)]:
        s = res[mask]
        line = f"  {sub_name:<22}"
        r0 = s[s.label == ref].set_index("game_id")
        for lab in labels:
            name = lab.split("=", 1)[0]
            t = s[s.label == name].set_index("game_id")
            line += f" | {name}: n {len(t)} {t.ll.mean():.4f} / {t.br.mean():.4f} / {t.ae.mean():.2f}"
            if name != ref:
                j = t.join(r0, rsuffix="_0", how="inner")
                dw = j.assign(d=j.ll - j.ll_0).groupby("week")["d"].sum() / len(j)
                se = np.sqrt(len(dw) * ((dw - dw.mean()) ** 2).sum() / max(len(dw) - 1, 1))
                line += f" (ll diff {(j.ll - j.ll_0).mean():+.4f}, SE {se:.4f}; MAE diff {(j.ae - j.ae_0).mean():+.3f})"
        print(line)

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fl.to_csv(out / "starter_flags_2026.csv", index=False)
        rs.to_csv(out / "starter_replacements_2026.csv", index=False)
        tgm.to_csv(out / "starter_team_games_2026.csv", index=False)
        sec_rows = rs[rs["team_id"].isin(sec)].sort_values(["role", "team"])
        print(f"\nSEC replacements ({len(sec_rows)} rows):")
        print(sec_rows[["team", "role", "status", "starter", "starter_usual_share", "starter_share", "replacement",
                        "repl_usual_share", "repl_share", "repl_epa", "repl_n"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
