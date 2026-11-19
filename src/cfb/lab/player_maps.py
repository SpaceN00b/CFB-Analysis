"""Player location and play-style maps from parsed play text (conference test bench).

Experimental: no feature group or model reads this. It works on the detailed stat-crew text
that `cfb.core.playtext` parses (every game from 2025 week 9 on), so the display season is
2026 and the league-wide fits use 2025 (week 9 on) and 2026.

Four parts, each returned as plain records for the viewer:

- defense: where each defender is credited with tackles, sacks, hurries, breakups and
  interceptions (side from the play's run or pass direction, depth from the yards at the tackle
  or the throw), an approximate primary location, and a role model (softmax regression on that
  location profile, trained on every other FBS defender, tested on the chosen conference).
- offense: where each ball carrier and target goes (direction x depth) and which play types are
  called for him (formation, down, red zone, run direction, screen/short/intermediate/deep).
- rb: running backs' success by direction and formation, against the FBS average, and by
  opponent against an expectation from the back's own baseline and the opponent's run defense
  in that direction.
- receiving: no source marks drops, so the proxy is a clean incompletion (incomplete with no
  breakup or interception credited). A logistic model with throw depth/direction, pressure,
  down, and ridge effects for passer, receiver and defense splits each team's excess clean
  incompletions between its QB and its receivers.

Directions are the offense's: "left" is the offense's left, which is the defense's right.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.features.strength.playtext import fit_clean_inc  # noqa: F401  (shared with the str_pass_ci group)

SCRIMMAGE_ACTIONS = ("rush", "pass", "sack")
DIRS = ("left", "middle", "right")

# Defender event depth (yards past the line of scrimmage, offense's direction).
DEF_DEPTHS = (("Backfield", -99, -1), ("Line 0-2", 0, 2), ("Short 3-7", 3, 7),
              ("Mid 8-15", 8, 15), ("Deep 16+", 16, 999))
# Target depth by air yards.
TGT_DEPTHS = (("Behind LOS", -99, 0), ("Short 1-9", 1, 9), ("Inter 10-19", 10, 19),
              ("Deep 20+", 20, 999))
# Rush gain.
RUSH_GAINS = (("Loss/0", -99, 0), ("1-3", 1, 3), ("4-9", 4, 9), ("10+", 10, 999))
# Air-yard bins for the incompletion model's baseline.
AIR_BINS = (-99, 0, 5, 10, 15, 20, 30, 999)

ROLE_NAMES = {"IDL": "Interior DL", "EDGE": "Edge", "LB": "Off-ball LB", "CB": "Cornerback",
              "S": "Safety", "DL": "Defensive line", "DB": "Defensive back"}
FINE_POS = {"DT": "IDL", "NT": "IDL", "DE": "EDGE", "EDGE": "EDGE", "LB": "LB", "ILB": "LB",
            "MLB": "LB", "CB": "CB", "S": "S", "SAF": "S"}
COARSE_POS = {"DL": "DL", "DT": "DL", "NT": "DL", "DE": "DL", "EDGE": "DL", "LB": "LB",
              "ILB": "LB", "MLB": "LB", "OLB": "LB", "DB": "DB", "CB": "DB", "S": "DB", "SAF": "DB"}
FINE_TO_COARSE = {"IDL": "DL", "EDGE": "DL", "LB": "LB", "CB": "DB", "S": "DB", "NB": "DB"}
ROLE_NAMES["NB"] = "Nickel"
ALIGN_NAMES = {"X": "WR-X (split end)", "Z": "WR-Z (flanker)", "SLOT": "Slot WR", "TE": "Tight end",
               "RB": "Running back", "QB": "Quarterback"}
# Team depth chart labels (cfb.depth_charts) -> roles. DE is an edge in a four-man front and an
# interior end when the chart also lists a stand-up edge (JACK, OLB, ...), i.e. a three-man front.
CHART_IDL = {"DT", "NT", "LDT", "RDT", "DL", "NG", "N", "T", "3T", "TECH"}
CHART_DE = {"DE", "LDE", "RDE", "END", "E"}
CHART_NB = {"NB", "STAR", "SPUR", "NICKEL", "NICK", "CASH", "MONEY", "NKL", "N/S"}
CHART_CB = {"LCB", "RCB", "CB", "FCB", "BCB", "BC", "FC", "C"}
CHART_S = {"SS", "FS", "BS", "S", "FSS", "SAF", "BOS"}
CHART_ALIGN = {"WR-X": "X", "WR-Z": "Z", "WR-SL": "SLOT", "WR-H": "SLOT", "WR-F": "SLOT", "WR-A": "SLOT",
               "WR-Y": "SLOT", "TE": "TE", "TE-Y": "TE", "TE-H": "TE", "RB": "RB", "FB": "RB", "QB": "QB"}
SR_MARKET_FIX = {"Miami (FL)": "Miami", "UMass": "Massachusetts", "San Jose State": "San José State",
                 "Appalachian State": "App State", "Louisiana-Monroe": "UL Monroe"}

P4_CONFERENCES = ("ACC", "Big 12", "Big Ten", "SEC")
P4_EXTRA = ("Notre Dame",)
INJURY_FILE = "raw/injuries/cfbdepth-player-updates.csv"   # under the data dir
MIN_DEF_EVENTS = 15      # defender events for a location profile and the role model
MIN_TARGETS = 15         # targets for a receiver map
MIN_RUSHES = 30          # rushes for an RB map
MIN_QB_ATT = 60          # attempts for a team QB row
MIN_REC_TGT = 20         # targets for a team receiver row
EFFECT_Z = 1.96          # passer / receiver effect flag (z of the ridge estimate)
CELL_P = 0.01            # receiver play-type cell flag (one-sided binomial p)
CELL_K = 10.0            # teammates' cell rate shrunk toward the FBS cell rate with this many throws


def _bin(values: pd.Series, bins) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    v = pd.to_numeric(values, errors="coerce")
    for name, lo, hi in bins:
        out[(v >= lo) & (v <= hi)] = name
    return out


def _r(x, nd=3):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), nd)


# ----------------------------------------------------------------------------------------------
# Loading


def load(data_dir: str | Path = "data", seasons=(2025, 2026)) -> dict[str, pd.DataFrame]:
    data = Path(data_dir)
    core = data / "core"
    detail = pd.read_parquet(core / "play_detail.parquet")
    detail = detail[detail["season"].isin(seasons) & (detail["text_format"] == "detailed")]
    pbp = pd.read_parquet(core / "pbp.parquet", columns=[
        "play_id", "offense_id", "defense_id", "period", "down", "distance", "yards_to_goal",
        "yards_gained", "success", "is_scrimmage", "garbage_time", "play_type"])
    plays = detail.merge(pbp, on="play_id", how="inner")
    plays = plays[plays["action"].isin(SCRIMMAGE_ACTIONS) & plays["is_scrimmage"]
                  & ~plays["no_play"] & ~plays["kneel"] & ~plays["spike"]].reset_index(drop=True)
    players = pd.read_parquet(core / "play_players.parquet")
    players = players[players["play_id"].isin(plays["play_id"]) & players["athlete_id"].notna()]
    teams = pd.read_parquet(core / "teams.parquet")
    games = pd.read_parquet(core / "game_index.parquet")
    roster = pd.concat([pd.read_parquet(data / "tables" / "roster" / f"{s}.parquet") for s in seasons])
    roster = roster.assign(athlete_id=pd.to_numeric(roster["id"], errors="coerce").astype("Int64"))
    roster = roster.dropna(subset=["athlete_id"])
    sr = _sportradar_positions(data, seasons)
    return {"plays": plays, "players": players, "teams": teams, "games": games, "roster": roster,
            "sr_pos": sr, "chart": depth_chart_roles(data), "injuries": load_injuries(data / INJURY_FILE)}


def _norm_name(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def load_injuries(path: Path) -> pd.DataFrame:
    """CFBDepth player injury list (Name, Team, Pos, Status, Last Update, Update), the larger
    injury source (all FBS, non-conference games). Empty when the file is missing."""
    cols = ["team", "key", "injury_status", "injury_date", "injury_note"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    u = pd.read_csv(path)
    u = u.assign(team=u["Team"], key=u["Name"].map(_norm_name), injury_status=u["Status"],
                 injury_date=u["Last Update"], injury_note=u["Update"])
    return u.drop_duplicates(["team", "key"])[cols]


def chart_role(position: str, pos_group: str, three_man: bool):
    """Role (IDL, EDGE, LB, NB, CB, S) and side (L, R or None) for a defensive chart label."""
    pos = str(position).upper()
    side = pos[0] if len(pos) > 2 and pos[0] in "LR" and pos[1:] in {"DE", "DT", "CB", "OLB", "ILB"} else None
    if pos in CHART_IDL:
        return "IDL", side
    if pos in CHART_DE:
        return ("IDL" if three_man else "EDGE"), side
    if pos_group == "EDGE":
        return "EDGE", side
    if pos in CHART_NB:
        return "NB", None
    if pos in CHART_CB:
        return "CB", side
    if pos in CHART_S or pos_group == "DB":
        return "S", None
    if pos_group == "LB":
        return "LB", side
    if pos_group == "DL":
        return "IDL", side
    return None, None


def depth_chart_roles(data: Path) -> pd.DataFrame:
    """One row per charted athlete (latest download): chart label, depth, defensive role and side,
    offensive alignment (X, Z, SLOT, TE, RB, QB). Empty when the depth_charts table is missing."""
    files = glob.glob(str(data / "tables" / "depth_charts" / "*.parquet"))
    cols = ["season", "athlete_id", "chart_team", "chart_pos", "chart_depth", "chart_role", "chart_side",
            "chart_align", "chart_asof"]
    if not files:
        return pd.DataFrame(columns=cols)
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d["unit"].isin(["offense", "defense"]) & d["athlete_id"].notna()]
    d = d[d["asof_utc"] == d.groupby("team")["asof_utc"].transform("max")]
    edge_teams = set(d.loc[(d["unit"] == "defense") & (d["pos_group"] == "EDGE") & (d["depth"] == 1), "team"])
    roles = [chart_role(p, g, t in edge_teams) if u == "defense" else (None, None)
             for p, g, t, u in zip(d["position"], d["pos_group"], d["team"], d["unit"])]
    d = d.assign(chart_role=[r for r, _ in roles], chart_side=[sd for _, sd in roles],
                 chart_align=d["position"].astype(str).str.upper().map(CHART_ALIGN).where(d["unit"] == "offense"))
    d = d.sort_values(["athlete_id", "depth"], na_position="last").drop_duplicates("athlete_id")
    d = d.rename(columns={"team": "chart_team", "position": "chart_pos", "depth": "chart_depth",
                          "asof_utc": "chart_asof"})
    d["athlete_id"] = d["athlete_id"].astype("int64")
    return d[cols]


def _sportradar_positions(data: Path, seasons) -> pd.DataFrame:
    """Most common Sportradar position per (season, CFBD team name, jersey, last name)."""
    files = [f for s in seasons for f in glob.glob(str(data / "tables" / "sr_play_stats" / f"{s}_*.parquet"))]
    hier = glob.glob(str(data / "tables" / "sr_hierarchy" / "*.parquet"))
    if not files or not hier:
        return pd.DataFrame(columns=["season", "team", "jersey", "last", "sr_position"])
    h = pd.read_parquet(hier[0])
    h = h[h["division"] == "FBS"]
    alias_team = {a: SR_MARKET_FIX.get(m, m) for a, m in zip(h["alias"], h["market"])}
    cols = ["season", "team_alias", "player_jersey", "player_name", "player_position"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    s = s.dropna(subset=["player_position", "player_jersey", "player_name"])
    s = s.assign(team=s["team_alias"].map(alias_team),
                 jersey=pd.to_numeric(s["player_jersey"], errors="coerce"),
                 last=s["player_name"].str.split().str[-1].str.lower())
    s = s.dropna(subset=["team", "jersey"])
    g = (s.groupby(["season", "team", "jersey", "last", "player_position"]).size()
         .rename("n").reset_index().sort_values("n", ascending=False)
         .drop_duplicates(["season", "team", "jersey", "last"]))
    return g.rename(columns={"player_position": "sr_position"})[["season", "team", "jersey", "last", "sr_position"]]


def player_info(b: dict) -> pd.DataFrame:
    """Per (season, athlete): name, CFBD position, Sportradar position, role labels."""
    r = b["roster"][["season", "athlete_id", "firstName", "lastName", "team", "jersey", "position"]].copy()
    r["last"] = r["lastName"].astype("string").str.split().str[-1].str.lower()
    r = r.merge(b["sr_pos"], on=["season", "team", "jersey", "last"], how="left")
    r["player"] = (r["firstName"].fillna("") + " " + r["lastName"].fillna("")).str.strip()
    inj = b.get("injuries")
    if inj is not None and len(inj):
        r["key"] = r["player"].map(_norm_name)
        r = r.merge(inj, on=["team", "key"], how="left")
        cur = (r["season"] == r["season"].max()).to_numpy()   # the list is current: latest season only
        r.loc[~cur, ["injury_status", "injury_date", "injury_note"]] = None
    else:
        r["injury_status"] = r["injury_date"] = r["injury_note"] = None
    fine = r["position"].map(lambda p: FINE_POS.get(p) if p in ("DT", "NT", "DE", "EDGE", "CB", "S") else None)
    sr_fine = r["sr_position"].map(lambda p: FINE_POS.get(p) if isinstance(p, str) and p not in ("LB",) else None)
    generic_lb = r["position"].eq("LB") | r["sr_position"].isin(["LB", "MLB", "ILB"])
    fine = fine.where(fine.notna(), sr_fine)
    fine = fine.where(fine.notna(), np.where(generic_lb, "LB", None))
    r["role_fine"] = fine
    r["role_coarse"] = r["position"].map(COARSE_POS)
    r.loc[r["role_coarse"].isna(), "role_coarse"] = r["sr_position"].map(COARSE_POS)
    r = r.drop_duplicates(["season", "athlete_id"])[
        ["season", "athlete_id", "player", "position", "sr_position", "role_fine", "role_coarse",
         "injury_status", "injury_date", "injury_note"]]
    ch = b.get("chart")
    if ch is not None and len(ch):
        r = r.merge(ch.drop(columns=["chart_team", "chart_asof"]), on=["season", "athlete_id"], how="left")
    else:
        for c in ("chart_pos", "chart_depth", "chart_role", "chart_side", "chart_align"):
            r[c] = None
    return r


# ----------------------------------------------------------------------------------------------
# Defense


def defense_events(b: dict) -> pd.DataFrame:
    """One row per defender credit on a scrimmage play, with side and depth."""
    pl = b["plays"]
    d = b["players"]
    d = d[(d["side"] == "defense") & d["role"].isin(["tackler", "sacker", "hurry", "breakup", "interceptor"])]
    cols = ["play_id", "game_id", "season", "week", "offense_id", "defense_id", "action", "run_dir",
            "pass_dir", "completed", "text_yards", "air_yards", "n_tacklers", "garbage_time"]
    e = d[["play_id", "role", "athlete_id", "team_id"]].merge(pl[cols], on="play_id", how="inner")
    kind = np.select(
        [(e["role"] == "tackler") & (e["action"] == "rush"),
         (e["role"] == "tackler") & (e["action"] == "pass") & e["completed"].fillna(False),
         e["role"] == "sacker", e["role"] == "hurry", e["role"] == "breakup", e["role"] == "interceptor"],
        ["run_tackle", "pass_tackle", "sack", "hurry", "breakup", "int"], default="other")
    e["kind"] = kind
    e = e[e["kind"] != "other"].copy()
    lat = np.where(e["kind"] == "run_tackle", e["run_dir"],
                   np.where(e["kind"].isin(["pass_tackle", "breakup", "int"]), e["pass_dir"], None))
    e["lat"] = pd.Series(lat, index=e.index, dtype="string")
    depth = np.where(e["kind"].isin(["run_tackle", "pass_tackle"]), pd.to_numeric(e["text_yards"], errors="coerce"),
                     np.where(e["kind"].isin(["breakup", "int"]), pd.to_numeric(e["air_yards"], errors="coerce"),
                              np.where(e["kind"] == "sack", pd.to_numeric(e["text_yards"], errors="coerce"), np.nan)))
    e["depth"] = depth.astype(float)
    e.loc[(e["kind"] == "sack") & e["depth"].isna(), "depth"] = -6.0
    e.loc[e["kind"] == "hurry", "depth"] = -5.0
    e["air"] = pd.to_numeric(e["air_yards"], errors="coerce")
    e["zone_depth"] = _bin(e["depth"], DEF_DEPTHS)
    return e.reset_index(drop=True)


FEATURES = ["f_run", "run_depth_med", "run_tfl", "run_long", "pass_tkl_share", "pass_depth_med",
            "cov_rate", "rush_rate", "cov_air", "mid_run", "mid_pass", "side_skew", "shared",
            "depth_med"]


def defender_profiles(e: pd.DataFrame) -> pd.DataFrame:
    """Location features per (season, team, athlete)."""
    key = ["season", "team_id", "athlete_id"]
    rows = []
    for k, g in e.groupby(key, sort=False):
        run = g[g["kind"] == "run_tackle"]
        pas = g[g["kind"] == "pass_tackle"]
        cov = g[g["kind"].isin(["breakup", "int"])]
        tk = g[g["kind"].isin(["run_tackle", "pass_tackle"])]
        located = g[g["lat"].notna()]
        left = (located["lat"] == "left").sum()
        right = (located["lat"] == "right").sum()
        n = len(g)
        rec = dict(zip(key, k))
        rec.update(
            n_events=n, n_games=g["game_id"].nunique(),
            n_run_tkl=len(run), n_pass_tkl=len(pas), n_sack=(g["kind"] == "sack").sum(),
            n_hurry=(g["kind"] == "hurry").sum(), n_pbu=(g["kind"] == "breakup").sum(),
            n_int=(g["kind"] == "int").sum(),
            f_run=len(run) / len(tk) if len(tk) else np.nan,
            run_depth_med=run["depth"].median() if len(run) else np.nan,
            run_tfl=(run["depth"] <= 0).mean() if len(run) else np.nan,
            run_long=(run["depth"] >= 8).mean() if len(run) else np.nan,
            pass_tkl_share=len(pas) / n,
            pass_depth_med=pas["depth"].median() if len(pas) else np.nan,
            cov_rate=len(cov) / n,
            rush_rate=g["kind"].isin(["sack", "hurry"]).sum() / n,
            cov_air=pd.concat([cov["air"], pas["air"]]).mean() if len(cov) + len(pas) else np.nan,
            mid_run=(run["lat"] == "middle").mean() if len(run) else np.nan,
            mid_pass=(pd.concat([pas, cov])["lat"] == "middle").mean() if len(pas) + len(cov) else np.nan,
            side_skew=abs(left - right) / (left + right) if left + right else np.nan,
            shared=(tk["n_tacklers"] == 2).fillna(False).mean() if len(tk) else np.nan,
            depth_med=g["depth"].median(),
            left=int(left), middle=int((located["lat"] == "middle").sum()), right=int(right),
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def softmax_fit(X: np.ndarray, y: np.ndarray, k: int, l2: float = 1e-2, iters: int = 3000,
                lr: float = 0.5) -> np.ndarray:
    """Multinomial logistic regression by full-batch gradient descent. W includes the bias row."""
    n = len(X)
    Xb = np.hstack([X, np.ones((n, 1))])
    W = np.zeros((Xb.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(iters):
        Z = Xb @ W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = Xb.T @ (P - Y) / n
        G[:-1] += l2 * W[:-1]
        W -= lr * G
    return W


def softmax_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Z = np.hstack([X, np.ones((len(X), 1))]) @ W
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def role_model(prof: pd.DataFrame, label_col: str, train_mask: np.ndarray, test_mask: np.ndarray):
    """Fit on train rows with a label, evaluate on test rows with a label, predict all rows."""
    X = pd.DataFrame({c: pd.to_numeric(prof[c], errors="coerce").astype("Float64")
                      .to_numpy(dtype=float, na_value=np.nan) for c in FEATURES}, index=prof.index)
    med = X[train_mask].median()
    X = X.fillna(med)
    mu, sd = X[train_mask].mean(), X[train_mask].std().replace(0, 1)
    Xs = ((X - mu) / sd).to_numpy()
    labels = sorted(prof.loc[train_mask, label_col].dropna().unique())
    idx = {c: i for i, c in enumerate(labels)}
    tr = train_mask & prof[label_col].isin(labels).to_numpy()
    te = test_mask & prof[label_col].isin(labels).to_numpy()
    W = softmax_fit(Xs[tr], prof.loc[tr, label_col].map(idx).to_numpy(), len(labels))
    P = softmax_predict(W, Xs)
    pred = np.array(labels)[P.argmax(axis=1)]
    truth = prof[label_col].to_numpy()
    acc = float((pred[te] == truth[te]).mean()) if te.any() else None
    maj = prof.loc[tr, label_col].value_counts().idxmax()
    base = float((truth[te] == maj).mean()) if te.any() else None
    conf = pd.crosstab(pd.Series(truth[te], name="listed"), pd.Series(pred[te], name="predicted"))
    coef = pd.DataFrame(W[:-1], index=FEATURES, columns=labels)
    return {"labels": labels, "P": P, "pred": pred, "acc": acc, "baseline": base, "majority": maj,
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "confusion": conf, "coef": coef}


def role_model_folds(prof: pd.DataFrame, label_col: str, label_ok: np.ndarray, folds: list[np.ndarray],
                     disp: np.ndarray):
    """role_model once per fold: train on labeled rows outside the fold, test on the fold's
    display rows. Rows in a fold take that fold's prediction; accuracy and confusion pool the
    folds. With one fold this is role_model(train=~fold, test=fold)."""
    first = None
    P, pred = None, None
    truth = prof[label_col].to_numpy()
    tests, base_hits, per = [], 0, []
    for f in folds:
        r = role_model(prof, label_col, label_ok & ~f, disp & f & label_ok)
        if first is None:
            first = r
            P, pred = r["P"].copy(), r["pred"].copy()
        if r["labels"] != first["labels"]:
            raise ValueError("folds disagree on role labels")
        P[f], pred[f] = r["P"][f], r["pred"][f]
        te = disp & f & label_ok & prof[label_col].isin(first["labels"]).to_numpy()
        tests.append(te)
        base_hits += int((truth[te] == r["majority"]).sum())
        per.append({"n_test": r["n_test"], "acc": r["acc"]})
    te = np.logical_or.reduce(tests)
    acc = float((pred[te] == truth[te]).mean()) if te.any() else None
    conf = pd.crosstab(pd.Series(truth[te], name="listed"), pd.Series(pred[te], name="predicted"))
    full = role_model(prof, label_col, label_ok, label_ok)   # coefficients on every labeled row
    return {"labels": first["labels"], "P": P, "pred": pred, "acc": acc,
            "baseline": base_hits / te.sum() if te.any() else None, "n_train": first["n_train"],
            "n_test": int(te.sum()), "confusion": conf, "coef": full["coef"], "folds": per}


ALIGNMENT = {"IDL": "Interior line", "EDGE": "Edge (end of the line)", "LB": "Box linebacker depth",
             "CB": "Outside corner", "S": "Safety", "NB": "Slot / nickel"}


def side_lean(left: int, right: int, center: float = 0.5, width: float = 0.15, min_n: int = 20,
              defense: bool = True) -> str | None:
    """Side a player's plays lean to, when the left share is `width` away from the league's
    (`center`: stat crews chart more plays to the offense's right, about 45% left in 2026)."""
    lr = left + right
    if lr < min_n:
        return None
    share = left / lr
    if share >= center + width:
        return "offense's left (defense's right)" if defense else "offense's left"
    if share <= center - width:
        return "offense's right (defense's left)" if defense else "offense's right"
    return None


def primary_location(p: pd.Series, mid_avg: float, center: float) -> str:
    """Where a defender makes plays: depth level and lateral zone, relative to the FBS average."""
    if p["rush_rate"] >= 0.35:
        depth = "Backfield (pass rush)"
    elif p["depth_med"] <= 2:
        depth = "At the line (0-2 yds)"
    elif p["depth_med"] <= 6:
        depth = "Second level (3-6 yds)"
    elif p["depth_med"] <= 12:
        depth = "Intermediate (7-12 yds)"
    else:
        depth = "Deep (13+ yds)"
    tot = p["left"] + p["middle"] + p["right"]
    lean = side_lean(p["left"], p["right"], center)
    if tot < 10:
        side = "side unknown"
    elif p["middle"] / tot >= mid_avg + 0.07:
        side = "between the tackles"
    elif p["middle"] / tot <= mid_avg - 0.10:
        side = "outside"
    else:
        side = "sideline to sideline"
    if lean:
        side += f", leans {lean}"
    return f"{depth}, {side}"


def defense_section(b: dict, info: pd.DataFrame, season: int, conf_ids: set[int],
                    team_name: dict, folds: list[set[int]] | None = None) -> dict:
    e = defense_events(b)
    prof = defender_profiles(e)
    prof = prof.merge(info, on=["season", "athlete_id"], how="left")
    prof = prof[prof["n_events"] >= MIN_DEF_EVENTS].reset_index(drop=True)
    in_conf = prof["team_id"].isin(conf_ids).to_numpy()
    disp = in_conf & (prof["season"] == season).to_numpy()
    # Folds: the display teams split by conference; each fold is predicted by a model trained
    # on every other team (leave one conference out). One conference = one fold.
    fm = [prof["team_id"].isin(f).to_numpy() for f in (folds or [conf_ids])]
    cfbd = role_model_folds(prof, "role_fine", prof["role_fine"].notna().to_numpy(), fm, disp)
    coarse = role_model_folds(prof, "role_coarse", prof["role_coarse"].notna().to_numpy(), fm, disp)
    has_chart = prof["chart_role"].notna().to_numpy() if "chart_role" in prof else np.zeros(len(prof), bool)
    use_chart = min((has_chart & ~f).sum() for f in fm) >= 100
    # Chart labels are a 2026-11-19 snapshot of P4 + Notre Dame: train on the other charted teams,
    # test on the held-out conference. Without charts the CFBD / Sportradar labels are used.
    fine = role_model_folds(prof, "chart_role", has_chart, fm, disp) if use_chart else cfbd
    prof["pred_role"] = fine["pred"]
    prof["pred_cfbd"] = cfbd["pred"]
    prof["pred_p"] = fine["P"].max(axis=1)
    prof["pred_coarse"] = coarse["pred"]
    for i, c in enumerate(fine["labels"]):
        prof[f"p_{c}"] = fine["P"][:, i]

    loc = e[e["lat"].notna()]
    mid_avg = float((loc["lat"] == "middle").mean())
    center = float((loc["lat"] == "left").sum() / loc["lat"].isin(["left", "right"]).sum())
    ed = e[e["season"] == season]
    out = []
    for _, p in prof[disp].iterrows():
        g = ed[(ed["athlete_id"] == p["athlete_id"]) & (ed["team_id"] == p["team_id"])]
        grid = {}
        for zd, _, _ in DEF_DEPTHS:
            for lt in DIRS:
                grid[f"{zd}|{lt}"] = int(((g["zone_depth"] == zd) & (g["lat"] == lt)).sum())
        listed = p["role_fine"] if isinstance(p["role_fine"], str) else p["role_coarse"]
        chart = p.get("chart_role")
        if use_chart:
            listed = chart if isinstance(chart, str) else None
            mismatch = isinstance(chart, str) and chart != p["pred_role"]
        elif isinstance(p["role_fine"], str):
            mismatch = p["role_fine"] != p["pred_role"]
        else:
            mismatch = isinstance(p["role_coarse"], str) and FINE_TO_COARSE[p["pred_role"]] != p["role_coarse"]
        out.append({
            "team": team_name.get(int(p["team_id"])), "athlete_id": int(p["athlete_id"]),
            "player": p["player"] if isinstance(p["player"], str) else str(p["athlete_id"]),
            "pos": p["position"] if isinstance(p["position"], str) else None,
            "sr_pos": p["sr_position"] if isinstance(p["sr_position"], str) else None,
            "listed_role": ROLE_NAMES.get(listed) if isinstance(listed, str) else None,
            "chart_pos": p.get("chart_pos") if isinstance(p.get("chart_pos"), str) else None,
            "chart_depth": int(p["chart_depth"]) if pd.notna(p.get("chart_depth")) else None,
            "cfbd_role": ROLE_NAMES.get(p["role_fine"]) if isinstance(p["role_fine"], str) else None,
            **_injury(p),
            "games": int(p["n_games"]), "events": int(p["n_events"]),
            "run_tkl": int(p["n_run_tkl"]), "pass_tkl": int(p["n_pass_tkl"]), "sacks": int(p["n_sack"]),
            "hurries": int(p["n_hurry"]), "pbu": int(p["n_pbu"]), "ints": int(p["n_int"]),
            "pass_rush": int(p["n_sack"] + p["n_hurry"]),
            "location": primary_location(p, mid_avg, center),
            "alignment": ALIGNMENT[p["pred_role"]] + (f", {side_lean(p['left'], p['right'], center)}"
                                                      if side_lean(p["left"], p["right"], center) else ""),
            "depth_med": _r(p["depth_med"], 1), "run_depth_med": _r(p["run_depth_med"], 1),
            "side_skew": _r(p["side_skew"], 2), "left": int(p["left"]), "middle": int(p["middle"]),
            "right": int(p["right"]),
            "pred_role": ROLE_NAMES[p["pred_role"]], "pred_p": _r(p["pred_p"], 2),
            "probs": {ROLE_NAMES[c]: _r(p[f"p_{c}"], 2) for c in fine["labels"]},
            "mismatch": bool(mismatch), "grid": grid,
        })
    conf_tab = fine["confusion"]
    model = {
        "fine": {"labels": [ROLE_NAMES[c] for c in fine["labels"]], "acc": _r(fine["acc"]),
                 "folds": [{"n_test": f["n_test"], "acc": _r(f["acc"])} for f in fine.get("folds", [])],
                 "baseline": _r(fine["baseline"]), "n_train": fine["n_train"], "n_test": fine["n_test"],
                 "confusion": {ROLE_NAMES.get(i, i): {ROLE_NAMES.get(j, j): int(conf_tab.loc[i, j])
                                                      for j in conf_tab.columns} for i in conf_tab.index}},
        "coarse": {"labels": [ROLE_NAMES[c] for c in coarse["labels"]], "acc": _r(coarse["acc"]),
                   "baseline": _r(coarse["baseline"]), "n_train": coarse["n_train"],
                   "n_test": coarse["n_test"]},
        "features": FEATURES, "min_events": MIN_DEF_EVENTS, "mid_avg": _r(mid_avg),
        "left_share": _r(center), "labels_from": "depth charts" if use_chart else "CFBD / Sportradar",
        "cfbd": {"acc": _r(cfbd["acc"]), "baseline": _r(cfbd["baseline"]), "n_test": cfbd["n_test"]},
    }
    if use_chart:
        model["chart_check"] = chart_check(prof, disp, has_chart, center)
    return {"players": out, "model": model, "_profiles": prof, "_coef": fine["coef"]}


def _injury(p) -> dict:
    st = p.get("injury_status")
    if not isinstance(st, str):
        return {}
    return {"injury_status": st, "injury_date": p.get("injury_date"), "injury_note": p.get("injury_note")}


def chart_check(prof: pd.DataFrame, test: np.ndarray, has_chart: np.ndarray, center: float) -> dict:
    """How the chart labels compare with the CFBD labels and the CFBD-label model on the test
    players, and whether chart sides (L/R) show in where plays are charted."""
    t = prof[test & has_chart]
    known = t["role_fine"].notna()
    agree = float((t.loc[known, "role_fine"] == t.loc[known, "chart_role"]).mean()) if known.any() else None
    non_nb = t["chart_role"] != "NB"
    pairs = pd.crosstab(t.loc[known, "chart_role"], t.loc[known, "role_fine"])
    s = prof[has_chart & prof["chart_side"].isin(["L", "R"]).to_numpy()].copy()
    s = s[(s["left"] + s["right"]) >= 8]     # linemen have few tackles with a direction
    s["left_share"] = s["left"] / (s["left"] + s["right"])
    side = {}
    for role, g in s.groupby("chart_role"):
        by = g.groupby("chart_side")["left_share"]
        n, m, v = by.size(), by.mean(), by.var()
        d = {k: {"n": int(n.get(k, 0)), "offense_left_share": _r(m.get(k))} for k in ("L", "R")}
        if n.get("L", 0) >= 5 and n.get("R", 0) >= 5:
            se = float(np.sqrt(v["L"] / n["L"] + v["R"] / n["R"]))
            d["r_minus_l"] = _r(m["R"] - m["L"])
            d["z"] = _r((m["R"] - m["L"]) / se, 2) if se > 0 else None
        side[ROLE_NAMES.get(role, role)] = d
    return {
        "n": int(len(t)), "cfbd_label_known": int(known.sum()), "cfbd_label_agree": _r(agree),
        "cfbd_vs_chart": {ROLE_NAMES.get(i, i): {ROLE_NAMES.get(j, j): int(pairs.loc[i, j]) for j in pairs.columns}
                          for i in pairs.index},
        "cfbd_model_acc_on_chart": _r(float((t.loc[non_nb, "pred_cfbd"] == t.loc[non_nb, "chart_role"]).mean())
                                      if non_nb.any() else None),
        "chart_model_acc_same_players": _r(float((t.loc[non_nb, "pred_role"] == t.loc[non_nb, "chart_role"]).mean())
                                           if non_nb.any() else None),
        "n_non_nickel": int(non_nb.sum()),
        "side": side, "side_n": int(len(s)),
    }


# ----------------------------------------------------------------------------------------------
# Offense


def offense_touches(b: dict) -> pd.DataFrame:
    pl = b["plays"]
    o = b["players"]
    o = o[(o["side"] == "offense") & o["role"].isin(["rusher", "target", "passer"])]
    cols = ["play_id", "game_id", "season", "week", "offense_id", "defense_id", "action", "shotgun",
            "no_huddle", "run_dir", "pass_depth", "pass_dir", "completed", "intercepted", "broken_up",
            "qb_hurried", "text_yards", "air_yards", "yac", "down", "distance", "yards_to_goal",
            "yards_gained", "success", "garbage_time"]
    t = o[["play_id", "role", "athlete_id", "team_id"]].merge(pl[cols], on="play_id", how="inner")
    t = t[t["team_id"] == t["offense_id"]]
    t = t[~((t["role"] == "rusher") & (t["action"] != "rush"))]
    t = t[~((t["role"] == "target") & (t["action"] != "pass"))]
    t["tgt_depth"] = _bin(t["air_yards"], TGT_DEPTHS)
    t["gain_bin"] = _bin(t["text_yards"], RUSH_GAINS)
    t["formation"] = np.where(t["shotgun"].fillna(False), "Shotgun", "Under center")
    t["down_grp"] = np.where(t["down"] >= 3, "3rd/4th", np.where(t["down"] == 2, "2nd", "1st"))
    t["red_zone"] = t["yards_to_goal"] <= 20
    t["touch_type"] = np.where(
        t["role"] == "rusher", "Run " + t["run_dir"].fillna("?").astype(str),
        np.where(t["role"] == "target", t["tgt_depth"].fillna("Pass ?").astype(str), "Pass thrown"))
    return t.reset_index(drop=True)


def _style_receiver(g: pd.DataFrame, center: float, mid_avg: float) -> str:
    tg = g[g["role"] == "target"]
    air = pd.to_numeric(tg["air_yards"], errors="coerce").dropna()
    if not len(air):
        return "unknown"
    behind = (air <= 0).mean()
    deep = (air >= 20).mean()
    mid = (tg["pass_dir"] == "middle").mean()
    if behind >= 0.35:
        s = "Screens / swing passes"
    elif deep >= 0.25 or air.mean() >= 14:
        s = "Vertical / deep"
    elif mid >= mid_avg + 0.12:
        s = "Middle of the field"
    else:
        s = "Outside short-intermediate"
    lean = side_lean(int((tg["pass_dir"] == "left").sum()), int((tg["pass_dir"] == "right").sum()),
                     center, min_n=15, defense=False)
    if lean:
        s += f", targets lean {lean}"
    return s


def _style_rusher(g: pd.DataFrame) -> str:
    r = g[g["role"] == "rusher"]
    if not len(r):
        return "unknown"
    mid = (r["run_dir"] == "middle").mean()
    s = "Inside runner" if mid >= 0.5 else ("Outside runner" if mid <= 0.3 else "Inside/outside mix")
    sg = r["shotgun"].fillna(False).mean()
    s += ", mostly shotgun" if sg >= 0.85 else (", mostly under center" if sg <= 0.4 else "")
    tg = (g["role"] == "target").sum()
    if tg / max(len(r) + tg, 1) >= 0.2:
        s += ", receiving back"
    return s


def offense_section(b: dict, info: pd.DataFrame, season: int, conf_ids: set[int], team_name: dict) -> dict:
    t = offense_touches(b)
    tl = t[(t["season"] == season) & (t["role"] == "target")]
    center = float((tl["pass_dir"] == "left").sum() / tl["pass_dir"].isin(["left", "right"]).sum())
    mid_avg = float((tl["pass_dir"] == "middle").mean())
    t = t[(t["season"] == season) & t["team_id"].isin(conf_ids)]
    ic = ["season", "athlete_id", "player", "position", "chart_pos", "chart_depth", "chart_align",
          "injury_status", "injury_date", "injury_note"]
    by_align = align_summary(tl.merge(info[ic], on=["season", "athlete_id"], how="left"), center)
    t = t.merge(info[ic], on=["season", "athlete_id"], how="left")
    team_tgt = t[t["role"] == "target"].groupby("team_id").size()
    team_rush = t[t["role"] == "rusher"].groupby("team_id").size()
    out = []
    for (tid, aid), g in t.groupby(["team_id", "athlete_id"]):
        tg = g[g["role"] == "target"]
        ru = g[g["role"] == "rusher"]
        pa = g[g["role"] == "passer"]
        if len(tg) < MIN_TARGETS and len(ru) < MIN_RUSHES and len(pa) < MIN_QB_ATT:
            continue
        rec = {"team": team_name.get(int(tid)), "athlete_id": int(aid),
               "player": g["player"].iloc[0] if isinstance(g["player"].iloc[0], str) else str(aid),
               "pos": g["position"].iloc[0] if isinstance(g["position"].iloc[0], str) else None,
               "chart_pos": g["chart_pos"].iloc[0] if isinstance(g["chart_pos"].iloc[0], str) else None,
               "chart_depth": int(g["chart_depth"].iloc[0]) if pd.notna(g["chart_depth"].iloc[0]) else None,
               **_injury(g.iloc[0]),
               "games": int(g["game_id"].nunique()), "targets": int(len(tg)), "rushes": int(len(ru)),
               "attempts": int(len(pa)),
               "tgt_share": _r(len(tg) / team_tgt.get(tid, np.nan), 3),
               "rush_share": _r(len(ru) / team_rush.get(tid, np.nan), 3)}
        if len(tg):
            grid = {}
            for zd, _, _ in TGT_DEPTHS:
                for lt in DIRS:
                    c = tg[(tg["tgt_depth"] == zd) & (tg["pass_dir"] == lt)]
                    grid[f"{zd}|{lt}"] = [int(len(c)), int(c["completed"].fillna(False).sum()),
                                          int(pd.to_numeric(c["text_yards"], errors="coerce").fillna(0)[c["completed"].fillna(False)].sum())]
            rec["tgt_grid"] = grid
            air = pd.to_numeric(tg["air_yards"], errors="coerce")
            rec.update(adot=_r(air.mean(), 1), catch_rate=_r(tg["completed"].fillna(False).mean()),
                       yac=_r(pd.to_numeric(tg["yac"], errors="coerce").mean(), 1),
                       yds_per_tgt=_r(pd.to_numeric(tg["text_yards"], errors="coerce").fillna(0)
                                      .where(tg["completed"].fillna(False), 0).mean(), 1))
        if len(ru):
            rgrid = {}
            for gb, _, _ in RUSH_GAINS:
                for lt in DIRS:
                    rgrid[f"{gb}|{lt}"] = int(((ru["gain_bin"] == gb) & (ru["run_dir"] == lt)).sum())
            rec["rush_grid"] = rgrid
            rec.update(ypc=_r(pd.to_numeric(ru["text_yards"], errors="coerce").mean(), 2),
                       rush_success=_r(ru["success"].astype(float).mean()),
                       stuff_rate=_r((pd.to_numeric(ru["text_yards"], errors="coerce") <= 0).mean()))
        if len(pa):
            pgrid = {}
            pa = pa.assign(tgt_depth=_bin(pa["air_yards"], TGT_DEPTHS))
            for zd, _, _ in TGT_DEPTHS:
                for lt in DIRS:
                    c = pa[(pa["tgt_depth"] == zd) & (pa["pass_dir"] == lt)]
                    pgrid[f"{zd}|{lt}"] = [int(len(c)), int(c["completed"].fillna(False).sum())]
            rec["pass_grid"] = pgrid
            rec["ints"] = int(pa["intercepted"].fillna(False).sum())
            rec.update(comp_pct=_r(pa["completed"].fillna(False).mean()),
                       pass_adot=_r(pd.to_numeric(pa["air_yards"], errors="coerce").mean(), 1))
        touches = g[g["role"].isin(["rusher", "target"])] if len(tg) + len(ru) else g
        mix = touches["touch_type"].value_counts(normalize=True)
        rec["touch_mix"] = {k: _r(v) for k, v in mix.items()}
        rec["situation"] = {
            "Shotgun": _r((touches["formation"] == "Shotgun").mean()),
            "Under center": _r((touches["formation"] == "Under center").mean()),
            "No huddle": _r(touches["no_huddle"].fillna(False).mean()),
            "1st down": _r((touches["down_grp"] == "1st").mean()),
            "2nd down": _r((touches["down_grp"] == "2nd").mean()),
            "3rd/4th down": _r((touches["down_grp"] == "3rd/4th").mean()),
            "Red zone": _r(touches["red_zone"].mean()),
        }
        if len(pa) >= MIN_QB_ATT and len(pa) >= len(tg) + len(ru):
            rec["style"] = "Passer"
        elif len(ru) >= len(tg):
            rec["style"] = _style_rusher(g)
        else:
            rec["style"] = _style_receiver(g, center, mid_avg)
        out.append(rec)
    return {"players": out, "left_share": _r(center), "mid_share": _r(mid_avg), "by_align": by_align}


def align_summary(tl: pd.DataFrame, center: float) -> list[dict]:
    """Targets of every charted team (P4 + Notre Dame) by the receiver's chart alignment:
    depth mix, direction mix, aDOT and the share of the team's targets."""
    tl = tl[tl["chart_align"].notna()]
    out = []
    for a in ("X", "Z", "SLOT", "TE", "RB"):
        g = tl[tl["chart_align"] == a]
        if g.empty:
            continue
        lr = g["pass_dir"].isin(["left", "right"]).sum()
        depth = pd.to_numeric(g["air_yards"], errors="coerce")
        out.append({
            "align": ALIGN_NAMES[a], "targets": int(len(g)), "players": int(g["athlete_id"].nunique()),
            "share": _r(len(g) / len(tl)), "adot": _r(depth.mean(), 1),
            "depth": {name: _r(float(((depth >= lo) & (depth <= hi)).mean())) for name, lo, hi in TGT_DEPTHS},
            "dir": {d: _r(float((g["pass_dir"] == d).mean())) for d in DIRS},
            "left_of_lr": _r(float((g["pass_dir"] == "left").sum() / lr)) if lr else None,
        })
    return out


# ----------------------------------------------------------------------------------------------
# Running backs


def rb_section(b: dict, info: pd.DataFrame, season: int, conf_ids: set[int], team_name: dict,
               k_rb: float = 15.0, k_def: float = 25.0) -> dict:
    t = offense_touches(b)
    t = t[(t["season"] == season) & (t["role"] == "rusher") & ~t["garbage_time"] & t["run_dir"].notna()]
    t = t.merge(info[["season", "athlete_id", "player", "position"]], on=["season", "athlete_id"], how="left")
    t = t[t["position"].isin(["RB", "FB"])].copy()
    t["succ"] = t["success"].astype(float)
    t["stuff"] = (pd.to_numeric(t["text_yards"], errors="coerce") <= 0).astype(float)
    t["yds"] = pd.to_numeric(t["text_yards"], errors="coerce")
    fbs_dir = t.groupby("run_dir")["succ"].mean()
    fbs_cell = t.groupby(["run_dir", "formation"])["succ"].mean()
    fbs_stuff_dir = t.groupby("run_dir")["stuff"].mean()
    # Opponent run defense by direction, shrunk to the FBS rate.
    dd = t.groupby(["defense_id", "run_dir"]).agg(n=("succ", "size"), s=("succ", "sum")).reset_index()
    dd["rate"] = (dd["s"] + k_def * dd["run_dir"].map(fbs_dir)) / (dd["n"] + k_def)
    dd["adj"] = dd["rate"] - dd["run_dir"].map(fbs_dir)
    opp_adj = {(int(a), d): v for a, d, v in zip(dd["defense_id"], dd["run_dir"], dd["adj"])}
    opp_rate = {(int(a), d): v for a, d, v in zip(dd["defense_id"], dd["run_dir"], dd["rate"])}

    games = b["games"].set_index("game_id")
    c = t[t["team_id"].isin(conf_ids)]
    players, game_rows = [], []
    for (tid, aid), g in c.groupby(["team_id", "athlete_id"]):
        if len(g) < MIN_RUSHES:
            continue
        name = g["player"].iloc[0] if isinstance(g["player"].iloc[0], str) else str(aid)
        base = {}
        for d in DIRS:
            gd = g[g["run_dir"] == d]
            base[d] = (gd["succ"].sum() + k_rb * fbs_dir[d]) / (len(gd) + k_rb)
        cells = []
        for (d, f), gc in g.groupby(["run_dir", "formation"]):
            n = len(gc)
            exp_rate = fbs_cell.get((d, f), fbs_dir[d])
            # Opponent mix: expected success had an average FBS back run these carries vs these defenses.
            exp_opp = float(np.mean([exp_rate + opp_adj.get((int(o), d), 0.0) for o in gc["defense_id"]]))
            cells.append({"dir": d, "formation": f, "n": int(n), "success": _r(gc["succ"].mean()),
                          "stuff": _r(gc["stuff"].mean()), "ypc": _r(gc["yds"].mean(), 1),
                          "fbs_success": _r(exp_rate), "exp_vs_opps": _r(exp_opp),
                          "delta": _r(gc["succ"].mean() - exp_opp),
                          "z": _r((gc["succ"].mean() - exp_opp) / math.sqrt(max(exp_opp * (1 - exp_opp), 1e-6) / n), 2)})
        cells.sort(key=lambda x: (x["delta"] if x["n"] >= 8 else 9))
        dir_rows = []
        for d in DIRS:
            gd = g[g["run_dir"] == d]
            n = len(gd)
            if not n:
                continue
            exp_opp = float(np.mean([fbs_dir[d] + opp_adj.get((int(o), d), 0.0) for o in gd["defense_id"]]))
            z = (gd["succ"].mean() - exp_opp) / math.sqrt(max(exp_opp * (1 - exp_opp), 1e-6) / n)
            dir_rows.append({"dir": d, "formation": "All", "n": int(n), "success": _r(gd["succ"].mean()),
                             "stuff": _r(gd["stuff"].mean()), "ypc": _r(gd["yds"].mean(), 1),
                             "fbs_success": _r(fbs_dir[d]), "exp_vs_opps": _r(exp_opp),
                             "delta": _r(gd["succ"].mean() - exp_opp), "z": _r(z, 2)})
        trouble = [x for x in dir_rows + cells if x["n"] >= 8 and x["z"] is not None and x["z"] <= -1.28]
        ok = [x for x in dir_rows if x["n"] >= 8]
        weakest = min(ok, key=lambda x: x["delta"]) if ok else None
        situ = {}
        for label, m in (("1st & 10", (g["down"] == 1) & (g["distance"] >= 10)),
                         ("2nd & 7+", (g["down"] == 2) & (g["distance"] >= 7)),
                         ("Short yardage (<=2 to go)", g["distance"] <= 2),
                         ("Red zone", g["yards_to_goal"] <= 20),
                         ("Goal to go (<=5)", g["yards_to_goal"] <= 5)):
            gs = g[m]
            situ[label] = {"n": int(len(gs)), "success": _r(gs["succ"].mean()) if len(gs) else None,
                           "stuff": _r(gs["stuff"].mean()) if len(gs) else None}
        rec = {"team": team_name.get(int(tid)), "athlete_id": int(aid), "player": name,
               "rushes": int(len(g)), "games": int(g["game_id"].nunique()),
               "success": _r(g["succ"].mean()), "stuff": _r(g["stuff"].mean()), "ypc": _r(g["yds"].mean(), 2),
               "dirs": {d: {"n": int((g["run_dir"] == d).sum()),
                            "success": _r(g.loc[g["run_dir"] == d, "succ"].mean()),
                            "stuff": _r(g.loc[g["run_dir"] == d, "stuff"].mean()),
                            "ypc": _r(g.loc[g["run_dir"] == d, "yds"].mean(), 1),
                            "fbs_success": _r(fbs_dir[d]), "fbs_stuff": _r(fbs_stuff_dir[d]),
                            "baseline_shrunk": _r(base[d])} for d in DIRS},
               "cells": cells, "dir_vs_opps": dir_rows, "trouble": trouble, "weakest": weakest,
               "situations": situ}
        players.append(rec)
        for gid, gg in g.groupby("game_id"):
            opp = int(gg["defense_id"].iloc[0])
            gi = games.loc[gid] if gid in games.index else None
            row = {"team": rec["team"], "athlete_id": int(aid), "player": name, "game_id": int(gid),
                   "week": int(gg["week"].iloc[0]), "opponent": team_name.get(opp, str(opp)),
                   "home": bool(gi is not None and int(gi["home_id"]) == int(tid)),
                   "rushes": int(len(gg)), "success": _r(gg["succ"].mean()), "ypc": _r(gg["yds"].mean(), 1),
                   "stuff": _r(gg["stuff"].mean()), "dirs": {}}
            worst = None
            for d in DIRS:
                gd = gg[gg["run_dir"] == d]
                n = len(gd)
                exp = min(max(base[d] + opp_adj.get((opp, d), 0.0), 0.02), 0.98)
                cell = {"n": int(n), "success": _r(gd["succ"].mean()) if n else None,
                        "stuff": _r(gd["stuff"].mean()) if n else None,
                        "ypc": _r(gd["yds"].mean(), 1) if n else None,
                        "expected": _r(exp), "opp_allowed": _r(opp_rate.get((opp, d))),
                        "opp_adj": _r(opp_adj.get((opp, d), 0.0))}
                if n:
                    z = (gd["succ"].mean() - exp) / math.sqrt(exp * (1 - exp) / n)
                    cell["z"] = _r(z, 2)
                    cell["flag"] = bool(n >= 4 and z <= -1.28)
                    if cell["flag"] and (worst is None or z < worst[1]):
                        worst = (d, z)
                row["dirs"][d] = cell
            row["worst_dir"] = worst[0] if worst else None
            game_rows.append(row)
    return {"players": players, "games": game_rows,
            "fbs": {"dir_success": {d: _r(fbs_dir[d]) for d in DIRS},
                    "dir_stuff": {d: _r(fbs_stuff_dir[d]) for d in DIRS}},
            "k_rb": k_rb, "k_def": k_def}


# ----------------------------------------------------------------------------------------------
# Receivers and QBs: clean incompletions


def target_rows(b: dict) -> pd.DataFrame:
    """One row per pass with a named target and passer (both seasons, all FBS teams)."""
    pl = b["plays"]
    pp = b["players"]
    pl = pl[(pl["action"] == "pass") & ~pl["garbage_time"]]
    tg = pp[pp["role"] == "target"][["play_id", "athlete_id", "team_id"]].drop_duplicates("play_id")
    qb = pp[pp["role"] == "passer"][["play_id", "athlete_id"]].drop_duplicates("play_id")
    t = pl.merge(tg.rename(columns={"athlete_id": "rec_id"}), on="play_id").merge(
        qb.rename(columns={"athlete_id": "qb_id"}), on="play_id")
    t = t[t["team_id"] == t["offense_id"]].copy()
    t["air"] = pd.to_numeric(t["air_yards"], errors="coerce")
    t = t[t["air"].notna() & t["pass_dir"].notna()]
    # Stat crews credit a hurry almost only when the pass falls incomplete (2026: 3,886 of 4,230
    # hurried passes), so hurried throws are pressure-affected misses: left out, not modelled.
    t["hurried"] = t["qb_hurried"].fillna(False).astype(bool)
    t["clean_inc"] = (~t["completed"].fillna(False) & ~t["broken_up"].fillna(False)
                      & ~t["intercepted"].fillna(False)).astype(float)
    t["air_bin"] = pd.cut(t["air"], AIR_BINS, labels=False, right=True)
    t["tgt_depth"] = _bin(t["air"], TGT_DEPTHS)
    return t.reset_index(drop=True)


def receiving_section(b: dict, info: pd.DataFrame, season: int, conf_ids: set[int], team_name: dict) -> dict:
    allt = target_rows(b)
    t = allt[~allt["hurried"]].reset_index(drop=True)
    fit = fit_clean_inc(t, sweeps=60)
    sig = lambda x: 1 / (1 + np.exp(-x))
    t["p"] = fit["p"]
    t["p_team"] = sig(fit["eta"] - fit["row_eff"]["qb"] - fit["row_eff"]["rec"])  # average QB and receivers
    t["p_qb"] = sig(fit["eta"] - fit["row_eff"]["qb"])     # average QB, his receivers
    t["p_rec"] = sig(fit["eta"] - fit["row_eff"]["rec"])   # average receiver, his QBs
    names = info.drop_duplicates("athlete_id", keep="last").set_index("athlete_id")
    qb_eff = fit["qb"].set_index("id")
    rec_eff = fit["rec"].set_index("id")
    league_cell = t.groupby(["tgt_depth", "pass_dir"])["clean_inc"].mean().to_dict()
    ts = t[(t["season"] == season) & t["team_id"].isin(conf_ids)]
    teams_out = []
    for tid, g in ts.groupby("team_id"):
        qbs, recs = [], []
        for qid, gq in g.groupby("qb_id"):
            if len(gq) < MIN_QB_ATT:
                continue
            e = qb_eff.loc[qid]
            hq = allt[(allt["qb_id"] == qid) & (allt["season"] == season) & (allt["team_id"] == tid)]
            qbs.append({"athlete_id": int(qid), "player": _name(names, qid), "targets": int(len(gq)),
                        "hurried": int(hq["hurried"].sum()), "hurried_share": _r(hq["hurried"].mean()),
                        "clean_inc": _r(gq["clean_inc"].mean()), "expected": _r(gq["p_qb"].mean()),
                        "effect_pp": _r(e["pp"], 1), "z": _r(e["z"], 2),
                        "career_n": int(e["n"]), "flag": bool(e["z"] >= EFFECT_Z),
                        "watch": bool(1.64 <= e["z"] < EFFECT_Z)})
        team_rate = g["clean_inc"].mean()
        for rid, gr in g.groupby("rec_id"):
            if len(gr) < MIN_REC_TGT:
                continue
            e = rec_eff.loc[rid]
            mates = g[g["rec_id"] != rid]
            recs.append({"athlete_id": int(rid), "player": _name(names, rid),
                         "pos": names.loc[rid, "position"] if rid in names.index else None,
                         "chart_pos": (names.loc[rid, "chart_pos"] if rid in names.index
                                       and isinstance(names.loc[rid, "chart_pos"], str) else None),
                         **(_injury(names.loc[rid]) if rid in names.index else {}),
                         "targets": int(len(gr)), "clean_inc": _r(gr["clean_inc"].mean()),
                         "teammates": _r(mates["clean_inc"].mean()), "expected": _r(gr["p_rec"].mean()),
                         "adot": _r(gr["air"].mean(), 1),
                         "effect_pp": _r(e["pp"], 1), "z": _r(e["z"], 2), "career_n": int(e["n"]),
                         "flag": bool(e["z"] >= EFFECT_Z), "watch": bool(1.64 <= e["z"] < EFFECT_Z)})
        # Play-type cells: receiver vs teammates on the same depth x direction. Teammates' rate is
        # shrunk to the FBS rate for the cell (CELL_K throws); flag = binomial tail p < CELL_P.
        cells = []
        for (rid, d, lt), gc in g.groupby(["rec_id", "tgt_depth", "pass_dir"]):
            n = len(gc)
            mates = g[(g["rec_id"] != rid) & (g["tgt_depth"] == d) & (g["pass_dir"] == lt)]
            if n < 5:
                continue
            q = (mates["clean_inc"].sum() + CELL_K * league_cell.get((d, lt), 0.2)) / (len(mates) + CELL_K)
            k = int(gc["clean_inc"].sum())
            pval = binom_sf(k, n, q)
            cells.append({"athlete_id": int(rid), "player": _name(names, rid), "depth": d, "dir": lt,
                          "n": int(n), "clean": k, "clean_inc": _r(k / n),
                          "teammates": _r(mates["clean_inc"].mean()) if len(mates) else None,
                          "teammates_shrunk": _r(q), "mates_n": int(len(mates)), "p": _r(pval, 4),
                          "flag": bool(pval < CELL_P and n >= 6 and k / n - q >= 0.2)})
        qflag = any(q["flag"] for q in qbs)
        rflag = [r["player"] for r in recs if r["flag"]]
        cflag = sorted({c["player"] for c in cells if c["flag"]})
        if qflag and rflag:
            verdict = "Both: QB and " + ", ".join(rflag)
        elif qflag:
            verdict = "QB issue"
        elif rflag:
            verdict = "Receiver issue: " + ", ".join(rflag)
        elif cflag:
            verdict = "No overall issue; play-type spots: " + ", ".join(cflag)
        else:
            verdict = "No clear issue"
        teams_out.append({"team": team_name.get(int(tid)), "targets": int(len(g)),
                          "clean_inc": _r(team_rate), "expected": _r(g["p_team"].mean()),
                          "verdict": verdict, "qbs": sorted(qbs, key=lambda x: -x["targets"]),
                          "receivers": sorted(recs, key=lambda x: -x["targets"]),
                          "cells": sorted(cells, key=lambda x: x["p"])})
    base = {
        "rows": int(len(t)), "hurried_rows_left_out": int(allt["hurried"].sum()),
        "clean_inc_rate": _r(t["clean_inc"].mean()), "late_down_coef": _r(fit["late_coef"]),
        "by_depth": {d: _r(v) for d, v in t.groupby("tgt_depth")["clean_inc"].mean().items()},
        "tau": {"qb": 0.4, "rec": 0.4, "def": 0.25}, "effect_z": EFFECT_Z, "cell_p": CELL_P,
        "cell_k": CELL_K,
    }
    # Clean incompletions by chart alignment, every charted team, current season, with the
    # model's expectation for an average receiver on the same throws (p_rec).
    ch = names["chart_align"].dropna() if "chart_align" in names else pd.Series(dtype=object)
    cur = t[t["season"] == season].assign(align=lambda d: d["rec_id"].map(ch))
    cur = cur[cur["align"].isin(["X", "Z", "SLOT", "TE", "RB"])]
    by = []
    for a, g in cur.groupby("align"):
        eff = rec_eff.reindex(g["rec_id"].unique())
        by.append({"align": ALIGN_NAMES[a], "targets": int(len(g)), "clean_inc": _r(g["clean_inc"].mean()),
                   "expected": _r(g["p_rec"].mean()), "adot": _r(g["air"].mean(), 1),
                   "flagged": int((eff["z"] >= EFFECT_Z).sum()), "receivers": int(len(eff))})
    base["by_align"] = sorted(by, key=lambda x: list(ALIGN_NAMES.values()).index(x["align"]))
    return {"teams": sorted(teams_out, key=lambda x: x["team"]), "model": base}


def binom_sf(k: int, n: int, q: float) -> float:
    """P(X >= k) for X ~ Binomial(n, q)."""
    return float(sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1)))


def _name(names: pd.DataFrame, aid) -> str:
    if aid in names.index and isinstance(names.loc[aid, "player"], str):
        return names.loc[aid, "player"]
    return str(aid)


# ----------------------------------------------------------------------------------------------
# Build


def build(data_dir="data", season: int = 2026, conference: str = "SEC", fit_seasons=(2025, 2026)) -> dict:
    b = load(data_dir, fit_seasons)
    teams = b["teams"]
    ts = teams[teams["season"] == season]
    if conference.upper() == "P4":
        # Every charted team: the four power conferences and Notre Dame, one fold per conference.
        scope = ts["conference"].isin(P4_CONFERENCES) | ts["team"].isin(P4_EXTRA)
        groups = ts[scope].assign(g=lambda d: d["conference"].where(d["conference"].isin(P4_CONFERENCES), d["team"]))
        folds = [set(g["team_id"].astype(int)) for _, g in groups.groupby("g")]
        fold_names = [n for n, _ in groups.groupby("g")]
        conference = "P4"
    else:
        scope = ts["conference"] == conference
        folds, fold_names = None, [conference]
    conf_ids = set(ts.loc[scope, "team_id"].astype(int))
    team_name = {int(i): n for i, n in zip(teams["team_id"], teams["team"])}
    info = player_info(b)
    plays = b["plays"]
    ps = plays[plays["season"] == season]
    weeks = sorted(ps["week"].unique().tolist())
    defense = defense_section(b, info, season, conf_ids, team_name, folds)
    for f, n in zip(defense["model"]["fine"].get("folds", []), fold_names):
        f["name"] = n
    result = {
        "meta": {
            "season": season, "conference": conference, "fit_seasons": list(fit_seasons),
            "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "weeks": [int(weeks[0]), int(weeks[-1])] if weeks else None,
            "plays_season": int(len(ps)),
            "plays_conf": int((ps["offense_id"].isin(conf_ids) | ps["defense_id"].isin(conf_ids)).sum()),
            "teams": sorted(team_name[i] for i in conf_ids),
            "chart_asof": (pd.Timestamp(b["chart"]["chart_asof"].max()).strftime("%Y-%m-%d")
                           if len(b["chart"]) else None),
            "chart_teams": int(b["chart"]["chart_team"].nunique()) if len(b["chart"]) else 0,
            "injury_rows": int(len(b["injuries"])), "folds": fold_names,
        },
        "defense": {k: v for k, v in defense.items() if not k.startswith("_")},
        "offense": offense_section(b, info, season, conf_ids, team_name),
        "rb": rb_section(b, info, season, conf_ids, team_name),
        "receiving": receiving_section(b, info, season, conf_ids, team_name),
    }
    result["_profiles"] = defense["_profiles"]
    result["_coef"] = defense["_coef"]
    return result


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    if o is pd.NA:
        return None
    return o


def write(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = result["meta"]
    stem = f"player_maps_{m['season']}_{m['conference'].lower()}"
    payload = _clean({k: v for k, v in result.items() if not k.startswith("_")})
    js = out_dir / f"{stem}.json"
    js.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    result["_profiles"].to_parquet(out_dir / f"{stem}_defender_profiles.parquet", index=False)
    result["_coef"].to_csv(out_dir / f"{stem}_role_coef.csv")
    template = (Path(__file__).with_name("player_maps_viewer.html")).read_text(encoding="utf-8")
    html = out_dir / f"{stem}.html"
    html.write_text(template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"))
                                     .replace("</", "<\\/")), encoding="utf-8")
    return js, html


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--conference", default="SEC")
    ap.add_argument("--fit-seasons", default="2025,2026")
    ap.add_argument("--out", default=None, help="output folder (default data/lab)")
    a = ap.parse_args(argv)
    fit = tuple(int(s) for s in a.fit_seasons.split(","))
    res = build(a.data_dir, a.season, a.conference, fit)
    out = Path(a.out) if a.out else Path(a.data_dir) / "lab"
    js, html = write(res, out)
    dm = res["defense"]["model"]
    print(f"defenders {len(res['defense']['players'])}  role model acc {dm['fine']['acc']} "
          f"(majority {dm['fine']['baseline']}, n_test {dm['fine']['n_test']}); coarse acc {dm['coarse']['acc']} "
          f"(majority {dm['coarse']['baseline']})")
    print(f"offense players {len(res['offense']['players'])}  RBs {len(res['rb']['players'])}  "
          f"teams {len(res['receiving']['teams'])}")
    for t in res["receiving"]["teams"]:
        print(f"  {t['team']:<18} clean inc {t['clean_inc']} exp {t['expected']}  {t['verdict']}")
    print(js)
    print(html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
