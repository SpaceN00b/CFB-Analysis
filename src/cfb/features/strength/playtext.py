"""Features from the detailed stat-crew play text (group `str_pass_ci`, plus shared helpers).

The detailed text (`cfb.core.playtext`, `text_format == "detailed"`) names pass depth and
direction, air yards, run direction, tacklers, breakups and hurries. It exists for every game
from 2025 week 9 on, so these columns are empty before that. Method and evaluation:
docs/player-maps.md ("Model features").

`str_pass_ci`: clean incompletions. A clean incompletion is a pass to a named receiver that
falls incomplete with no breakup, interception or hurry credited (hurries are credited almost
only on incompletions, so hurried throws are left out). At each cutoff a logistic model is fit
on every visible unhurried target of the cutoff season and the one before it: a cell per
air-yard bin and direction, a 3rd/4th-down term, and ridge effects for passer, receiver and
defense, by backfitting. Garbage time is left out.

    str_pci_qb    expected starter's passer effect (logit; positive = more clean incompletions)
    str_pci_rec   receivers' effect, weighted by their share of the team's targets
    str_pci_def   the team's defense effect (positive = forces more clean incompletions)
    str_pci_n     log(1 + the team's visible unhurried targets as offense)

Expected starter: the passer with the most attempts in the team's last visible game with
detailed text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from .common import team_keys

DIRS = ("left", "middle", "right")
AIR_BINS = (-99, 0, 5, 10, 15, 20, 30, 999)
TAU = {"qb": 0.4, "rec": 0.4, "def": 0.25}   # prior SD of the ridge effects, logit scale
SWEEPS = 40

PLAY_COLS = ["game_id", "play_id", "season", "text_format", "action", "run_dir", "pass_dir", "pass_depth",
             "completed", "intercepted", "broken_up", "qb_hurried", "air_yards", "text_yards", "no_play", "kneel",
             "spike"]
PBP_COLS = ["game_id", "play_id", "offense_id", "defense_id", "down", "distance", "yards_to_goal", "success",
            "ppa", "is_scrimmage", "garbage_time"]
PLAYER_COLS = ["play_id", "season", "role", "athlete_id", "team_id", "side"]


def flag(s: pd.Series) -> np.ndarray:
    """Boolean array from a boolean, numeric or object column; missing counts as False."""
    v = pd.to_numeric(s.astype("object").where(s.notna(), 0), errors="coerce")
    return v.fillna(0).to_numpy(dtype=float) != 0


def detailed_plays(view, seasons) -> pd.DataFrame:
    """Visible scrimmage rushes, passes and sacks with detailed text, joined to pbp."""
    if not (view.has("play_detail") and view.has("pbp")):
        return pd.DataFrame(columns=PLAY_COLS + PBP_COLS[2:])
    d = view.table("play_detail", seasons=list(seasons), columns=PLAY_COLS)
    d = d[(d["text_format"] == "detailed").to_numpy()]
    if d.empty:
        return pd.DataFrame(columns=PLAY_COLS + PBP_COLS[2:])
    p = view.table("pbp", seasons=list(seasons), columns=PBP_COLS)
    p = p[p["game_id"].isin(d["game_id"].unique()).to_numpy()]
    x = d.merge(p.drop(columns="game_id"), on="play_id", how="inner")
    keep = (x["action"].isin(["rush", "pass", "sack"]).to_numpy() & flag(x["is_scrimmage"])
            & ~flag(x["no_play"]) & ~flag(x["kneel"]) & ~flag(x["spike"]))
    x = x[keep].reset_index(drop=True)
    for c in ("completed", "intercepted", "broken_up", "qb_hurried", "garbage_time", "success"):
        x[c] = flag(x[c])
    return x


def play_roles(view, seasons, play_ids, roles) -> pd.DataFrame:
    if not view.has("play_players"):
        return pd.DataFrame(columns=PLAYER_COLS)
    pp = view.table("play_players", seasons=list(seasons), columns=PLAYER_COLS)
    pp = pp[pp["role"].isin(roles).to_numpy() & pp["play_id"].isin(play_ids).to_numpy()]
    return pp[pp["athlete_id"].notna().to_numpy()]


def target_rows(plays: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per pass with a named target and passer; clean_inc, air bin, hurried."""
    pl = plays[(plays["action"] == "pass").to_numpy() & ~flag(plays["garbage_time"])]
    tg = players[players["role"] == "target"][["play_id", "athlete_id", "team_id"]].drop_duplicates("play_id")
    qb = players[players["role"] == "passer"][["play_id", "athlete_id"]].drop_duplicates("play_id")
    t = pl.merge(tg.rename(columns={"athlete_id": "rec_id"}), on="play_id").merge(
        qb.rename(columns={"athlete_id": "qb_id"}), on="play_id")
    t = t[(t["team_id"] == t["offense_id"]).to_numpy()].copy()
    t["air"] = pd.to_numeric(t["air_yards"], errors="coerce")
    t = t[t["air"].notna().to_numpy() & t["pass_dir"].notna().to_numpy()].copy()
    t["hurried"] = flag(t["qb_hurried"])
    t["clean_inc"] = (~flag(t["completed"]) & ~flag(t["broken_up"]) & ~flag(t["intercepted"])).astype(float)
    t["air_bin"] = pd.cut(t["air"], AIR_BINS, labels=False, right=True)
    return t.reset_index(drop=True)


def fit_clean_inc(t: pd.DataFrame, tau=(TAU["qb"], TAU["rec"], TAU["def"]), sweeps: int = SWEEPS) -> dict:
    """Logistic model on unhurried targets: a cell per air-yard bin x direction, a 3rd/4th-down
    term, and ridge effects for passer, receiver and defense (prior SD `tau`, logit scale),
    fit by backfitting with one Newton step per effect and sweep.

    Returns p and eta per row, the 3rd/4th-down coefficient, cell coefficients, per-row
    effects, and one frame per effect group: id, coef, se, z, pp (effect in percentage
    points averaged over the player's rows), exp_without, n, obs.
    """
    y = t["clean_inc"].to_numpy(dtype=float)
    cell = (t["air_bin"].astype(int).to_numpy() * 3
            + t["pass_dir"].map({"left": 0, "middle": 1, "right": 2}).to_numpy(dtype=int))
    n_cell = len(AIR_BINS) * 3
    late = (pd.to_numeric(t["down"], errors="coerce").fillna(1) >= 3).to_numpy(dtype=float)
    groups = {}
    for name, col in (("qb", "qb_id"), ("rec", "rec_id"), ("def", "defense_id")):
        codes, uniq = pd.factorize(t[col].astype("int64"))
        groups[name] = (codes, uniq)
    lam = {"qb": 1 / tau[0] ** 2, "rec": 1 / tau[1] ** 2, "def": 1 / tau[2] ** 2}
    a = np.zeros(n_cell)
    bl = 0.0
    eff = {g: np.zeros(len(groups[g][1])) for g in groups}

    def eta():
        e = a[cell] + bl * late
        for g, (codes, _) in groups.items():
            e = e + eff[g][codes]
        return e

    for _ in range(sweeps):
        p = 1 / (1 + np.exp(-eta()))
        a += np.bincount(cell, y - p, n_cell) / (np.bincount(cell, p * (1 - p), n_cell) + 1.0)
        p = 1 / (1 + np.exp(-eta()))
        bl += ((y - p) * late).sum() / ((p * (1 - p) * late).sum() + 1.0)
        for g, (codes, uniq) in groups.items():
            p = 1 / (1 + np.exp(-eta()))
            k = len(uniq)
            grad = np.bincount(codes, y - p, k) - lam[g] * eff[g]
            hess = np.bincount(codes, p * (1 - p), k) + lam[g]
            eff[g] += grad / hess
    e = eta()
    p = 1 / (1 + np.exp(-e))
    w = p * (1 - p)
    out = {"p": p, "eta": e, "late_coef": bl, "cell_coef": a,
           "row_eff": {g: eff[g][codes] for g, (codes, _) in groups.items()}}
    for g, (codes, uniq) in groups.items():
        k = len(uniq)
        n = np.maximum(np.bincount(codes, minlength=k), 1)
        se = 1 / np.sqrt(np.bincount(codes, w, k) + lam[g])
        p0 = 1 / (1 + np.exp(-(e - eff[g][codes])))
        out[g] = pd.DataFrame({"id": uniq, "coef": eff[g], "se": se, "z": eff[g] / se,
                               "pp": np.bincount(codes, p - p0, k) / n * 100,
                               "exp_without": np.bincount(codes, p0, k) / n,
                               "n": np.bincount(codes, minlength=k), "obs": np.bincount(codes, y, k) / n})
    return out


def _last_game_starter(t_all: pd.DataFrame, gstart: dict) -> dict:
    """team_id -> passer with the most attempts in the team's last visible detailed game."""
    if t_all.empty:
        return {}
    c = t_all.groupby(["offense_id", "game_id", "qb_id"]).size().rename("n").reset_index()
    c["start"] = c["game_id"].map(gstart)
    c = c.sort_values(["offense_id", "start", "n"], ascending=[True, False, False])
    last = c.drop_duplicates("offense_id")[["offense_id", "game_id"]]
    c = c.merge(last, on=["offense_id", "game_id"])
    top = c.sort_values("n", ascending=False).drop_duplicates("offense_id")
    return dict(zip(top["offense_id"].astype("int64"), top["qb_id"].astype("int64")))


def pass_ci_state(view) -> pd.DataFrame:
    """Per team at the cutoff: qb, rec, def effects and target count (empty frame without data)."""
    seasons = [view.season - 1, view.season]
    plays = detailed_plays(view, seasons)
    cols = ["str_pci_qb", "str_pci_rec", "str_pci_def", "str_pci_n"]
    if plays.empty:
        return pd.DataFrame(columns=cols)
    players = play_roles(view, seasons, plays.loc[plays["action"] == "pass", "play_id"], ["target", "passer"])
    t_all = target_rows(plays, players)
    t = t_all[~t_all["hurried"].to_numpy()].reset_index(drop=True)
    if len(t) < 200:
        return pd.DataFrame(columns=cols)
    fit = fit_clean_inc(t)
    qb = dict(zip(fit["qb"]["id"].astype("int64"), fit["qb"]["coef"]))
    rec = dict(zip(fit["rec"]["id"].astype("int64"), fit["rec"]["coef"]))
    de = dict(zip(fit["def"]["id"].astype("int64"), fit["def"]["coef"]))
    gi = view.table("game_index", columns=["game_id", "start_time_utc"])
    gstart = dict(zip(gi["game_id"], gi["start_time_utc"]))
    starter = _last_game_starter(t_all, gstart)
    # Receivers weighted by targets this season (the previous season when the team has none yet).
    cur = t[t["season"] == view.season]
    have_cur = set(cur["offense_id"].astype("int64"))
    base = pd.concat([cur, t[(t["season"] != view.season).to_numpy() & ~t["offense_id"].isin(have_cur).to_numpy()]])
    base = base.assign(rec_coef=base["rec_id"].astype("int64").map(rec))
    rec_team = base.groupby("offense_id")["rec_coef"].mean()
    n_team = t.groupby("offense_id").size()
    teams = sorted(set(n_team.index.astype("int64")) | set(int(i) for i in de))
    out = pd.DataFrame(index=pd.Index(teams, name="team_id"))
    out["str_pci_qb"] = [qb.get(starter.get(i), np.nan) if i in starter else np.nan for i in teams]
    out["str_pci_rec"] = rec_team.reindex(teams).to_numpy(dtype=float)
    out["str_pci_def"] = [de.get(i, np.nan) for i in teams]
    out["str_pci_n"] = np.log1p(n_team.reindex(teams).fillna(0).to_numpy(dtype=float))
    return out


class PassAccuracy(FeatureGroup):
    name = "str_pass_ci"
    prefix = "str_"
    level = "team"
    columns = [
        FeatureColumn("str_pci_qb", "Expected starter's clean-incompletion effect as passer (logit, shrunk; "
                      "detailed play text, 2025 week 9 on)", ("play_detail", "play_players", "pbp"), 2025),
        FeatureColumn("str_pci_rec", "Receivers' clean-incompletion effect, weighted by their targets for the team",
                      ("play_detail", "play_players", "pbp"), 2025),
        FeatureColumn("str_pci_def", "Defense's clean-incompletion effect (positive = forces more)",
                      ("play_detail", "play_players", "pbp"), 2025),
        FeatureColumn("str_pci_n", "log(1 + the team's visible unhurried targets with detailed text)",
                      ("play_detail", "play_players", "pbp"), 2025),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        st = pass_ci_state(view)
        out = keys[["game_id", "team_id"]].reset_index(drop=True).copy()
        st = st.reindex(out["team_id"].to_numpy())
        for c in ("str_pci_qb", "str_pci_rec", "str_pci_def", "str_pci_n"):
            out[c] = st[c].to_numpy(dtype=np.float64) if c in st else np.nan
        return out
