"""Player-impact study, part 2: defensive absences and several players out at once.

Inputs: data/exploration/impact_base.parquet, data/exploration/impact_roles_ps.parquet
(scripts/impact_players.py --plays-stats; receivers' share = target share).

Defense has no snap counts in CFBD (spec 5.34.0) and /plays/stats has no Tackle rows, so the
defensive role is the box-score tackle share (sh_tkl) and the "playmaking" share (TFL + sacks +
passes defended + interceptions, sh_dplay). "Regular defenders" = the team's 11 DL/LB/DB with
the highest tackle share before the game (a stand-in for starters). Key offensive players =
QB >= 50% of attempts, RB >= 30% of carries, WR/TE >= 15% of targets.

Out:  new  = no line this game, played the previous one (known after the game; sizes effects)
      prev = no line in the previous game only (streak 1; known before kickoff)

    python scripts/impact_multi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_select  # noqa: E402

from cfb.eval.bootstrap import paired_bootstrap  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.models import MarginModel, log_loss  # noqa: E402

DEF = ["DL", "LB", "DB"]


def team_counts(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["new"] = ((r["absent"] == 1) & (r["streak"] == 0)).astype(float)
    r["prev"] = (r["streak"] == 1).astype(float)
    r["prevany"] = (r["absent_prev"] == 1).astype(float)
    d = r[r["pos"].isin(DEF)].sort_values("sh_tkl", ascending=False)
    d["rank"] = d.groupby(["game_id", "team_id"]).cumcount()
    d = d[d["rank"] < 11]
    d["top3_play"] = d.groupby(["game_id", "team_id"])["sh_dplay"].rank(ascending=False, method="first") <= 3
    off = r[((r["pos"] == "QB") & (r["sh_pass"] >= 0.5)) | ((r["pos"] == "RB") & (r["sh_car"] >= 0.3)) |
            (r["pos"].isin(["WR", "TE"]) & (r["sh_rec"] >= 0.15))]
    cols = {}
    for k in ("new", "prev", "prevany"):
        cols[f"def_n_{k}"] = d[k]
        cols[f"def_tkl_{k}"] = d[k] * d["sh_tkl"]
        cols[f"def_play_{k}"] = d[k] * d["sh_dplay"]
        cols[f"def_top3_{k}"] = d[k] * d["top3_play"]
        for p in DEF:
            cols[f"def_{p}_{k}"] = d[k] * (d["pos"] == p)
    f = pd.DataFrame(cols)
    f[["game_id", "team_id"]] = d[["game_id", "team_id"]]
    f = f.groupby(["game_id", "team_id"]).sum()
    oc = {}
    for k in ("new", "prev", "prevany"):
        oc[f"off_n_{k}"] = off[k]
        for p in ("QB", "RB", "WR", "TE"):
            oc[f"off_{p}_{k}"] = off[k] * (off["pos"] == p)
    o = pd.DataFrame(oc)
    o[["game_id", "team_id"]] = off[["game_id", "team_id"]]
    o = o.groupby(["game_id", "team_id"]).sum()
    t = f.join(o, how="outer").fillna(0.0)
    for k in ("new", "prev", "prevany"):
        t[f"key_n_{k}"] = t[f"off_n_{k}"] + t[f"def_n_{k}"]
    return t.reset_index()


def ols(df, xs, y="resid"):
    Z = np.column_stack([np.ones(len(df))] + [df[x].to_numpy(float) for x in xs])
    yv = df[y].to_numpy(float)
    b, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    s2 = np.sum((yv - Z @ b) ** 2) / (len(yv) - Z.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Z.T @ Z)))
    return b, se


def fmt_rows(df, by, label):
    g = df.groupby(by)["resid"].agg(["mean", "std", "size"])
    print(f"  {label}")
    for k, row in g.iterrows():
        print(f"    {k!s:>5}: {row['mean']:+6.2f} (SE {row['std'] / np.sqrt(row['size']):.2f})  n {int(row['size']):5d}")


def main():
    base = pd.read_parquet("data/exploration/impact_base.parquet")
    roles = pd.read_parquet("data/exploration/impact_roles_ps.parquet")
    roles["sh_rec"] = roles["sh_tgt"]
    t = team_counts(roles)
    side = pd.concat([base.assign(team_id=base["home_id"], sgn=1.0), base.assign(team_id=base["away_id"], sgn=-1.0)])
    side["resid"] = side["sgn"] * (side["margin"] - side["pred"])
    side = side.merge(t, on=["game_id", "team_id"], how="left").fillna(0.0)
    h = side[side.season <= 2025].copy()
    print(f"team-games 2018-2025 {len(h):,}; residual vs model_v2, team's side, points")

    # ---- defense ---------------------------------------------------------------------------
    print("\nA. defense (11 regular defenders by tackle share)")
    h["def_n_new_c"] = h["def_n_new"].clip(upper=3).astype(int)
    fmt_rows(h, "def_n_new_c", "regular defenders new out (3 = 3+)")
    for xs in (["def_n_new"], ["def_DL_new", "def_LB_new", "def_DB_new"], ["def_tkl_new"], ["def_play_new"],
               ["def_top3_new"]):
        b, se = ols(h, xs)
        print("    " + "  ".join(f"{x} {bb:+.2f} ({ss:.2f})" for x, bb, ss in zip(xs, b[1:], se[1:])))
    h["def_n_prev_c"] = h["def_n_prev"].clip(upper=3).astype(int)
    fmt_rows(h, "def_n_prev_c", "regular defenders who missed only the last game (pregame)")
    b, se = ols(h, ["def_n_prev"])
    print(f"    def_n_prev {b[1]:+.2f} ({se[1]:.2f}) per defender")
    print(f"  mean regular defenders new out per team-game {h['def_n_new'].mean():.2f}; "
          f"missed only last game {h['def_n_prev'].mean():.2f}")

    # ---- several players out -----------------------------------------------------------------
    print("\nB. several key players out in the same game (new out; offense key + regular defenders)")
    h["key_c"] = h["key_n_new"].clip(upper=4).astype(int)
    fmt_rows(h, "key_c", "key players new out (4 = 4+)")
    h["off_c"] = h["off_n_new"].clip(upper=2).astype(int)
    fmt_rows(h, "off_c", "key offensive players new out (2 = 2+)")
    xs = ["off_QB_new", "off_RB_new", "off_WR_new", "off_TE_new", "def_n_new"]
    single = h[h["key_n_new"] <= 1]
    b, se = ols(single, xs)
    print("  additive fit on team-games with 0-1 key out: " +
          "  ".join(f"{x} {bb:+.2f} ({ss:.2f})" for x, bb, ss in zip(xs, b[1:], se[1:])))
    multi = h[h["key_n_new"] >= 2].copy()
    multi["pred_add"] = b[0] + sum(b[i + 1] * multi[x] for i, x in enumerate(xs))
    gap = multi["resid"] - multi["pred_add"]
    print(f"  team-games with 2+ key out: n {len(multi):,}; actual {multi['resid'].mean():+.2f} "
          f"(SE {multi['resid'].std() / np.sqrt(len(multi)):.2f}); additive prediction {multi['pred_add'].mean():+.2f}; "
          f"gap {gap.mean():+.2f} (SE {gap.std() / np.sqrt(len(multi)):.2f})")
    for lo, hi in ((2, 2), (3, 3), (4, 99)):
        m = multi[(multi["key_n_new"] >= lo) & (multi["key_n_new"] <= hi)]
        g = m["resid"] - m["pred_add"]
        print(f"    {lo}{'+' if hi == 99 else ''} out: n {len(m):4d} actual {m['resid'].mean():+6.2f} additive "
              f"{m['pred_add'].mean():+6.2f} gap {g.mean():+5.2f} (SE {g.std() / np.sqrt(max(len(m), 1)):.2f})")
    xo = ["off_QB_new", "off_RB_new", "off_WR_new", "off_TE_new"]
    b, se = ols(h[h["off_n_new"] <= 1], xo)
    print("  offense only, additive fit on team-games with 0-1 key offensive out: " +
          "  ".join(f"{x} {bb:+.2f} ({ss:.2f})" for x, bb, ss in zip(xo, b[1:], se[1:])))
    mo = h[h["off_n_new"] >= 2].copy()
    mo["pred_add"] = b[0] + sum(b[i + 1] * mo[x] for i, x in enumerate(xo))
    g = mo["resid"] - mo["pred_add"]
    print(f"  2+ key offensive out: n {len(mo)}; actual {mo['resid'].mean():+.2f}; additive {mo['pred_add'].mean():+.2f}; "
          f"gap {g.mean():+.2f} (SE {g.std() / np.sqrt(len(mo)):.2f}); with the QB among them: n {int((mo['off_QB_new'] > 0).sum())}, "
          f"actual {mo.loc[mo['off_QB_new'] > 0, 'resid'].mean():+.2f}; without: actual {mo.loc[mo['off_QB_new'] == 0, 'resid'].mean():+.2f}")
    h["off2"] = (h["off_n_new"] >= 2).astype(float)
    h["key3"] = (h["key_n_new"] >= 3).astype(float)
    b, se = ols(h, xs + ["off2", "key3"])
    print("  all team-games, additive + interaction terms: " +
          "  ".join(f"{x} {bb:+.2f} ({ss:.2f})" for x, bb, ss in zip(xs + ["off2", "key3"], b[1:], se[1:])))

    # ---- pregame log loss -----------------------------------------------------------------------
    model_select.LAST_SEASON = 2026
    X, _ = model_select.load()
    feats = [c for c in t.columns if c not in ("game_id", "team_id")]
    hh = X[["game_id", "home_id"]].merge(t, left_on=["game_id", "home_id"], right_on=["game_id", "team_id"], how="left")
    aa = X[["game_id", "away_id"]].merge(t, left_on=["game_id", "away_id"], right_on=["game_id", "team_id"], how="left")
    X = pd.concat([X, pd.DataFrame({f"d_m_{c}": hh[c].fillna(0).to_numpy() - aa[c].fillna(0).to_numpy() for c in feats},
                                   index=X.index)], axis=1)
    cols0 = list(C.MODEL_V2_COLUMNS)

    def sf(extra, seasons):
        preds = []
        for s in seasons:
            tr = X[(X["season"] >= C.TRAIN_START) & (X["season"] < s) & X["y"].notna()]
            te = X[(X["season"] == s) & X["in_eval"] & X["y"].notna()]
            m = MarginModel(cols0 + [f"d_m_{c}" for c in extra], lam=C.LAM).fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
            preds.append(pd.Series(m.predict_proba(te), index=te.index))
        return pd.concat(preds)

    S = list(range(2018, 2026))
    p0, q0 = sf([], S), sf([], [2026])
    d, d6 = X.loc[p0.index], X.loc[q0.index]
    cl = (d["season"].astype(str) + "_" + d["season_type"].astype(str) + "_" + d["week"].astype(str)).to_numpy()
    per0 = {s: log_loss(d.loc[d.season == s, "y"], p0[d.season == s]) for s in S}
    print(f"\nC. pregame: model_v2 2018-2025 {log_loss(d['y'], p0):.4f}, 2026 {log_loss(d6['y'], q0):.4f}")
    variants = {
        "def count, missed only last": ["def_n_prev"],
        "def tackle share, missed only last": ["def_tkl_prev"],
        "def count, missed last (any streak)": ["def_n_prevany"],
        "def by position, missed only last": ["def_DL_prev", "def_LB_prev", "def_DB_prev"],
        "key count (off + def), missed only last": ["key_n_prev"],
        "off count + def count, missed only last": ["off_n_prev", "def_n_prev"],
        "off by position + def count, missed only last": ["off_QB_prev", "off_RB_prev", "off_WR_prev", "off_TE_prev",
                                                          "def_n_prev"],
    }
    for lab, extra in variants.items():
        p, q = sf(extra, S), sf(extra, [2026])
        b = paired_bootstrap(d["y"].to_numpy(), p.loc[d.index].to_numpy(), p0.to_numpy(), clusters=cl)
        b6 = paired_bootstrap(d6["y"].to_numpy(), q.loc[d6.index].to_numpy(), q0.to_numpy(),
                              clusters=d6["week"].astype(str).to_numpy())
        better = sum(log_loss(d.loc[d.season == s, "y"], p[d.season == s]) < per0[s] for s in S)
        print(f"  + {lab:<46} 2018-25 {b['diff']:+.4f} (SE {b['se']:.4f}) better {better}/8   "
              f"2026 {b6['diff']:+.4f} (SE {b6['se']:.4f})")


if __name__ == "__main__":
    main()
