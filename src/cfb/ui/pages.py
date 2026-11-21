"""Pages of the local app. Each page is a function; app.py puts them in the navigation.
Kept apart from app.py so tests can run one page at a time (streamlit.testing)."""
from __future__ import annotations

import time as _time
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from cfb import settings
from cfb.ui import data as D
from cfb.ui import sync_job

PCT = st.column_config.NumberColumn(format="percent")


# ---- shared state -------------------------------------------------------------------------

def data_dir() -> Path:
    return D.resolve_data_dir()


@st.cache_data(show_spinner=False)
def _bundle(folder: str, mtime: float) -> dict:
    return D.load_bundle(Path(folder))


def current_bundle() -> tuple[dict | None, str | None]:
    reports = D.list_reports(data_dir())
    if not reports:
        return None, None
    keys = [k for k, _ in reports]
    chosen = st.sidebar.selectbox("Report", keys, index=0, key="report_key",
                                  format_func=lambda k: k.replace("_regular_", " week ").replace("_", " "))
    folder = dict(reports)[chosen]
    return _bundle(str(folder), (folder / "bundle.json").stat().st_mtime), chosen


@st.cache_resource(show_spinner="Loading the warehouse…")
def _warehouse(path: str):
    from cfb.core.asof import Warehouse
    return Warehouse(path)



@st.cache_data(ttl=3600, show_spinner=False)
def _update():
    from cfb.ui.update import update_available
    return update_available()


def sidebar_update():
    """A sidebar note when a newer build of the packaged app is on GitHub Releases, or a sync runs."""
    if sync_job.JOB.running:
        st.sidebar.caption("Database sync running… progress is on the Data page.")
    rel = _update()
    if rel:
        link = rel.get("installer_url") or rel.get("url")
        st.sidebar.info(f"Update available: {rel['tag']}. [Download the installer]({link}), run it, then restart the app.")


def no_data():
    st.warning(f"No reports found under `{data_dir()}`. Open **Data** to sync the database.")


# ---- pages --------------------------------------------------------------------------------

def page_week():
    bundle, _ = current_bundle()
    if bundle is None:
        return no_data()
    meta = bundle["meta"]
    st.title(meta.get("label", "This week"))
    st.caption(f"Model: {meta.get('primary_label', meta['primary'])} · data through "
               f"{meta.get('data_through_utc', '?')[:10]} · lines as of {meta.get('lines_asof_local', '?')}")
    df = D.games_frame(bundle)
    if df.empty:
        return st.info("No games in this report.")
    c1, c2, c3 = st.columns(3)
    fbs_only = c1.toggle("FBS vs FBS only", value=True)
    confs = sorted({c for cs in df["conferences"] for c in cs})
    conf = c2.multiselect("Conference", confs)
    min_edge = c3.slider("Min. win-prob edge vs market", 0.0, 0.3, 0.0, 0.01)
    view = df
    if fbs_only:
        view = view[view["fbs_vs_fbs"]]
    if conf:
        view = view[view["conferences"].map(lambda cs: bool(set(cs) & set(conf)))]
    if min_edge > 0:
        view = view[view["edge"].fillna(0).abs() >= min_edge]
    cols = ["kickoff", "away", "home", "pick", "p_pick", "edge", "model_spread", "market_spread", "ats_pick",
            "ats_edge", "pred_total", "market_total", "ou_pick", "ou_edge", "score", "pick_correct"]
    num = st.column_config.NumberColumn
    st.dataframe(view[cols], hide_index=True, width="stretch", column_config={
        "kickoff": "Kickoff", "away": "Away", "home": "Home", "pick": "Pick",
        "p_pick": num("Win prob", format="percent"),
        "edge": num("Edge vs market", format="%+.3f", help="Model minus market win probability for the pick"),
        "model_spread": num("Model spread", format="%+.1f", help="Home spread; negative = home favoured"),
        "market_spread": num("Market spread", format="%+.1f"),
        "ats_pick": "ATS pick", "ats_edge": num("ATS edge (pts)", format="%.1f"),
        "pred_total": num("Model total", format="%.1f"), "market_total": num("Market total", format="%.1f"),
        "ou_pick": "O/U pick", "ou_edge": num("O/U edge (pts)", format="%.1f"),
        "score": "Final", "pick_correct": "Pick right"})
    st.subheader("Why")
    options = {int(r.game_id): f"{r.away} at {r.home}" for r in view.itertuples()}
    if options:
        gid = st.selectbox("Game", list(options), format_func=options.get)
        game = D.game_by_id(bundle, gid)
        for r in (game or {}).get("reasons", []):
            st.write(f"- {r['text']} (effect {r['effect']:+.3f})")


@st.cache_data(show_spinner="Reading final scores")
def _scores(folder: str, key: str, stamp: str) -> tuple[dict, str | None]:
    """game_id -> (home, away) points: later reports' last-week scores, then the warehouse."""
    from cfb.report import results as R

    scores = R.results_from_bundles(R.later_bundles(Path(folder) / "reports", key))
    note = None
    try:
        scores.update(R.results_from_outcomes(_warehouse(folder).outcomes()))
    except Exception as exc:  # no core tables in this folder
        note = f"Warehouse scores unavailable ({exc}); using later reports only."
    return scores, note


def page_results():
    from cfb.report import results as R

    bundle, key = current_bundle()
    if bundle is None:
        return no_data()
    meta = bundle["meta"]
    st.title(f"{meta.get('label', key)}: picks vs results")
    ver = D.data_version(data_dir()) or {}
    scores, note = _scores(str(data_dir()), key, str(ver.get("version", "")))
    graded = R.grade(bundle, scores)
    c = graded["counts"]
    st.caption(f"Picks as published by the weekly run of {meta.get('created_local', '?')} (lines as of "
               f"{meta.get('lines_asof_local', '?')}), not recomputed. Primary model: {graded['primary_label']}. "
               f"Final scores: {c['final']} of {c['games']} games; {c['pending']} pending (scores arrive with the "
               f"next database sync).")
    if note:
        st.info(note)
    m = graded["models"].get(graded["primary"]) or {}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Picks right (FBS vs FBS)", f"{graded['picks']['correct']}/{graded['picks']['n']}")
    k2.metric("Log loss", "—" if m.get("log_loss") is None else f"{m['log_loss']:.4f}")
    if graded["phase2"]:
        for col, name, rec in ((k3, "Spread vs line (W-L-P)", graded["ats"]), (k4, "Total vs line (W-L-P)", graded["ou"])):
            col.metric(name, f"{rec['won']}-{rec['lost']}-{rec['push']}" if rec else "—")
    if graded["models"]:
        st.dataframe(pd.DataFrame(graded["models"].values()).drop(columns=["primary"]), hide_index=True,
                     column_config={"accuracy": PCT, "label": "Model", "n": "Games"})
    df = R.frame(graded)
    c1, c2 = st.columns(2)
    if c1.toggle("FBS vs FBS only", value=True, key="res_fbs"):
        df = df[df["fbs_vs_fbs"]]
    if c2.toggle("Final scores only", key="res_final"):
        df = df[df["final"]]
    cols = ["kickoff", "away", "home", "pick", "p_pick", "market_p_pick", "score", "pick_correct", "log_loss"]
    if graded["phase2"]:
        cols += ["model_spread", "home_line", "ats_pick", "spread_line", "ats_result", "pred_total", "total_line", "ou_pick",
                 "ou_result"]
    num = st.column_config.NumberColumn
    st.dataframe(df[cols], hide_index=True, width="stretch", column_config={
        "kickoff": "Kickoff", "away": "Away", "home": "Home", "pick": "Pick",
        "p_pick": num("Win prob", format="percent"), "market_p_pick": num("Market on pick", format="percent"),
        "score": "Final", "pick_correct": "Pick right", "log_loss": num("Log loss", format="%.3f"),
        "model_spread": num("Model spread", format="%+.1f", help="Home spread; negative = home favoured"),
        "home_line": num("Home line at pick", format="%+.1f"),
        "spread_line": num("Pick's line", format="%+.1f"), "ats_pick": "ATS pick", "ats_result": "ATS",
        "pred_total": num("Model total", format="%.1f"), "total_line": num("Total line", format="%.1f"),
        "ou_pick": "O/U pick", "ou_result": "O/U"})


def page_matchup():
    from cfb.model import config as C
    from cfb.predict import matchup

    st.title("Matchup")
    bundle, _ = current_bundle()
    names = [n for _, n in D.teams_list(bundle)] if bundle else []
    c1, c2 = st.columns(2)
    if names:
        home = c1.selectbox("Home", names, index=0)
        away = c2.selectbox("Away", names, index=min(1, len(names) - 1))
    else:
        home = c1.text_input("Home")
        away = c2.text_input("Away")
    c3, c4, c5 = st.columns(3)
    neutral = c3.toggle("Neutral site")
    day = c4.date_input("As of", value=date.today())
    models = sorted(C.MODELS)
    default = bundle["meta"]["primary"] if bundle and bundle["meta"]["primary"] in models else models[-1]
    model_id = c5.selectbox("Model", models, index=models.index(default))
    with st.expander("QB out (injury override)"):
        q1, q2 = st.columns(2)
        qb_team = q1.selectbox("Team", ["", home, away])
        qb_repl = q2.text_input("Replacement QB (blank = next QB)")
    if st.button("Predict", type="primary", disabled=not (home and away) or home == away):
        cutoff = datetime.combine(day, time(8, 0), tzinfo=timezone.utc)
        qb_out = {qb_team: qb_repl or "out"} if qb_team else None
        with st.spinner("Computing features at this date (about 15 s for model_v2, 2 s for model_v1)…"):
            try:
                res = matchup(_warehouse(str(data_dir())), home, away, neutral=neutral, cutoff=cutoff,
                              model_id=model_id, qb_out=qb_out)
            except (SystemExit, Exception) as exc:  # matchup raises SystemExit on an unknown team
                return st.error(f"Could not predict: {exc}")
        p = res["p_home_win"]
        winner, pw = (res["home"], p) if p >= 0.5 else (res["away"], 1 - p)
        st.metric(f"{winner} wins", f"{pw:.1%}")
        st.caption(f"{res['away']} at {res['home']}{' (neutral)' if neutral else ''} · {model_id} · as of {res['cutoff_utc']}")
        for r in res.get("reasons", []):
            st.write(f"- {r['text']} (effect {r['effect']:+.3f})")
        for q in res.get("qb_out", []):
            st.info(f"QB override: {q}")


def page_teams():
    bundle, _ = current_bundle()
    if bundle is None:
        return no_data()
    st.title("Teams and rankings")
    tab_rank, tab_team = st.tabs(["Rankings vs polls", "Team"])
    with tab_rank:
        rk = bundle.get("rankings") or {}
        if rk:
            st.caption(f"{rk['source']['label']}: {rk['source']['note']}")
        st.dataframe(D.rankings_frame(bundle), hide_index=True, width="stretch")
    with tab_team:
        teams = D.teams_list(bundle)
        if not teams:
            return st.info("No team data in this report.")
        tid = st.selectbox("Team", [t for t, _ in teams], format_func=dict(teams).get)
        t = bundle["teams"][str(tid)]
        st.subheader(f"{t['name']} ({t.get('record', '')}, {t.get('conference', '')})")
        st.dataframe(D.team_games_frame(bundle, tid), hide_index=True, width="stretch",
                     column_config={"p_win": PCT})


def page_performance():
    bundle, _ = current_bundle()
    if bundle is None:
        return no_data()
    st.title("Model performance")
    st.caption((bundle.get("season") or {}).get("scope", ""))
    st.dataframe(D.own_metrics_frame(bundle).drop(columns=["model_id"]), hide_index=True,
                 column_config={"accuracy": PCT})
    st.subheader("Season-to-date log loss by week (lower is better)")
    wk = D.weekly_frame(bundle, "cumulative")
    if not wk.empty:
        st.line_chart(wk)
    st.subheader("Calibration")
    primary = bundle["meta"]["primary"]
    cal = D.calibration_frame(bundle, primary)
    if not cal.empty:
        st.scatter_chart(cal, x="p_mean", y="win_rate", size="n")
    pts = bundle.get("season_points") or {}
    if pts:
        st.subheader("Spread and total")
        st.json(pts, expanded=False)


def page_data():
    st.title("Data")
    d = data_dir()
    st.write(f"Data folder: `{d}`")
    ver = D.data_version(d)
    if ver:
        st.write(f"Database version **{ver.get('version')}**, published {ver.get('created_utc')}")
        quota = ver.get("cfbd_quota")
        if quota:
            st.caption(f"CFBD API: {quota.get('remaining_calls')} of {quota.get('monthly_limit')} calls left this month")
    try:
        from cfb import warehouse_sync
    except ImportError:
        warehouse_sync = None
    if warehouse_sync is None:
        st.info("Sync from the cloud database is not in this version yet.")
    else:
        _sync_controls(warehouse_sync)


def _sync_controls(warehouse_sync) -> None:
    job = sync_job.JOB
    if job.running:
        return _sync_progress()
    if st.button("Sync now", type="primary"):
        base = settings.warehouse_dir()

        def run(log):
            store = warehouse_sync.open_store(warehouse_sync.r2_settings())
            return warehouse_sync.sync(base, store, log=log)

        job.start(run)
        st.rerun()
    snap = job.snapshot()
    if snap.state == "done":
        r = snap.result
        if getattr(r, "changed", False):
            st.success(f"Synced {r.version}: downloaded {r.downloaded_files} files "
                       f"({r.downloaded_bytes / 1e6:.1f} MB), reused {r.linked_files}.")
        else:
            st.success(f"Up to date ({getattr(r, 'version', '?')}).")
    elif snap.state == "failed":
        exc = snap.error
        if isinstance(exc, warehouse_sync.WarehouseError):
            st.error(f"Sync failed: {exc}")
        else:
            st.error(f"Sync failed: {exc}. Check the R2 keys on the Settings page "
                     "(r2_account_id, r2_access_key_id, r2_secret_access_key).")
    if snap.lines and snap.state != "idle":
        st.code("\n".join(snap.lines[-12:]))


@st.fragment(run_every=1.0)
def _sync_progress() -> None:
    snap = sync_job.JOB.snapshot()
    if not snap.running:
        st.rerun()  # whole page: show the result and the new data
    secs = int(_time.time() - (snap.started or _time.time()))
    st.info(f"Syncing… {secs} s. You can use other pages meanwhile; the sync keeps going.")
    if snap.lines:
        st.code("\n".join(snap.lines[-12:]))


def page_settings():
    st.title("Settings")
    st.write(f"Config file: `{settings.config_path()}`")
    st.write(f"Warehouse folder: `{settings.warehouse_dir()}`")
    st.subheader("Keys")
    st.caption("Stored in Windows Credential Manager. Values are never shown. To sync the database the PC "
               "needs r2_account_id, r2_access_key_id and r2_secret_access_key (the read-only R2 token); "
               "r2_bucket defaults to cfb-warehouse. The CFBD key is not needed on the PC.")
    rows = [{"key": n, "set": settings.secret(n) is not None} for n in settings.SECRET_ENV]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.form("set_key", clear_on_submit=True):
        name = st.selectbox("Key", list(settings.SECRET_ENV))
        value = st.text_input("Value", type="password")
        if st.form_submit_button("Save") and value:
            try:
                settings.set_secret(name, value.strip())
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved {name}.")
