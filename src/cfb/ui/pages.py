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
    elif not _connected():
        st.sidebar.caption("Cloudflare not connected: save the R2 keys once on the Settings page.")
    rel = _update()
    if rel:
        link = rel.get("installer_url") or rel.get("url")
        st.sidebar.info(f"Update available: {rel['tag']}. [Download the installer]({link}), run it, then restart the app.")


@st.cache_data(ttl=30, show_spinner=False)
def _connected() -> bool:
    from cfb.ui import connection
    return connection.status().configured


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
        job.start(sync_job.run_sync)
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
            st.error(f"Sync failed: {exc}. Check the Cloudflare connection on the Settings page.")
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
    from cfb.ui import connection

    st.title("Settings")
    st.subheader("Cloudflare connection")
    stat = connection.status()
    if stat.configured:
        st.success(f"Connected: the R2 keys are saved in {stat.where()} and load on every launch.")
    else:
        st.warning("Not connected. Save the R2 keys once; the app loads them on every launch after that.")
    st.caption("From the Cloudflare dashboard: R2 > Manage API tokens > the read-only token. Paste its Access Key ID, "
               "Secret Access Key, and either the account ID or the S3 endpoint shown with it "
               "(https://<account id>.r2.cloudflarestorage.com). Leave a field blank to keep the saved value. "
               "Values are never shown.")
    with st.form("r2_connect", clear_on_submit=True):
        acc = st.text_input("Account ID or S3 endpoint", placeholder="saved" if stat.sources.get("r2_account_id") else "")
        kid = st.text_input("Access Key ID", type="password",
                            placeholder="saved" if stat.sources.get("r2_access_key_id") else "")
        sec = st.text_input("Secret Access Key", type="password",
                            placeholder="saved" if stat.sources.get("r2_secret_access_key") else "")
        bucket = st.text_input("Bucket (optional)", placeholder=settings.secret("r2_bucket") or "cfb-warehouse")
        submitted = st.form_submit_button("Save and test", type="primary")
    if submitted:
        try:
            saved = connection.save(acc, kid, sec, bucket)
        except Exception as exc:
            st.error(str(exc))
        else:
            _connected.clear()
            if saved:
                where = sorted({"Windows Credential Manager" if v == "keyring" else "the encrypted keys file"
                                for v in saved.values()})
                st.info(f"Saved {', '.join(saved)} to {', '.join(where)}.")
            _check_connection(connection)
    c1, c2 = st.columns(2)
    if c1.button("Test connection", disabled=not stat.configured):
        _check_connection(connection)
    if c2.button("Forget the R2 keys", disabled=not any(stat.sources.values())):
        connection.forget()
        _connected.clear()
        st.rerun()
    auto = st.toggle("Sync the database when the app starts", value=bool(connection.prefs().get("auto_sync", True)),
                     help="Downloads only the files that changed. Needs the connection above.")
    if auto != bool(connection.prefs().get("auto_sync", True)):
        connection.set_pref("auto_sync", auto)

    st.subheader("Files")
    st.write(f"Config file: `{settings.config_path()}`")
    st.write(f"Warehouse folder: `{settings.warehouse_dir()}`")
    backend = settings.keyring_backend()
    st.caption(f"Key store: {('Windows Credential Manager (' + backend + ')') if backend else 'no OS keyring'}; "
               f"fallback file `{settings.secrets_file()}` (encrypted for this Windows user).")

    st.subheader("Other keys")
    st.caption("The CFBD and Sportradar keys are not needed on the PC.")
    with st.form("set_key", clear_on_submit=True):
        name = st.selectbox("Key", list(settings.SECRET_ENV))
        value = st.text_input("Value", type="password")
        if st.form_submit_button("Save") and value:
            try:
                where = settings.set_secret(name, value.strip())
            except Exception as exc:
                st.error(str(exc))
            else:
                _connected.clear()
                st.success(f"Saved {name} ({'Windows Credential Manager' if where == 'keyring' else 'encrypted keys file'}).")
    labels = {"keyring": "Credential Manager", "file": "keys file", "environment": "environment"}
    rows = [{"key": n, "saved in": labels.get(settings.secret_source(n), "")} for n in settings.SECRET_ENV]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _check_connection(connection) -> None:
    with st.spinner("Checking the bucket…"):
        try:
            version = connection.check()
        except Exception as exc:
            st.error(f"Connection failed: {exc}")
        else:
            st.success(f"Connection works. Cloud database version: {version}.")


# ---- players and injuries (test model) ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading player availability…", max_entries=2)
def _week_model(path: str, mtime: float):
    from cfb.ui.availability import load_model
    return load_model(Path(path))


SETTING_TO_STATE = {"Listing": "listing", "Available": "available", "Out": "out"}


def page_players():
    from cfb.ui import availability as AV
    from cfb.ui import connection

    st.title("Players and injuries")
    pkgs = AV.local_packages()
    if not pkgs:
        st.info("No player-availability data on this PC yet. It downloads with the database sync.")
        if connection.status().configured and not sync_job.JOB.running:
            if st.button("Sync now", type="primary"):
                sync_job.JOB.start(sync_job.run_sync)
                st.rerun()
        elif sync_job.JOB.running:
            st.caption("A sync is running; open this page again when it finishes.")
        else:
            st.caption("Connect Cloudflare on the Settings page first.")
        return
    keys = [p.stem for p in pkgs]
    key = st.sidebar.selectbox("Injury week", keys, index=0,
                               format_func=lambda k: k.replace("_regular_", " week ").replace("_", " ")) \
        if len(keys) > 1 else keys[0]
    path = dict(zip(keys, pkgs))[key]
    try:
        m = _week_model(str(path), path.stat().st_mtime)
    except Exception as exc:
        return st.error(f"Could not read {path.name}: {exc}")
    meta = m.meta
    src = meta.get("sources") or {}
    st.caption(f"{meta.get('label', key)} · injury list {src.get('listings', '?')} · depth charts "
               f"{src.get('charts', '?')} · built {str(meta.get('created_utc', '?'))[:16].replace('T', ' ')} UTC")
    store = AV.Overrides()
    ov = store.week(key)
    tab_picks, tab_mark = st.tabs(["Picks", "Mark players"])
    with tab_mark:
        _mark_players(m, store, key, ov)
    ov = store.week(key)
    with tab_picks:
        _availability_picks(m, ov)
    n = sum(len(v) for v in ov.values())
    if n:
        st.subheader(f"Your changes this week ({n})")
        names = dict(m.week_teams())
        rows = [{"team": names.get(t, t), "player": m.names.get(a, a), "setting": s.title()}
                for t, d in ov.items() for a, s in d.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if st.button("Clear all my changes for this week"):
            store.clear(key)
            st.rerun()


def _availability_picks(m, ov) -> None:
    g = m.picks(ov)
    st.caption("Test model: model_v2 plus the depth-chart injury term. A starter who is out costs his value over the "
               "next man up on this week's chart, times P(out). \"Listed\" uses the injury list as is; "
               "\"Yours\" applies your settings from Mark players.")
    c1, c2 = st.columns(2)
    p4 = c1.toggle("P4 + Notre Dame games only", value=True)
    mine = c2.toggle("Only games with your changes", value=False)
    view = g
    if p4 and "p4" in view:
        view = view[view["p4"].astype(bool)]
    if mine:
        view = view[view["changed"]]
    view = view.sort_values("start_time_utc")
    view = view.assign(kickoff=pd.to_datetime(view["start_time_utc"], utc=True).dt.tz_convert("America/New_York")
                       .dt.strftime("%a %m-%d %I:%M %p"),
                       game=view["away_team"] + " @ " + view["home_team"] + view["neutral"].map({True: " (N)", False: ""}))
    num = st.column_config.NumberColumn
    st.dataframe(view[["kickoff", "game", "p_v2", "p_dc", "p_mine", "pick", "p_pick", "adj_mine", "flip_vs_listed",
                       "home_out", "away_out"]], hide_index=True, width="stretch", column_config={
        "kickoff": "Kickoff (ET)", "game": "Game",
        "p_v2": num("model_v2 P(home)", format="percent"),
        "p_dc": num("Listed P(home)", format="percent"),
        "p_mine": num("Yours P(home)", format="percent"),
        "pick": "Your pick", "p_pick": num("Win prob", format="percent"),
        "adj_mine": num("Injury adj (pts)", format="%+.1f", help="Away cost minus home cost; negative favours the away team"),
        "flip_vs_listed": "Pick changed", "home_out": "Home out (listed)", "away_out": "Away out (listed)"})


def _mark_players(m, store, key: str, ov: dict) -> None:
    teams = m.week_teams()
    if not teams:
        return st.info("No games in this package.")
    tid = st.selectbox("Team", [t for t, _ in teams], format_func=dict(teams).get, key="av_team")
    mine = ov.get(tid, {})
    df = m.players(tid, mine)
    if df.empty:
        return st.info("No QB, RB, WR or TE data for this team.")
    st.caption("Set a player to Available or Out to override the injury list; Listing follows the list. Only QB, RB, "
               "WR and TE starters change the pick (the model found no measurable effect for OL or defense).")
    ed_key = f"av_ed_{key}_{tid}"
    num = st.column_config.NumberColumn
    cols = ["athlete_id", "player", "grp", "setting", "listing", "p_listed", "starter", "cost_now", "value_pts",
            "next_up", "p_out_now", "chart", "games", "touches"]
    edited = st.data_editor(df[cols], hide_index=True, width="stretch", key=ed_key,
                            disabled=[c for c in cols if c != "setting"], column_config={
        "athlete_id": None, "player": "Player", "grp": "Pos", "chart": "Chart spot", "games": "Games",
        "touches": st.column_config.NumberColumn("Touches", help="Dropbacks (QB), carries (RB), targets (WR/TE)"),
        "listing": "Injury list", "p_listed": num("P(out) listed", format="%.2f"),
        "setting": st.column_config.SelectboxColumn("Your setting", options=list(SETTING_TO_STATE), required=True),
        "starter": "Starter", "value_pts": num("Value over next (pts)", format="%.1f"),
        "p_out_now": num("P(out) used", format="%.2f"), "cost_now": num("Cost (pts)", format="%.1f"),
        "next_up": "Next man up"})
    new = dict(zip(edited["athlete_id"].astype(int), edited["setting"]))
    old = dict(zip(df["athlete_id"].astype(int), df["setting"]))
    if new != old:
        states = {a: SETTING_TO_STATE[s] for a, s in new.items()}
        store.set_team(key, tid, states, {a: m.names.get(a) for a in states})
        st.session_state.pop(ed_key, None)
        st.rerun()
    res, base = m.team(tid, mine), m.team(tid, {})
    g = m.picks(ov)  # both teams' settings
    row = g[(g["home_id"] == tid) | (g["away_id"] == tid)]
    if len(row):
        r = row.iloc[0]
        st.write(f"**{dict(teams).get(tid)}** injury cost this game: **{res.cost:.1f} pts** "
                 f"(injury list as is: {base.cost:.1f}). Game: {r['away_team']} @ {r['home_team']}. "
                 f"Your pick **{r['pick']}** {r['p_pick']:.1%}. Chance {r['home_team']} wins: yours {r['p_mine']:.1%}, "
                 f"injury list as is {r['p_dc']:.1%}, model_v2 without injuries {r['p_v2']:.1%}.")
    if mine and st.button("Reset this team to the injury list"):
        store.set_team(key, tid, {})
        st.session_state.pop(ed_key, None)
        st.rerun()
