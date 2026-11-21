"""CFB Predictor: the local app (Streamlit). Start with `cfb-ui` (or the packaged .exe).

Reads the weekly report bundles (data/reports/<key>/bundle.json) for the slate, team pages,
rankings and performance, and the warehouse for on-demand matchups. Never calls the CFBD API.
"""
import streamlit as st

from cfb.ui import pages as P
from cfb.ui import sync_job

st.set_page_config(page_title="CFB Predictor", page_icon="🏈", layout="wide")

pages = [
    st.Page(P.page_week, title="This week", icon="📅", default=True),
    st.Page(P.page_results, title="Picks vs results", icon="✅"),
    st.Page(P.page_matchup, title="Matchup", icon="⚔️"),
    st.Page(P.page_teams, title="Teams", icon="🏟️"),
    st.Page(P.page_performance, title="Performance", icon="📈"),
    st.Page(P.page_data, title="Data", icon="🗄️"),
    st.Page(P.page_settings, title="Settings", icon="⚙️"),
]
if sync_job.JOB.take_fresh():  # a background sync switched to new data
    st.cache_data.clear()
    st.cache_resource.clear()
nav = st.navigation(pages)
P.sidebar_update()
nav.run()
