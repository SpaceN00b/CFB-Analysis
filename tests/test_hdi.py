"""cfb.hdi: parse the current reports and write them into the archive rows."""
from __future__ import annotations

import pandas as pd

from cfb import hdi


def pub(report="Update 1"):
    side = lambda team, rows: {"teamName": team, "rows": [{"name": n, "status": s, "exemptStatus": "NonExempt"}
                                                          for n, s in rows]}
    return {"10003242": {"ReportType": report, "conferenceTimeZone": "CT", "publishDate": "2026-11-19",
                         "postedTime": "19:10:00", "footer": {"date": "2026-11-21", "time": "11:00:00"},
                         "games": [side("Vanderbilt", [("QB #1 Blaze Berlowitz", "Out"), ("WR #0 Junior Sherrill", "Available")]),
                                   side("Florida", [("DB #3 Braydon Lee", "Questionable"), ("RB #2 Al Back", "Available"),
                                                    ("WR #13 Kayleb Barnett", "Exempt")])]},
            "pending": "Report Pending"}


ARCH = {"data": [
    {"Team": "Florida", "Player": "Braydon Lee", "Number": "#3", "Week": "Sat, 21 Nov 2026 00:00:00 GMT", "WeekNum": 12,
     "Opponent": "vs. Vanderbilt", "Initial": "Out", "Update 1": "-", "Update 2": "-", "Game Day": "-"},
    {"Team": "Florida", "Player": "Al Back", "Number": "#2", "Week": "Sat, 21 Nov 2026 00:00:00 GMT", "WeekNum": 12,
     "Opponent": "vs. Vanderbilt", "Initial": "Doubtful", "Update 1": "-", "Update 2": "-", "Game Day": "-"},
]}


def test_publish_rows_and_times():
    r = hdi.publish_rows(pub(), "SEC")
    assert len(r) == 5 and set(r["team"]) == {"Vanderbilt", "Florida"}
    q = r[r["player"] == "Blaze Berlowitz"].iloc[0]
    assert (q["pos"], q["number"], q["status"], q["opponent"], q["home"]) == ("QB", "1", "Out", "Florida", False)
    assert q["kickoff_utc"] == pd.Timestamp("2026-11-21T17:00:00Z")   # 11:00 CT
    assert q["posted_utc"] == pd.Timestamp("2026-11-20T01:10:00Z")    # 7:10 PM CT
    rep = hdi.reports(pub(), "SEC")
    assert len(rep) == 1 and rep.iloc[0]["conference"] == "SEC" and rep.iloc[0]["teams"] == "Vanderbilt vs Florida"


def test_merged_archive_overlays_current_report():
    rows = hdi.merged_archive(ARCH, pub(), "SEC")["data"]
    by = {r["Player"]: r for r in rows}
    assert by["Braydon Lee"]["Update 1"] == "Questionable" and by["Braydon Lee"]["Initial"] == "Out"
    assert by["Al Back"]["Update 1"] == "Available"          # listed before, cleared now
    assert by["Blaze Berlowitz"]["Update 1"] == "Out"        # new listing gets a row
    assert by["Blaze Berlowitz"]["Opponent"] == "at Florida" and by["Blaze Berlowitz"]["WeekNum"] == 12
    assert "Junior Sherrill" not in by and "Kayleb Barnett" not in by
    assert len(ARCH["data"]) == 2 and ARCH["data"][0]["Update 1"] == "-"   # input untouched


def test_empty_inputs():
    assert hdi.publish_rows(None, "SEC").empty and hdi.reports({}, "SEC").empty
    assert hdi.merged_archive(None, None, "SEC") == {"data": []}
