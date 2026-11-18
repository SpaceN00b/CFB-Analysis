"""playtext: structured fields parsed from the play text in CFBD's `plays` table.

CFBD's play text is ESPN's play-by-play text. It comes in two formats:

    detailed   the stat crew's text. Every play from 2025 week 9 on, and some penalty
               plays before that. Example:
               "(13:24) No Huddle-Shotgun #13 A.Simmons pass complete short middle to
               #0 N.Montgomery caught at MIZ27, for 4 yards loss to the MIZ27 (#19 J.Williams)"
    condensed  ESPN's rewritten text, 2014 to 2025 week 8. Example:
               "Drake Lindsey pass complete to Frank Bierman for 5 yds to the MINN 30"

Fields by format (NA where the text does not say):

    field                       detailed   condensed
    shotgun, no_huddle          yes        no
    run_dir (left/middle/right) yes        no
    pass_depth (short/deep)     yes        no
    pass_dir (left/middle/right) yes       no
    air_yards, yac              yes        no
    tacklers                    yes        no
    passer, rusher, target      yes        yes (full names, no jersey)
    sacker, interceptor,
    pass breakup                yes        yes
    QB hurry, forced fumble     yes        forced fumble only
    kneel, spike                yes        kneel only
    penalties                   yes        yes

Outputs (all keyed by game_id, play_id):

    play_detail    one row per play.
    play_players   one row per named player per role. `athlete_id` comes from the
                   season's roster: same team and jersey with a matching last name, else
                   a unique match on team, last name and first initial (detailed) or full
                   name (condensed).
    play_penalties one row per penalty called on the play.

air_yards is the distance from the line of scrimmage (CFBD yardsToGoal) to the catch
spot ("caught at"), the spot of an incomplete pass ("thrown to") or the interception
spot. The spot's team abbreviation is mapped to a team per game by checking the end
spot of each rush and pass against yardsToGoal and the yards in the text. yac is the
text's yards on a completion minus air_yards.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- patterns
_SUFFIX = r"(?:Jr|Sr|II|III|IV|V)\.?"
_UP = "A-Z\u00c0-\u00d6\u00d8-\u00de"  # capital letters, Latin-1 accents included
_LET = r"(?:[^\W\d_]|['\u2019\.\-])"  # letters (any case, accents) and name punctuation
_STOP = r"(?!(?:QB|PENALTY|TOUCHDOWN|NO|End|Touchback|SAFETY|Kneel|TURNOVER)\b)"
_TOKEN = rf"[{_UP}]{_LET}*"
# "#13 A.Simmons", "#7 C.Del Rio-Wilson", "#0 R.Vick, Jr.", "#5 R. Tisdale Jr."
HASH_NAME = (rf"#\d{{1,2}}\s+[{_UP}]{_LET}*?\.\s?{_TOKEN}(?:[ ]{_STOP}{_TOKEN})*"
             rf"(?:,\s{_SUFFIX})?(?: (?:ll|lll|ii|iii|iv)\b)?")
# "Sayin,Julian", "Logan Jr.,Chauncey", "Siemian, Trevor"
LAST_FIRST = rf"[{_UP}]{_LET}+?(?: [{_UP}]{_LET}+)?(?: {_SUFFIX})?,\s?[{_UP}]{_LET}*"
PLAYER = rf"(?:{HASH_NAME}|{LAST_FIRST})"
# team as the actor: "Troy rush", "Kneel down by Georgia State at"
TEAM_ACTOR = r"[A-Z][A-Za-z&'\.\-]*(?: [A-Z][A-Za-z&'\.\-]*){0,3}"
SPOT = r"[A-Za-z&][A-Za-z&\. ]{0,9}?\s?\d{1,2}\b"

_CLOCK = re.compile(r"^\(\s*(\d{1,2}):(\d{2})\s*\)\s*")
_FORMATION = re.compile(r"^((?:No Huddle-Shotgun|No Huddle|Shotgun|\[NH\]|\[SG\])[\s-]*)+")
_YARDS = r"(?:for (?:(?P<yds>\d+) yards?(?: (?P<gl>gain|loss))?|(?P<nogain>no gain)))"

_D_RUSH = re.compile(rf"^(?:(?P<p>{PLAYER}|{TEAM_ACTOR}) )?\s*rush(?: (?P<dir>left|right|middle))? {_YARDS}")
_D_PASS = re.compile(
    rf"^(?:(?P<p>{PLAYER}|{TEAM_ACTOR}) )?\s*pass (?P<res>complete|incomplete)"
    rf"(?: (?P<depth>short|deep))?(?: (?P<dir>left|right|middle))?(?: to (?P<t>{PLAYER}))?"
)
_D_INT = re.compile(rf"^(?P<p>{PLAYER}) pass intercepted by (?P<i>{PLAYER}) at (?P<spot>{SPOT})")
_D_SACK = re.compile(rf"^(?P<p>{PLAYER}|{TEAM_ACTOR}) sacked for loss of (?P<yds>\d+) yards?")
_D_KNEEL = re.compile(r"^Kneel down(?: by (?P<p>.+?))? at ")
_CAUGHT = re.compile(rf"caught at (?P<spot>{SPOT})")
_THROWN = re.compile(rf"thrown to (?P<spot>{SPOT})")
_PASS_YDS = re.compile(rf", {_YARDS}")
_END_SPOT_TACKLE = re.compile(rf"to the (?P<spot>{SPOT})(?:\s*\((?P<tk>[^()]*)\))?")
_BREAKUP = re.compile(rf"broken up by (?P<p>{PLAYER})")
_HURRY = re.compile(rf"QB hurried by (?P<a>{PLAYER})(?: and (?P<b>{PLAYER}))?")
_FUMBLE = re.compile(rf"fumbled? by (?P<p>{PLAYER}|{TEAM_ACTOR})(?: at (?P<spot>{SPOT}))?(?: forced by (?P<f>{PLAYER}))?")
_IN_PARENS = re.compile(PLAYER)

# condensed
_C_NAME = r"[A-Z][A-Za-z'\.\-]*(?: [A-Za-z'\.\-]+){0,3}?"
_C_PASS = re.compile(rf"^(?P<p>{_C_NAME}) pass(?: (?P<res>complete|incomplete))?(?: to (?P<t>{_C_NAME})(?=,| for |$| [A-Z]{{2,}}| broken))?")
_C_INT = re.compile(rf"^(?P<p>{_C_NAME}) pass intercepted(?:,? (?P<i>{_C_NAME}) return)?")
_C_RUSH = re.compile(rf"^(?P<p>{_C_NAME}) run for")
# scoring summaries: "Maurki James 1 Yd Run (Nathanial Eichner Kick)", "Cooper Barkate 9 Yd pass from Darian Mensah (...)"
_C_TD_RUN = re.compile(rf"^(?P<p>{_C_NAME}) (?P<yds>\d+) Yd Run\b")
_C_TD_PASS = re.compile(rf"^(?P<t>{_C_NAME}) (?P<yds>\d+) Yd pass from (?P<p>{_C_NAME})(?= \(|$)")
_C_SACK = re.compile(rf"^(?P<p>{_C_NAME}) sacked by (?P<a>{_C_NAME})(?: and (?P<b>{_C_NAME}))?(?= for)")
_C_BREAKUP = re.compile(rf"broken up by (?P<p>{_C_NAME})(?=,|$| [A-Z]{{2,}})")
_C_FUMBLE = re.compile(rf"(?P<p>{_C_NAME}) fumbled(?:, forced by (?P<f>{_C_NAME}),)?")
_C_KNEEL = re.compile(r"\bkneel", re.I)
_C_YDS = re.compile(r"for (?:(?P<yds>\d+) yds?|a loss of (?P<loss>\d+) yards?|(?P<nogain>no gain))")

# penalties
_PEN_SPLIT = re.compile(r"\bPENALTY\b|\b[A-Z][A-Z\.& ]+ Penalty, ")
_PEN_YDS = re.compile(rf"(?P<yds>\d+) yards? from (?:the )?(?P<frm>{SPOT}) to (?:the )?(?P<to>{SPOT})")
_PEN_PLAYER = re.compile(r"\((?P<p>[#A-Z][^()]*[A-Za-z]{2,}[^()]*)\)")
_PEN_CODE = re.compile(r"^[A-Z]{2,4}: ")
# "on CMC-R.Blum, Offensive Holding, 1 yards, accepted. No Play."
_PEN_ON = re.compile(r"^on (?P<team>[A-Za-z&\. ]+?)-(?P<p>[^,]+), (?P<type>[^,]+?)(?: \(\d+ Yards?\))?, "
                     r"(?P<yds>\d+) yards?(?:, (?P<status>accepted|declined))?")
_PEN_ON_PLAYER = re.compile(rf" on (?P<p>{PLAYER})")

PASS_DIRS = ("left", "middle", "right")

DETAIL_COLUMNS = [
    "game_id", "play_id", "season", "season_type", "week", "text_format",
    "shotgun", "no_huddle", "action", "run_dir", "pass_depth", "pass_dir",
    "completed", "intercepted", "kneel", "spike", "text_yards", "air_yards", "yac",
    "n_tacklers", "broken_up", "qb_hurried", "fumbled", "penalties", "no_play",
]
PLAYER_COLUMNS = [
    "game_id", "play_id", "season", "role", "name", "jersey", "side", "team_id",
    "athlete_id", "match",
]
PENALTY_COLUMNS = [
    "game_id", "play_id", "season", "n", "team_text", "side", "team_id", "penalty",
    "status", "yards", "no_play", "name", "jersey", "athlete_id", "match",
]
OFFENSE_ROLES = ("passer", "rusher", "target", "fumbler")
DEFENSE_ROLES = ("tackler", "sacker", "interceptor", "breakup", "hurry", "forced_fumble")


# --------------------------------------------------------------------------- helpers
_DETAILED_HINT = re.compile(r"#\d{1,2}\s+[A-Z]|^(?:No Huddle|Shotgun|\[NH\]|\[SG\])|^[A-Z][A-Za-z'\-]+,[A-Z]")


def text_format(text: str | None) -> str:
    """'detailed' (stat crew text), 'condensed' (ESPN's rewrite) or 'none'."""
    if not text:
        return "none"
    return "detailed" if _CLOCK.match(text) or _DETAILED_HINT.search(text) else "condensed"


def _yards(m: re.Match | None) -> int | None:
    if m is None:
        return None
    if m.groupdict().get("nogain"):
        return 0
    if m.groupdict().get("loss"):
        return -int(m.group("loss"))
    if m.group("yds") is None:
        return None
    y = int(m.group("yds"))
    return -y if m.groupdict().get("gl") == "loss" else y


def split_spot(spot: str | None) -> tuple[str, int] | None:
    if not spot:
        return None
    m = re.match(r"^(.*?)\s?(\d{1,2})$", spot.strip())
    if not m:
        return None
    return norm_abbr(m.group(1)), int(m.group(2))


def norm_abbr(abbr: str) -> str:
    return re.sub(r"[^A-Z&]", "", abbr.upper())


def _main_clause(body: str) -> str:
    """The play itself: text before the first penalty."""
    m = _PEN_SPLIT.search(body)
    return body[: m.start()] if m else body


def _tacklers(clause: str) -> list[str]:
    """Players in the parenthesis after the play's end spot, before any recovery or return."""
    cut = len(clause)
    for marker in (" recovered by", " return ", " return for", " lateral to"):
        i = clause.find(marker)
        if i != -1:
            cut = min(cut, i)
    head = clause[:cut]
    # the parenthesis may follow "forced by X" on fumbles, so look for the last one in head
    for m in re.finditer(r"\(([^()]*)\)", head):
        inner = m.group(1)
        if inner.startswith("H:") or not re.search(r"[A-Za-z]{2,}", inner):
            continue
        return [x.strip() for x in _IN_PARENS.findall(inner)]
    return []


def parse_detailed(text: str) -> dict:
    """Fields of one detailed-format play. Players are (role, name as written)."""
    out: dict = {"players": [], "spots": {}}
    m = _CLOCK.match(text)
    body = text[m.end():] if m else text
    f = _FORMATION.match(body)
    form = f.group(0) if f else ""
    out["shotgun"] = "Shotgun" in form or "[SG]" in form
    out["no_huddle"] = "No Huddle" in form or "[NH]" in form
    body = body[f.end():] if f else body
    clause = _main_clause(body)
    players = out["players"]

    if (k := _D_KNEEL.match(clause)):
        out["action"] = "kneel"
        out["kneel"] = True
        if k.group("p") and re.fullmatch(PLAYER, k.group("p")):
            players.append(("rusher", k.group("p")))
    elif (mi := _D_INT.match(clause)):
        out.update(action="pass", completed=False, intercepted=True)
        players += [("passer", mi.group("p")), ("interceptor", mi.group("i"))]
        out["spots"]["pass"] = mi.group("spot")
    elif (mp := _D_PASS.match(clause)):
        out["action"] = "pass"
        done = mp.group("res") == "complete"
        out["completed"] = done
        out["intercepted"] = False
        if mp.group("p") and re.fullmatch(PLAYER, mp.group("p")):
            players.append(("passer", mp.group("p")))
        if mp.group("t"):
            players.append(("target", mp.group("t")))
        if ", Spike" in clause or "Spike" in clause[mp.end(): mp.end() + 8]:
            out["action"] = "spike"
            out["spike"] = True
        out["pass_depth"] = mp.group("depth")
        out["pass_dir"] = mp.group("dir")
        sm = (_CAUGHT if done else _THROWN).search(clause, mp.end())
        if sm:
            out["spots"]["pass"] = sm.group("spot")
        if done:
            out["text_yards"] = _yards(_PASS_YDS.search(clause, mp.end()))
            tk = _END_SPOT_TACKLE.search(clause, mp.end())
            out["tacklers"] = _tacklers(clause[tk.start():]) if tk else []
        if (b := _BREAKUP.search(clause)):
            players.append(("breakup", b.group("p")))
    elif (ms := _D_SACK.match(clause)):
        out["action"] = "sack"
        out["text_yards"] = -int(ms.group("yds"))
        if re.match(PLAYER, ms.group("p")):
            players.append(("passer", ms.group("p")))
        tk = _END_SPOT_TACKLE.search(clause, ms.end())
        for name in _tacklers(clause[tk.start():]) if tk else []:
            players.append(("sacker", name))
    elif (mr := _D_RUSH.match(clause)):
        out["action"] = "rush"
        out["run_dir"] = mr.group("dir")
        out["text_yards"] = _yards(mr)
        if mr.group("p") and re.match(PLAYER, mr.group("p")):
            players.append(("rusher", mr.group("p")))
        tk = _END_SPOT_TACKLE.search(clause, mr.end())
        out["tacklers"] = _tacklers(clause[tk.start():]) if tk else []
    elif clause.strip() == "" and body.startswith("PENALTY"):
        out["action"] = "penalty"
    else:
        out["action"] = "other"

    if out["action"] in ("rush", "pass") and (tk := _END_SPOT_TACKLE.search(clause)):
        out["spots"]["end"] = tk.group("spot")
    for name in out.get("tacklers", []):
        players.append(("tackler", name))
    if (h := _HURRY.search(clause)):
        out["qb_hurried"] = True
        players += [("hurry", h.group(n)) for n in ("a", "b") if h.group(n)]
    if (fm := _FUMBLE.search(clause)):
        out["fumbled"] = True
        if re.match(PLAYER, fm.group("p")):
            players.append(("fumbler", fm.group("p")))
        if fm.group("f"):
            players.append(("forced_fumble", fm.group("f")))
    return out


def parse_condensed(text: str) -> dict:
    out: dict = {"players": [], "spots": {}}
    clause = _main_clause(text)
    players = out["players"]
    if _C_KNEEL.search(clause):
        out.update(action="kneel", kneel=True)
    elif (mi := _C_INT.match(clause)):
        out.update(action="pass", completed=False, intercepted=True)
        players.append(("passer", mi.group("p")))
        if mi.group("i"):
            players.append(("interceptor", mi.group("i")))
    elif (mt := _C_TD_PASS.match(clause)):
        out.update(action="pass", completed=True, intercepted=False, text_yards=int(mt.group("yds")))
        players += [("passer", mt.group("p")), ("target", mt.group("t"))]
    elif (mt := _C_TD_RUN.match(clause)):
        out.update(action="rush", text_yards=int(mt.group("yds")))
        players.append(("rusher", mt.group("p")))
    elif (mp := _C_PASS.match(clause)):
        # "X pass to Y" (no result word) is a completion
        done = mp.group("res") == "complete" or (mp.group("res") is None and mp.group("t") is not None)
        out.update(action="pass", completed=done, intercepted=False)
        players.append(("passer", mp.group("p")))
        if mp.group("t"):
            players.append(("target", mp.group("t")))
        if out["completed"]:
            out["text_yards"] = _yards(_C_YDS.search(clause, mp.end()))
        if (b := _C_BREAKUP.search(clause)):
            players.append(("breakup", b.group("p")))
    elif (ms := _C_SACK.match(clause)):
        out["action"] = "sack"
        players.append(("passer", ms.group("p")))
        players += [("sacker", ms.group(n)) for n in ("a", "b") if ms.group(n)]
        y = _yards(_C_YDS.search(clause, ms.end()))
        out["text_yards"] = y
    elif (mr := _C_RUSH.match(clause)):
        out["action"] = "rush"
        if mr.group("p") != "TEAM":
            players.append(("rusher", mr.group("p")))
        out["text_yards"] = _yards(_C_YDS.search(clause, mr.start()))
    elif re.match(r"^[A-Z][A-Z\.& ]+ Penalty, |^PENALTY", text):
        out["action"] = "penalty"
    else:
        out["action"] = "other"
    if (fm := _C_FUMBLE.search(clause)):
        out["fumbled"] = True
        players.append(("fumbler", fm.group("p").strip()))
        if fm.group("f"):
            players.append(("forced_fumble", fm.group("f").strip()))
    return out


def parse_penalties(text: str) -> list[dict]:
    """Every penalty on the play: team text, type, status, yards, player."""
    out = []
    starts = list(_PEN_SPLIT.finditer(text))
    for i, m in enumerate(starts):
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[m.end(): seg_end].strip()
        if m.group(0) != "PENALTY":  # condensed "NOTRE DAME Penalty, Defensive Offside (X) to the ..."
            team = m.group(0)[: -len(" Penalty, ")].strip()
            body = seg
        else:
            if seg.startswith("Before the snap, "):
                seg = seg[len("Before the snap, "):]
            if (mo := _PEN_ON.match(seg)):
                name = mo.group("p").strip()
                out.append({
                    "team_text": mo.group("team").strip(),
                    "penalty": mo.group("type").strip(),
                    "status": mo.group("status") or "accepted",
                    "yards": int(mo.group("yds")),
                    "no_play": bool(re.search(r"no play", seg, re.I)),
                    "name": None if name == mo.group("team").strip() else name,
                })
                continue
            team, _, body = seg.partition(" ")
            # two-word abbreviations such as "NC ST"
            if re.match(r"^ST\b", body):
                team, body = team + " ST", body[3:]
            body = body.strip()
        offsets = body.split(" offsetting ")
        for j, part in enumerate(offsets):
            if j > 0:  # "KSU Pass Interference offsetting SAM Ineligible Downfield on Pass offsetting"
                t2, _, part = part.strip().partition(" ")
                team_j = t2
            else:
                team_j = team
            status = "offsetting" if len(offsets) > 1 else ("declined" if re.search(r"\bdeclined\b", part) else "accepted")
            pm = _PEN_PLAYER.search(part) or _PEN_ON_PLAYER.search(part)
            ym = _PEN_YDS.search(part)
            ptype = re.split(r" \(| \d+ yards? from| declined| offsetting|\. NO PLAY| to the |, |\.$| on (?=#\d|[A-Z][A-Za-z'\-]+, ?[A-Z])| enforced",
                             part, maxsplit=1)[0]
            ptype = _PEN_CODE.sub("", ptype).strip(" .")
            name = pm.group("p").strip() if pm else None
            if name and re.fullmatch(r"-?\d+ Yards?", name):
                name = None
            out.append({
                "team_text": team_j.strip(),
                "penalty": ptype or None,
                "status": status,
                "yards": int(ym.group("yds")) if ym else None,
                "no_play": bool(re.search(r"no play", text[m.start():], re.I)),
                "name": name,
            })
    return out


# --------------------------------------------------------------------------- names
def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


_SUFFIX_RE = re.compile(r"(?:,?\s+(?:jr|sr|ii|iii|iv|v|ll|lll)\.?)+$", re.I)


def split_name(name: str) -> tuple[int | None, str, str]:
    """(jersey, first or initial, last) from any of the written forms."""
    name = name.strip()
    jersey = None
    m = re.match(r"^#(\d{1,2})\s+(.*)$", name)
    if m:
        jersey, name = int(m.group(1)), m.group(2)
        first, _, last = name.partition(".")
        return jersey, _plain(first), _plain(_SUFFIX_RE.sub("", last.strip()))
    if "," in name and not re.search(r",\s*(?:Jr|Sr|II|III|IV|V)\.?$", name):
        last, _, first = name.partition(",")
        return None, _plain(first), _plain(_SUFFIX_RE.sub("", last.strip()))
    name = _SUFFIX_RE.sub("", name)
    m = re.match(r"^([A-Za-z]{1,3})\.\s?(\S.*)$", name)  # "R.Blum", "Av. Johnson"
    if m:
        return None, _plain(m.group(1)), _plain(m.group(2))
    parts = name.split()
    if len(parts) == 1:
        return None, "", _plain(parts[0])
    return None, _plain(parts[0]), _plain(" ".join(parts[1:]))


class RosterIndex:
    """Lookups of athlete id by (season, team) and jersey or name."""

    def __init__(self, roster: pd.DataFrame):
        self.by_jersey: dict = defaultdict(list)
        self.by_initial: dict = defaultdict(list)
        self.by_full: dict = defaultdict(list)
        if roster is None or roster.empty:
            return
        r = roster.dropna(subset=["id", "team", "season"])
        for season, team, aid, first, last, jersey in zip(
            r["season"], r["team"], r["id"], r["firstName"].fillna(""), r["lastName"].fillna(""),
            r["jersey"] if "jersey" in r else [np.nan] * len(r),
        ):
            key = (int(season), team)
            f, l_ = _plain(first), _plain(_SUFFIX_RE.sub("", last))
            rec = (int(aid), f, l_)
            if jersey == jersey and jersey is not None:
                self.by_jersey[key + (int(jersey),)].append(rec)
            self.by_initial[key + (l_, f[:1])].append(rec)
            self.by_full[key + (f + l_,)].append(rec)

    def match(self, season: int, team: str, name: str) -> tuple[int | None, int | None, str]:
        jersey, first, last = split_name(name)
        key = (int(season), team)
        if jersey is not None:
            cands = [c for c in self.by_jersey.get(key + (jersey,), [])
                     if c[2] == last or c[2].startswith(last) or last.startswith(c[2])]
            if len(cands) == 1:
                return cands[0][0], jersey, "jersey"
        if first and last:
            if len(first) > 1:
                c = self.by_full.get(key + (first + last,), [])
                if len(c) == 1:
                    return c[0][0], jersey, "name"
            c = self.by_initial.get(key + (last, first[:1]), [])
            if len(c) == 1:
                return c[0][0], jersey, "name"
        return None, jersey, "none"


# --------------------------------------------------------------------------- build
class AbbrMap:
    """Which team a spot or penalty abbreviation ("MIZ", "NC ST") stands for.

    Votes come from the end spot of each detailed rush and pass: with the line of
    scrimmage (yardsToGoal) and the text's yards, "to the MIZ31" is either the
    defense's 31 or the offense's 31. A game's own votes decide; an abbreviation
    with no votes in the game falls back to its votes in other games for one of the
    game's two teams.
    """

    def __init__(self, rows):
        """rows: (game_id, offense, defense, yards_to_goal, text_yards, end_spot)."""
        game: dict = defaultdict(Counter)
        overall: dict = defaultdict(Counter)
        for gid, off, dfn, ytg, yds, spot in rows:
            sp = split_spot(spot)
            if sp is None or yds is None or ytg is None or ytg != ytg:
                continue
            abbr, n = sp
            if n == 50:
                continue
            end = ytg - yds
            team = dfn if abs(n - end) <= 1 else off if abs((100 - n) - end) <= 1 else None
            if team is not None:
                game[(gid, abbr)][team] += 1
                overall[abbr][team] += 1
        self.game = {k: c.most_common(1)[0][0] for k, c in game.items()}
        self.overall = overall

    def team(self, gid: int, abbr: str, offense: str, defense: str) -> str | None:
        t = self.game.get((gid, abbr))
        if t is not None:
            return t
        c = self.overall.get(abbr)
        if not c:
            return None
        a, b = c.get(offense, 0), c.get(defense, 0)
        if a > 2 * b:
            return offense
        if b > 2 * a:
            return defense
        return None


def _spot_ytg(spot: str | None, gid: int, offense: str, defense: str, amap: AbbrMap) -> int | None:
    sp = split_spot(spot)
    if sp is None:
        return None
    abbr, n = sp
    if n == 50:
        return 50
    team = amap.team(gid, abbr, offense, defense)
    if team is None:
        return None
    return 100 - n if team == offense else n


def _subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _team_from_text(team_text: str, gid: int, offense: str, defense: str, amap: AbbrMap) -> str | None:
    """Team of a penalty: an abbreviation known for the game, else a name match
    ("NOTRE DAME", "WESTRN KENTUCKY" as letters in order of the team name)."""
    t = amap.team(gid, norm_abbr(team_text), offense, defense)
    if t is not None:
        return t
    up = _plain(team_text)
    if len(up) < 3:
        return None
    names = {offense: _plain(offense), defense: _plain(defense)}
    exact = [k for k, v in names.items() if v == up]
    if len(exact) == 1:
        return exact[0]
    hits = [k for k, v in names.items() if len(up) >= 4 and up[0] == v[0] and _subseq(up, v)]
    return hits[0] if len(hits) == 1 else None


def build_play_text(plays: pd.DataFrame, game_index: pd.DataFrame,
                    roster: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """play_detail, play_players and play_penalties from the raw plays table."""
    empty = {
        "play_detail": pd.DataFrame({c: [] for c in DETAIL_COLUMNS}),
        "play_players": pd.DataFrame({c: [] for c in PLAYER_COLUMNS}),
        "play_penalties": pd.DataFrame({c: [] for c in PENALTY_COLUMNS}),
    }
    if plays.empty:
        return empty
    gi = game_index[["game_id", "home_id", "away_id", "home_team", "away_team"]]
    p = plays.merge(gi, left_on="gameId", right_on="game_id", how="inner")
    p = p[(p["offense"] == p["home_team"]) | (p["offense"] == p["away_team"])].reset_index(drop=True)
    if p.empty:
        return empty
    team_id = {}
    for gid, ht, hid, at, aid in zip(p["game_id"], p["home_team"], p["home_id"], p["away_team"], p["away_id"]):
        team_id[(gid, ht)] = int(hid)
        team_id[(gid, at)] = int(aid)

    texts = p["playText"].astype("object").where(p["playText"].notna(), None).tolist()
    fmts = [text_format(t) for t in texts]
    parsed = [parse_detailed(t) if f == "detailed" else parse_condensed(t) if f == "condensed" else
              {"players": [], "spots": {}, "action": "none"} for t, f in zip(texts, fmts)]

    gids = p["game_id"].astype("int64").tolist()
    offs, defs = p["offense"].tolist(), p["defense"].tolist()
    ytg = pd.to_numeric(p["yardsToGoal"], errors="coerce").tolist()
    amap = AbbrMap([
        (g, o, d, y, r.get("text_yards"), r["spots"].get("end"))
        for g, o, d, y, r, f in zip(gids, offs, defs, ytg, parsed, fmts) if f == "detailed"
    ])

    seasons = p["season"].astype("int64").tolist()
    pids = p["id"].astype("str").tolist()
    rix = RosterIndex(roster) if roster is not None else RosterIndex(pd.DataFrame())

    detail_rows, player_rows, pen_rows = [], [], []
    for i, r in enumerate(parsed):
        gid, off, dfn, fmt = gids[i], offs[i], defs[i], fmts[i]
        air = None
        if fmt == "detailed" and r.get("spots", {}).get("pass") and ytg[i] == ytg[i]:
            s = _spot_ytg(r["spots"]["pass"], gid, off, dfn, amap)
            if s is not None:
                a = int(ytg[i]) - s
                air = a if -20 <= a <= 80 else None
        yac = r.get("text_yards") - air if (air is not None and r.get("completed") and r.get("text_yards") is not None) else None
        pens = parse_penalties(texts[i]) if texts[i] and ("PENALTY" in texts[i] or "Penalty," in texts[i]) else []
        tack = r.get("tacklers")
        detail_rows.append((
            r.get("action"), r.get("shotgun") if fmt == "detailed" else None,
            r.get("no_huddle") if fmt == "detailed" else None, r.get("run_dir"),
            r.get("pass_depth"), r.get("pass_dir"), r.get("completed"), r.get("intercepted"),
            bool(r.get("kneel", False)), bool(r.get("spike", False)), r.get("text_yards"), air, yac,
            len(tack) if (fmt == "detailed" and tack is not None) else None,
            ("breakup" in {x[0] for x in r["players"]}) if r.get("action") == "pass" else None,
            bool(r.get("qb_hurried", False)) if fmt == "detailed" and r.get("action") in ("pass", "sack") else None,
            bool(r.get("fumbled", False)), len(pens), any(x["no_play"] for x in pens),
        ))
        for role, name in r["players"]:
            side = "offense" if role in OFFENSE_ROLES else "defense"
            team = off if side == "offense" else dfn
            aid, jersey, how = rix.match(seasons[i], team, name)
            player_rows.append((gid, pids[i], seasons[i], role, name, jersey, side,
                                team_id.get((gid, team)), aid, how))
        for n, pen in enumerate(pens, start=1):
            team = _team_from_text(pen["team_text"], gid, off, dfn, amap)
            side = None if team is None else ("offense" if team == off else "defense")
            aid = jersey = None
            how = None
            if pen["name"]:
                if team is not None:
                    aid, jersey, how = rix.match(seasons[i], team, pen["name"])
                else:
                    jersey, how = split_name(pen["name"])[0], "none"
            pen_rows.append((gid, pids[i], seasons[i], n, pen["team_text"], side,
                             team_id.get((gid, team)) if team else None, pen["penalty"],
                             pen["status"], pen["yards"], pen["no_play"], pen["name"], jersey, aid, how))

    d = pd.DataFrame(detail_rows, columns=[
        "action", "shotgun", "no_huddle", "run_dir", "pass_depth", "pass_dir", "completed",
        "intercepted", "kneel", "spike", "text_yards", "air_yards", "yac", "n_tacklers",
        "broken_up", "qb_hurried", "fumbled", "penalties", "no_play",
    ])
    detail = pd.DataFrame({
        "game_id": p["game_id"].astype("int64"),
        "play_id": p["id"].astype("str"),
        "season": p["season"].astype("int64"),
        "season_type": p["season_type"].astype("str"),
        "week": p["week"].astype("int64"),
        "text_format": fmts,
    })
    for col in ("action", "run_dir", "pass_depth", "pass_dir"):
        detail[col] = d[col].astype("string")
    for col in ("shotgun", "no_huddle", "completed", "intercepted", "broken_up", "qb_hurried"):
        detail[col] = d[col].astype("boolean")
    for col in ("kneel", "spike", "fumbled", "no_play"):
        detail[col] = d[col].astype(bool)
    for col in ("text_yards", "air_yards", "yac", "n_tacklers"):
        detail[col] = pd.array(d[col].tolist(), dtype="Int64")
    detail["penalties"] = d["penalties"].astype("int64")

    players = pd.DataFrame(player_rows, columns=PLAYER_COLUMNS)
    pens = pd.DataFrame(pen_rows, columns=PENALTY_COLUMNS)
    for frame, ints in ((players, ("jersey", "team_id", "athlete_id")),
                        (pens, ("yards", "jersey", "team_id", "athlete_id"))):
        for col in ints:
            frame[col] = pd.array([None if v is None or v != v else int(v) for v in frame[col]], dtype="Int64")
        frame["game_id"] = frame["game_id"].astype("int64")
        frame["play_id"] = frame["play_id"].astype("str")
        frame["season"] = frame["season"].astype("int64")
    pens["no_play"] = pens["no_play"].astype(bool)
    pens["n"] = pens["n"].astype("int64")
    return {
        "play_detail": detail[DETAIL_COLUMNS],
        "play_players": players[PLAYER_COLUMNS],
        "play_penalties": pens[PENALTY_COLUMNS],
    }


# --------------------------------------------------------------------------- command
def build(data_dir="data", write: bool = True) -> dict[str, pd.DataFrame]:
    """Parse every stored play. Writes data/core/play_{detail,players,penalties}.parquet."""
    from pathlib import Path

    from ..store import Store
    from .game_index import build_game_index, build_weeks
    from .pbp import load_raw_plays

    store = Store(data_dir)
    core = Path(data_dir) / "core" / "game_index.parquet"
    game_index = pd.read_parquet(core) if core.exists() else build_game_index(store, build_weeks(store))
    try:
        roster = store.load("roster")
    except FileNotFoundError:
        roster = None
    tables = build_play_text(load_raw_plays(store), game_index, roster)
    if write:
        out = Path(data_dir) / "core"
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            tmp = out / f"{name}.parquet.tmp"
            frame.to_parquet(tmp, index=False)
            tmp.replace(out / f"{name}.parquet")
    return tables


def summarize(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of plays with each field, by season (detailed-format rushes and passes)."""
    d = tables["play_detail"]
    det = d[d["text_format"] == "detailed"]
    rush, pas = det[det["action"] == "rush"], det[det["action"] == "pass"]
    return pd.DataFrame({
        "plays": d.groupby("season").size(),
        "detailed": det.groupby("season").size(),
        "rush_dir": rush.groupby("season")["run_dir"].apply(lambda s: s.notna().mean()),
        "pass_depth": pas.groupby("season")["pass_depth"].apply(lambda s: s.notna().mean()),
        "air_yards": pas.groupby("season")["air_yards"].apply(lambda s: s.notna().mean()),
    }).fillna(0).round(3)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .. import settings

    ap = argparse.ArgumentParser(description="Parse play text into data/core/play_{detail,players,penalties}.parquet")
    ap.add_argument("--data-dir", default=str(settings.data_dir()))
    ap.add_argument("--check", action="store_true", help="parse in memory and report, without writing")
    args = ap.parse_args(argv)
    tables = build(args.data_dir, write=not args.check)
    for name, frame in tables.items():
        print(f"{name:16s} {len(frame):>10,d} rows")
    print(summarize(tables).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
