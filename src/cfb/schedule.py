"""Windows Task Scheduler tasks that run the local engine (cfb.engine) without the app open.

    python -m cfb.engine schedule show
    python -m cfb.engine schedule install weekly [--day MON] [--time 05:30] [--no-wake]
    python -m cfb.engine schedule install daily [--time 06:00]    # light refresh, Tuesday to Sunday
    python -m cfb.engine schedule remove weekly|daily|all

Two tasks per edition (cfb.edition), named after the app, so main and experimental never
replace each other's tasks:
  weekly   `engine all`: refresh the season, rebuild, predict, report, back up. Default Monday
           05:30 PC time (experimental: 06:00), after Sunday's data settles.
  daily    `engine light`: games, lines, polls, weather; rebuild the week's report. Optional,
           default 06:00 (experimental 06:30) on the days the weekly task does not run.

Both: run as soon as possible after a missed start (PC off or asleep), wake the PC to run,
run on battery, at most one instance, stop after 4 hours. They run as the signed-in user
(InteractiveToken: when the user is signed in, a locked screen included), because the keys are
in that user's Windows Credential Manager. No password is stored and no admin rights are needed.

The task XML is built here and registered with `schtasks /Create /XML`; nothing else on the
system is changed. Times are the PC's local time.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from . import edition as _edition
from . import settings

NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
KINDS = ("weekly", "daily")
DEFAULT_TIME = {"weekly": "05:30", "daily": "06:00"}
ENGINE_COMMAND = {"weekly": "all", "daily": "light"}
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console window flashes when the app calls schtasks


class ScheduleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plan:
    kind: str  # weekly | daily
    days: tuple[str, ...]  # names from DAYS
    time: str  # HH:MM, PC local time
    wake: bool = True

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ScheduleError(f"unknown task kind {self.kind!r}")
        bad = [d for d in self.days if d not in DAYS]
        if bad or not self.days:
            raise ScheduleError(f"bad days {self.days!r}")
        hh, _, mm = self.time.partition(":")
        if not (hh.isdigit() and mm.isdigit() and 0 <= int(hh) < 24 and 0 <= int(mm) < 60):
            raise ScheduleError(f"bad time {self.time!r}; use HH:MM")


def shift(hhmm: str, minutes: int) -> str:
    t = datetime(2000, 1, 3, int(hhmm[:2]), int(hhmm[3:5])) + timedelta(minutes=minutes)
    return f"{t:%H:%M}"


def default_plan(kind: str, ed: _edition.Edition | None = None) -> Plan:
    ed = ed or _edition.current()
    time = shift(DEFAULT_TIME[kind], ed.schedule_offset_min)
    days = ("Monday",) if kind == "weekly" else DAYS[1:]
    return Plan(kind, days, time)


def parse_day(text: str) -> str:
    t = text.strip().lower()
    for d in DAYS:
        if d.lower().startswith(t[:3]) and len(t) >= 3:
            return d
    raise ScheduleError(f"unknown day {text!r}")


def task_name(kind: str, ed: _edition.Edition | None = None) -> str:
    return (ed or _edition.current()).task_name(kind)


def engine_command(kind: str) -> tuple[str, str, str]:
    """(program, arguments, working folder) the task runs.

    Packaged app: "CFB Engine.exe" next to the app (built without a console window), else the
    app itself. Source checkout: pythonw (no window) -m cfb.engine."""
    cmd = f"{ENGINE_COMMAND[kind]} --trigger scheduled"
    if settings.frozen():
        exe = Path(sys.executable)
        quiet = exe.with_name("CFB Engine.exe")
        prog = quiet if quiet.exists() else exe
        return str(prog), f"engine {cmd}", str(exe.parent)
    py = Path(sys.executable)
    quiet = py.with_name("pythonw.exe")
    prog = quiet if quiet.exists() else py
    return str(prog), f"-m cfb.engine {cmd}", os.getcwd()


def current_user() -> str:
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    domain = os.environ.get("USERDOMAIN")
    return f"{domain}\\{user}" if domain and user else user


def task_xml(plan: Plan, program: str, arguments: str, workdir: str, *, user: str | None = None,
             description: str | None = None, start: date | None = None) -> str:
    """Task Scheduler XML (schema 1.4, Windows 10 and later) for `plan`."""
    ed = _edition.current()
    start = start or date.today()
    days = "".join(f"<{d} />" for d in plan.days)
    desc = description or (f"{ed.app_name}: "
                           + ("weekly update (refresh, rebuild, predict, report, back up)" if plan.kind == "weekly"
                              else "daily light refresh (games, lines, polls, weather; rebuild the report)")
                           + ". Created by the app; change it on its Settings page.")
    user = user if user is not None else current_user()
    user_xml = f"<UserId>{escape(user)}</UserId>" if user else ""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="{NS}">
  <RegistrationInfo>
    <Author>{escape(ed.app_name)}</Author>
    <Description>{escape(desc)}</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start:%Y-%m-%d}T{plan.time}:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek>{days}</DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      {user_xml}
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>{'true' if plan.wake else 'false'}</WakeToRun>
    <ExecutionTimeLimit>PT4H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"{escape(program)}"</Command>
      <Arguments>{escape(arguments)}</Arguments>
      <WorkingDirectory>{escape(workdir)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def parse_xml(text: str) -> dict:
    """The parts of a task definition the app shows: days, time, wake, enabled, command."""
    root = ET.fromstring(re.sub(r"^\s*<\?xml[^>]*\?>", "", text))  # schtasks prints a UTF-16 declaration
    q = lambda path: root.find(path.replace("t:", "{%s}" % NS))  # noqa: E731
    start = (q("t:Triggers/t:CalendarTrigger/t:StartBoundary").text or "") if q("t:Triggers/t:CalendarTrigger/t:StartBoundary") is not None else ""
    dow = q("t:Triggers/t:CalendarTrigger/t:ScheduleByWeek/t:DaysOfWeek")
    days = [c.tag.split("}")[1] for c in dow] if dow is not None else []
    wake = q("t:Settings/t:WakeToRun")
    enabled = q("t:Settings/t:Enabled")
    cmd, args = q("t:Actions/t:Exec/t:Command"), q("t:Actions/t:Exec/t:Arguments")
    return {"days": days, "time": start[11:16] if len(start) >= 16 else None,
            "wake": wake is not None and wake.text == "true",
            "enabled": enabled is None or enabled.text == "true",
            "command": (cmd.text or "").strip('"') if cmd is not None else None,
            "arguments": args.text if args is not None else None}


# ---- schtasks ---------------------------------------------------------------------------------

def available() -> bool:
    return sys.platform == "win32"


def _run(args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    if not available():
        raise ScheduleError("Task Scheduler is only on Windows")
    kw = {"creationflags": NO_WINDOW} if sys.platform == "win32" else {}
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)


def install(plan: Plan, *, name: str | None = None, command: tuple[str, str, str] | None = None) -> str:
    """Create or replace the task. Returns its name."""
    name = name or task_name(plan.kind)
    prog, args, workdir = command or engine_command(plan.kind)
    xml = task_xml(plan, prog, args, workdir)
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="cfb-task-")
    os.close(fd)
    try:
        Path(path).write_text(xml, encoding="utf-16")  # schtasks reads UTF-16 task files
        r = _run(["schtasks", "/Create", "/TN", name, "/XML", path, "/F"])
    finally:
        Path(path).unlink(missing_ok=True)
    if r.returncode != 0:
        raise ScheduleError(f"schtasks /Create failed: {(r.stderr or r.stdout).strip()}")
    return name


def remove(kind: str, *, name: str | None = None) -> bool:
    """Delete the task. False when there was none."""
    name = name or task_name(kind)
    if query_xml(name) is None:
        return False
    r = _run(["schtasks", "/Delete", "/TN", name, "/F"])
    if r.returncode != 0:
        raise ScheduleError(f"schtasks /Delete failed: {(r.stderr or r.stdout).strip()}")
    return True


def query_xml(name: str) -> str | None:
    r = _run(["schtasks", "/Query", "/TN", name, "/XML"])
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def run_info(name: str) -> dict:
    """Next and last run time and the last result, from PowerShell (locale-independent)."""
    script = (f"$i = Get-ScheduledTaskInfo -TaskName '{name.replace(chr(39), chr(39) * 2)}' -ErrorAction Stop; "
              "[pscustomobject]@{next=$i.NextRunTime; last=$i.LastRunTime; result=$i.LastTaskResult} "
              "| ConvertTo-Json -Compress")
    try:
        r = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
        doc = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
    except (OSError, ValueError, subprocess.TimeoutExpired, ScheduleError):
        return {}
    out = {}
    for k in ("next", "last"):
        v = doc.get(k)
        if isinstance(v, dict):  # Windows PowerShell 5 dates: {"value": "/Date(ms)/", "DateTime": "..."}
            v = v.get("DateTime") or v.get("value")
        out[k] = v
    out["result"] = doc.get("result")
    return out


def show(kind: str, *, name: str | None = None, info: bool = True) -> dict:
    """{"kind", "name", "installed", plus parse_xml fields and next/last run when installed}."""
    name = name or task_name(kind)
    out: dict = {"kind": kind, "name": name, "installed": False}
    if not available():
        out["note"] = "Task Scheduler is only on Windows"
        return out
    xml = query_xml(name)
    if xml is None:
        return out
    out["installed"] = True
    try:
        out.update(parse_xml(xml))
    except ET.ParseError:
        pass
    if info:
        out.update(run_info(name))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="cfb.engine schedule", description="Windows Task Scheduler tasks of the engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    i = sub.add_parser("install")
    i.add_argument("kind", choices=KINDS)
    i.add_argument("--day", action="append", help="weekday, repeatable (weekly default Monday; daily Tue-Sun)")
    i.add_argument("--time", help="HH:MM PC local time")
    i.add_argument("--no-wake", action="store_true", help="do not wake the PC to run")
    r = sub.add_parser("remove")
    r.add_argument("kind", choices=(*KINDS, "all"))
    x = sub.add_parser("xml", help="print the task XML without installing it")
    x.add_argument("kind", choices=KINDS)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "show":
            print(json.dumps([show(k) for k in KINDS], indent=1, default=str))
        elif args.cmd in ("install", "xml"):
            base = default_plan(args.kind)
            days = tuple(parse_day(d) for d in getattr(args, "day", None) or ()) or base.days
            plan = Plan(args.kind, days, getattr(args, "time", None) or base.time, not getattr(args, "no_wake", False))
            if args.cmd == "xml":
                print(task_xml(plan, *engine_command(args.kind)))
            else:
                print(f"installed {install(plan)}: {', '.join(plan.days)} at {plan.time}")
        else:
            for k in KINDS if args.kind == "all" else (args.kind,):
                print(f"{task_name(k)}: {'removed' if remove(k) else 'not installed'}")
    except ScheduleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
