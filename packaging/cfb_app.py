"""Entry point of the packaged Windows app (PyInstaller, packaging/cfb_app.spec).

    "CFB Predictor.exe"                  start the app: local server, opens the browser
    "CFB Predictor.exe" --no-browser     start without opening the browser (smoke test)
    "CFB Predictor.exe" sync [...]       download the newest database from R2 (cfb.warehouse_sync)
    "CFB Predictor.exe" engine <cmd>     local engine: bootstrap, refresh, build, week, all, light,
                                         check, backup, status, schedule (cfb.engine)
    "CFB Predictor.exe" selftest [--system]   import everything the app needs, then exit (CI);
                                         --system also checks the key store and Task Scheduler

"CFB Engine.exe" is the same program built without a console window: Task Scheduler and the
app's Run buttons start the engine with it, so no window opens. Its output goes to the
engine's log files.
"""
import multiprocessing
import os
import re
import runpy
import sys


def _quiet_streams() -> None:
    """A windowless build has no console: sys.stdout and sys.stderr are None, and any print
    would fail. The engine writes its own log file, so discard what reaches the streams."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def system_checks() -> list[str]:
    """Windows Credential Manager round trip and a Task Scheduler register/query/delete of a
    throwaway task built from the app's own XML. Returns problems (empty when all is well)."""
    problems = []
    try:
        import keyring

        service, name = "cfb-selftest", "probe"
        keyring.set_password(service, name, "ok")
        got = keyring.get_password(service, name)
        keyring.delete_password(service, name)
        if got != "ok":
            problems.append(f"keyring read back {got!r}")
        print(f"keyring=ok ({type(keyring.get_keyring()).__name__})")
    except Exception as exc:
        problems.append(f"keyring: {exc.__class__.__name__}: {exc}")
    if sys.platform == "win32":
        from cfb import schedule as S

        name = "CFB Predictor selftest"
        try:
            plan = S.default_plan("weekly")
            S.install(plan, name=name, command=S.engine_command("weekly"))
            got = S.show("weekly", name=name)
            S.remove("weekly", name=name)
            if not got.get("installed") or got.get("days") != list(plan.days) or got.get("time") != plan.time:
                problems.append(f"task read back {got!r}")
            elif not re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d$", str(got.get("next"))):
                problems.append(f"task next run time unreadable: {got.get('next')!r}")
            else:
                print(f"schedule=ok ({got.get('command')} {got.get('arguments')}; next {got.get('next')})")
        except Exception as exc:
            problems.append(f"schedule: {exc.__class__.__name__}: {exc}")
            try:
                S.remove("weekly", name=name)
            except Exception:
                pass
    return problems


def selftest(system: bool = False) -> int:
    """Catch modules PyInstaller left out: import the app, the model and every feature group."""
    import streamlit  # noqa: F401
    from cfb import edition, engine, schedule, settings  # noqa: F401
    from cfb.features.base import discover_groups
    from cfb.model import config as C
    from cfb.report.config import load_config
    from cfb.ui import data, engine_job, launch, pages, update  # noqa: F401
    import cfb.predict  # noqa: F401
    import cfb.run_week  # noqa: F401
    import cfb.ingest  # noqa: F401

    try:
        from cfb import warehouse_sync  # noqa: F401
        import boto3  # noqa: F401  (S3 backend used on the PC)
        sync_ok = "yes"
    except ImportError as exc:
        sync_ok = f"no ({exc.name})"
    from cfb.ui import availability, connection  # noqa: F401  (Players page, Cloudflare connection)
    import cfb.lab.depth_impact  # noqa: F401  (the Players page recompute)

    groups = discover_groups()
    cfg = load_config()
    app = launch.app_path()
    from cfb.report.html import _template
    template_ok = "__BUNDLE__" in _template()  # the engine writes reports with it
    ed = edition.current()
    keys = key_store_check()
    print(f"edition={ed.name} groups={len(groups)} models={sorted(C.MODELS)} primary={cfg.primary} "
          f"config={settings.config_path()} app={app} build={update.current_build()} sync={sync_ok} "
          f"data={settings.data_dir()} archive={settings.asof_archive()} keys={keys}")
    problems = system_checks() if system else []
    if sys.platform == "win32" and "ok" not in keys:
        problems.append(f"no working key store ({keys})")
    for p in problems:
        print(f"selftest problem: {p}", file=sys.stderr)
    if not template_ok:
        problems.append("the dashboard template is missing")
    if not app.exists() or not groups or problems:
        print("selftest failed", file=sys.stderr)
        return 1
    print("selftest ok")
    return 0


def key_store_check() -> str:
    """Where saved keys can go in this build: a keyring round trip under a throwaway service name,
    and the DPAPI fallback on Windows. Leaves nothing behind."""
    from cfb import settings

    out = []
    backend = settings.keyring_backend()
    if backend:
        try:
            import keyring

            keyring.set_password("cfb-selftest", "probe", "v")
            ok = keyring.get_password("cfb-selftest", "probe") == "v"
            keyring.delete_password("cfb-selftest", "probe")
            out.append(f"keyring:{backend}:{'ok' if ok else 'lost'}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"keyring:{backend}:error({type(exc).__name__})")
    else:
        out.append("keyring:none")
    if sys.platform == "win32":
        try:
            ok = settings._dpapi(settings._dpapi(b"probe", True), False) == b"probe"
            out.append(f"dpapi:{'ok' if ok else 'mismatch'}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"dpapi:error({type(exc).__name__})")
    return ",".join(out)


def main() -> int:
    _quiet_streams()
    args = sys.argv[1:]
    if args[:1] == ["sync"]:
        sys.argv = ["cfb.warehouse_sync", *args]
        runpy.run_module("cfb.warehouse_sync", run_name="__main__")
        return 0
    if args[:1] == ["engine"]:
        from cfb.engine import main as engine

        return engine(args[1:])
    if args[:1] == ["selftest"]:
        return selftest(system="--system" in args)
    from cfb.ui.launch import main as launch
    print("CFB Predictor is running. Close this window to stop it.", flush=True)
    return launch(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
