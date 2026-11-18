"""Entry point of the packaged Windows app (PyInstaller, packaging/cfb_app.spec).

    "CFB Predictor.exe"                start the app: local server, opens the browser
    "CFB Predictor.exe" --no-browser   start without opening the browser (smoke test)
    "CFB Predictor.exe" sync [...]     download the newest database (Task Scheduler)
    "CFB Predictor.exe" selftest       import everything the app needs, then exit (CI)
"""
import multiprocessing
import runpy
import sys


def selftest() -> int:
    """Catch modules PyInstaller left out: import the app, the model and every feature group."""
    import streamlit  # noqa: F401
    from cfb import settings
    from cfb.features.base import discover_groups
    from cfb.model import config as C
    from cfb.report.config import load_config
    from cfb.ui import data, launch, pages, update  # noqa: F401
    import cfb.predict  # noqa: F401

    try:
        from cfb import warehouse_sync  # noqa: F401  (R2 sync; present once that step is merged)
        import boto3  # noqa: F401  (S3 backend used on the PC)
        sync_ok = "yes"
    except ImportError as exc:
        sync_ok = f"no ({exc.name})"
    groups = discover_groups()
    cfg = load_config()
    app = launch.app_path()
    print(f"groups={len(groups)} models={sorted(C.MODELS)} primary={cfg.primary} "
          f"config={settings.config_path()} app={app} build={update.current_build()} sync={sync_ok}")
    if not app.exists() or not groups:
        print("selftest failed", file=sys.stderr)
        return 1
    print("selftest ok")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["sync"]:
        try:
            import cfb.warehouse_sync  # noqa: F401  (added by the R2 sync step)
        except ImportError:
            print("Database sync is not in this build yet.", file=sys.stderr)
            return 1
        sys.argv = ["cfb.warehouse_sync", *args]
        runpy.run_module("cfb.warehouse_sync", run_name="__main__")
        return 0
    if args[:1] == ["selftest"]:
        return selftest()
    from cfb.ui.launch import main as launch
    print("CFB Predictor is running. Close this window to stop it.", flush=True)
    return launch(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
