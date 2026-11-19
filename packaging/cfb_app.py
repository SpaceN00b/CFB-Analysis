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
    from cfb.ui import availability, connection  # noqa: F401  (Players page, Cloudflare connection)
    import cfb.lab.depth_impact  # noqa: F401  (the Players page recompute)

    groups = discover_groups()
    cfg = load_config()
    app = launch.app_path()
    keys = key_store_check()
    print(f"groups={len(groups)} models={sorted(C.MODELS)} primary={cfg.primary} "
          f"config={settings.config_path()} app={app} build={update.current_build()} sync={sync_ok} keys={keys}")
    if not app.exists() or not groups or (sys.platform == "win32" and "ok" not in keys):
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
