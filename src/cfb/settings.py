"""Where the package keeps its files, and where secrets come from.

One codebase runs in two places:

- Cloud / source checkout (the default): data dir `data`, run config `dashboard/config.toml`,
  snapshots and the dated-copy archive under `/mnt/project-files/cfbd-data` when that mount
  exists. These are the values the code used before this module existed.
- Packaged app (`sys.frozen`, the Windows .exe): data under the user's local app-data folder
  (`%LOCALAPPDATA%\\cfb\\data` on Windows), a user config under `%APPDATA%\\cfb\\config.toml`,
  the dated-copy archive under `%LOCALAPPDATA%\\cfb\\asof`, and the run config bundled with the
  app as the fallback.

The folder name (`cfb`) and the keyring service come from the edition (cfb.edition): the
experimental edition uses `cfb-exp` for both, so the two editions never share files or keys.

Every value can be overridden with an environment variable:

    CFB_DATA_DIR        data directory
    CFB_CONFIG          run configuration (TOML)
    CFB_PROJECT_FILES   project files root (default /mnt/project-files)
    CFB_SNAPSHOT_DIR    folder with LATEST / FEATURES_LATEST and the snapshot tars
    CFB_ASOF_ARCHIVE    dated rating copies folder
    CFB_WAREHOUSE_DIR   base folder of the synced warehouse copies (cfb.warehouse_sync)

Secrets (`secret()`): an environment variable first, then the OS keyring (Windows Credential
Manager) when the optional `keyring` package is installed, else None. In the cloud the CFBD key
is injected by a proxy, so None is the normal answer there. A non-main edition without its own
CFBD key reads main's (the CFBD key only reads public data); R2 keys are never shared.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import edition as _edition

EDITION = _edition.current()
APP = EDITION.folder
KEYRING_SERVICE = EDITION.keyring_service
# Secrets a non-main edition may read from main's keyring when it has none of its own.
SHARED_SECRETS = ("cfbd_api_key",)
DEFAULT_PROJECT_FILES = Path("/mnt/project-files")
REPO_CONFIG = Path("dashboard") / "config.toml"

# Secret name -> environment variable that overrides the keyring.
SECRET_ENV = {
    "cfbd_api_key": "CFBD_API_KEY",
    "r2_account_id": "R2_ACCOUNT_ID",
    "r2_access_key_id": "R2_ACCESS_KEY_ID",
    "r2_secret_access_key": "R2_SECRET_ACCESS_KEY",
    "r2_bucket": "R2_BUCKET",
    "cloudflare_api_token": "CLOUDFLARE_API_TOKEN",
}


def frozen() -> bool:
    """True inside the packaged app (PyInstaller sets sys.frozen)."""
    return bool(getattr(sys, "frozen", False))


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def user_data_root() -> Path:
    """Per-user folder for data and reports (packaged app)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP


def user_config_root() -> Path:
    """Per-user folder for the user's config file."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / APP


def bundle_root() -> Path | None:
    """Folder holding files bundled with the packaged app, else None."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if frozen() and meipass else None


def data_dir() -> Path:
    env = _env_path("CFB_DATA_DIR")
    if env is not None:
        return env
    return user_data_root() / "data" if frozen() else Path("data")


def warehouse_dir() -> Path:
    """Base folder for warehouse copies synced from R2 (versions/, CURRENT, sync.lock).
    Local-only files (for example QB-out overrides) must live outside it."""
    env = _env_path("CFB_WAREHOUSE_DIR")
    if env is not None:
        return env
    return user_data_root() / "warehouse"


def config_path() -> Path:
    """Run configuration: $CFB_CONFIG, then the user's config file, then the repo's
    dashboard/config.toml (current directory), then the copy bundled with the app.
    The repo path is returned when none exists (load_config then uses its defaults)."""
    env = _env_path("CFB_CONFIG")
    if env is not None:
        return env
    candidates = [user_config_root() / "config.toml"] if frozen() else []
    candidates.append(REPO_CONFIG)
    if bundle_root() is not None:
        candidates.append(bundle_root() / REPO_CONFIG)
    for c in candidates:
        if c.exists():
            return c
    return REPO_CONFIG


def project_files() -> Path | None:
    """The project files mount (cloud), or None where it does not exist."""
    p = _env_path("CFB_PROJECT_FILES") or DEFAULT_PROJECT_FILES
    return p if p.exists() else None


def snapshot_dir() -> Path | None:
    env = _env_path("CFB_SNAPSHOT_DIR")
    if env is not None:
        return env
    pf = project_files()
    return pf / "cfbd-data" if pf is not None else None


def asof_archive() -> Path | None:
    """Dated rating copies folder: $CFB_ASOF_ARCHIVE, the project files archive, the packaged
    app's own folder (%LOCALAPPDATA%\\cfb\\asof), else None (source checkout without one)."""
    env = _env_path("CFB_ASOF_ARCHIVE")
    if env is not None:
        return env
    snap = snapshot_dir()
    if snap is not None and (snap / "asof").exists():
        return snap / "asof"
    if frozen():
        return user_data_root() / "asof"
    return None


def _keyring():
    try:
        import keyring  # optional: pip install 'cfb[keyring]'
    except Exception:
        return None
    return keyring


def secret(name: str) -> str | None:
    """A secret by name (see SECRET_ENV): environment variable, then OS keyring, else None."""
    env = SECRET_ENV.get(name)
    if env and os.environ.get(env):
        return os.environ[env]
    kr = _keyring()
    if kr is None:
        return None
    services = [KEYRING_SERVICE]
    if name in SHARED_SECRETS and KEYRING_SERVICE != _edition.EDITIONS[_edition.MAIN].keyring_service:
        services.append(_edition.EDITIONS[_edition.MAIN].keyring_service)
    for service in services:
        try:
            value = kr.get_password(service, name)
        except Exception:  # no usable keyring backend (headless Linux, CI)
            return None
        if value:
            return value
    return None


def secret_source(name: str) -> str | None:
    """Where secret(name) comes from: "environment", "keyring", "main keyring" or None."""
    env = SECRET_ENV.get(name)
    if env and os.environ.get(env):
        return "environment"
    kr = _keyring()
    if kr is None:
        return None
    try:
        if kr.get_password(KEYRING_SERVICE, name):
            return "keyring"
        if name in SHARED_SECRETS and KEYRING_SERVICE != _edition.EDITIONS[_edition.MAIN].keyring_service \
                and kr.get_password(_edition.EDITIONS[_edition.MAIN].keyring_service, name):
            return "main keyring"
    except Exception:
        return None
    return None


def set_secret(name: str, value: str) -> None:
    """Store a secret in the OS keyring. Raises RuntimeError without a usable keyring."""
    if name not in SECRET_ENV:
        raise KeyError(f"unknown secret {name!r}; known: {', '.join(sorted(SECRET_ENV))}")
    kr = _keyring()
    if kr is None:
        raise RuntimeError("the keyring package is not installed (pip install 'cfb[keyring]')")
    kr.set_password(KEYRING_SERVICE, name, value)


def delete_secret(name: str) -> None:
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(KEYRING_SERVICE, name)
    except Exception:
        pass
