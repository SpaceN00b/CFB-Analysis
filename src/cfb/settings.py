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
Manager) when the optional `keyring` package is installed and has a working backend, then the
user's secrets file, else None. `set_secret` writes to the keyring and reads the value back; when
the keyring is missing or does not keep the value, it writes the secrets file instead
(`%APPDATA%\\<folder>\\secrets.json`, each value encrypted for the Windows user with DPAPI; on
other systems a file only the user can read). Either way the value is still there on the next
launch. In the cloud the CFBD key is injected by a proxy, so None is the normal answer there.
A non-main edition without its own CFBD key reads main's (keyring, then main's secrets file; the
CFBD key only reads public data); R2 keys are never shared.

    CFB_SECRETS_FILE    the secrets file (tests)
"""
from __future__ import annotations

import os
import re
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
    "sportradar_api_key": "SPORTRADAR_API_KEY",
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


def keyring_backend() -> str | None:
    """Name of the keyring backend in use, or None when there is no usable one (not installed,
    or only keyring's fail / null backends, as on headless Linux)."""
    kr = _keyring()
    if kr is None:
        return None
    try:
        b = kr.get_keyring()
    except Exception:
        return None
    if b is None:
        return None
    mod = type(b).__module__
    if mod.endswith((".fail", ".null")):
        return None
    if mod.endswith(".chainer") and not getattr(b, "backends", None):
        return None
    return f"{mod.rsplit('.', 1)[-1]}.{type(b).__name__}"


# ---- secrets file (fallback when the keyring does not keep a value) --------------------------

def secrets_file() -> Path:
    env = _env_path("CFB_SECRETS_FILE")
    return env if env is not None else user_config_root() / "secrets.json"


def _dpapi(data: bytes, protect: bool) -> bytes:
    """Windows DPAPI (CryptProtectData / CryptUnprotectData): only this Windows user can decrypt."""
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    buf = ctypes.create_string_buffer(data, len(data))
    src = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = BLOB()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(src), None, None, None, None, 0x01, ctypes.byref(out))  # 0x01: no UI
    if not ok:
        raise OSError(f"DPAPI {'protect' if protect else 'unprotect'} failed (error {ctypes.GetLastError()})")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def _encode(value: str) -> dict:
    import base64

    if sys.platform == "win32":
        return {"dpapi": base64.b64encode(_dpapi(value.encode("utf-8"), True)).decode("ascii")}
    return {"plain": value}


def _decode(entry) -> str | None:
    import base64

    if not isinstance(entry, dict):
        return None
    if "dpapi" in entry:
        if sys.platform != "win32":
            return None
        return _dpapi(base64.b64decode(entry["dpapi"]), False).decode("utf-8")
    v = entry.get("plain")
    return v if isinstance(v, str) else None


def _file_read() -> dict:
    import json

    p = secrets_file()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _file_write(d: dict) -> None:
    import json

    p = secrets_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    os.replace(tmp, p)


def _file_secret(name: str) -> str | None:
    try:
        return _decode(_file_read().get(name)) or None
    except Exception:  # unreadable entry (another user's DPAPI value, corrupt file)
        return None


def _file_set(name: str, value: str | None) -> None:
    d = _file_read()
    if value is None:
        if name not in d:
            return
        d.pop(name)
    else:
        d[name] = _encode(value)
    _file_write(d)


# ---- secrets -----------------------------------------------------------------------------------

def _main_service() -> str:
    return _edition.EDITIONS[_edition.MAIN].keyring_service


def _shares_main(name: str) -> bool:
    """True when this edition may fall back to main's copy of the secret (the CFBD key only)."""
    return name in SHARED_SECRETS and KEYRING_SERVICE != _main_service()


def main_secrets_file() -> Path:
    """Main edition's secrets file (read only, for SHARED_SECRETS)."""
    return user_config_root().with_name(_edition.EDITIONS[_edition.MAIN].folder) / "secrets.json"


def _keyring_get(name: str, service: str | None = None) -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(service or KEYRING_SERVICE, name) or None
    except Exception:  # no usable keyring backend (headless Linux, CI)
        return None


def _main_file_secret(name: str) -> str | None:
    import json

    try:
        d = json.loads(main_secrets_file().read_text(encoding="utf-8"))
        return _decode(d.get(name)) if isinstance(d, dict) else None
    except Exception:
        return None


# Secrets sent as `Authorization: Bearer <value>`: a pasted "Bearer " prefix is dropped.
_BEARER_SECRETS = ("cfbd_api_key", "cloudflare_api_token")
_ACCOUNT_ID = re.compile(r"\b([0-9a-f]{32})\b", re.I)


def clean_secret(name: str, value: str | None) -> str | None:
    """A pasted value the way the code uses it: no surrounding whitespace or quotes, no "Bearer "
    prefix on tokens, and for r2_account_id the 32-character ID out of an S3 endpoint
    (https://<account id>.r2.cloudflarestorage.com). Applied when saving and when reading, so a
    value saved before this existed works without saving it again."""
    if value is None:
        return None
    v = value.strip().strip("\"'").strip()
    if name in _BEARER_SECRETS and v[:7].lower() == "bearer ":
        v = v[7:].strip()
    if name == "r2_account_id":
        m = _ACCOUNT_ID.search(v)
        if m:
            v = m.group(1).lower()
    return v or None


def _lookup(name: str) -> tuple[str | None, str | None]:
    """(value, source) for a secret, cleaned (clean_secret)."""
    value, source = _lookup_raw(name)
    value = clean_secret(name, value)
    return value, (source if value else None)


def _lookup_raw(name: str) -> tuple[str | None, str | None]:
    """(value, source) for a secret: environment, keyring, file, then main's keyring / main file."""
    env = SECRET_ENV.get(name)
    if env and os.environ.get(env):
        return os.environ[env], "environment"
    v = _keyring_get(name)
    if v:
        return v, "keyring"
    v = _file_secret(name)
    if v:
        return v, "file"
    if _shares_main(name):
        v = _keyring_get(name, _main_service())
        if v:
            return v, "main keyring"
        if os.environ.get("CFB_SECRETS_FILE") is None:  # tests point at their own file only
            v = _main_file_secret(name)
            if v:
                return v, "main file"
    return None, None


def secret(name: str) -> str | None:
    """A secret by name (see SECRET_ENV): environment variable, then OS keyring, then the secrets
    file (then main's copies, for the CFBD key in a non-main edition), else None."""
    return _lookup(name)[0]


def secret_source(name: str) -> str | None:
    """Where secret(name) comes from: "environment", "keyring", "file", "main keyring",
    "main file", or None when it is not set."""
    return _lookup(name)[1]


def set_secret(name: str, value: str) -> str:
    """Store a secret so it is there on the next launch. Returns where it went: "keyring" (Windows
    Credential Manager), else "file" (the secrets file, DPAPI-encrypted on Windows)."""
    if name not in SECRET_ENV:
        raise KeyError(f"unknown secret {name!r}; known: {', '.join(sorted(SECRET_ENV))}")
    value = clean_secret(name, value)
    if not value:
        raise ValueError(f"{name}: the value is empty")
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, name, value)
            if kr.get_password(KEYRING_SERVICE, name) == value:
                _file_set(name, None)  # no stale copy left in the file
                return "keyring"
        except Exception:
            pass
    _file_set(name, value)
    return "file"


def delete_secret(name: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, name)
        except Exception:
            pass
    try:
        _file_set(name, None)
    except OSError:
        pass
