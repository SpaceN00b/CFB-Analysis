"""Publish the Parquet warehouse to a bucket (Cloudflare R2) and sync it to a local copy.

    python -m cfb.warehouse_sync publish --data-dir data            # cloud: upload changed files
    python -m cfb.warehouse_sync sync --base %LOCALAPPDATA%/cfb/warehouse   # PC: pull changed files
    python -m cfb.warehouse_sync status [--base DIR]

The cloud is the only writer. What is published: every file under tables/, core/, derived/,
features/, predictions/ and reports/ of the data dir. raw/ (API JSON) and the dated-copy archive stay in
the cloud.

Bucket layout (content-addressed, so a reader never sees a half-written publish):
  blobs/<sha256>             file bytes, one object per distinct content
  manifests/<version>.json   every published manifest still kept
  manifests/index.json       {"versions": [...]} oldest first, at most KEEP_MANIFESTS
  manifest.json              the current manifest, written after its blobs

manifest.json: {"schema", "version", "key", "season", "season_type", "week", "created_utc",
"git_commit", "cfbd_quota", "total_bytes", "files": {"<relpath>": {"size", "sha256"}}}.
Blobs no kept manifest references are deleted after each publish.

Local layout under a base dir:
  versions/<version>/            a complete data dir (pass it to Warehouse), with its
                                 manifest.json (version, created time, CFBD quota)
  CURRENT                        the version in use, replaced atomically (os.replace)
  sync.lock
Unchanged files are hard-linked from the previous version (copied where linking fails), so
the files under versions/ must be treated as read-only.

Credentials (r2_settings): cfb.settings.secret (environment, then OS keyring) first, then a TOML
file ($CFB_R2_CONFIG, else <project files>/warehouse/r2.toml) for the non-secret account_id and bucket.
  R2_ACCOUNT_ID, R2_BUCKET (default cfb-warehouse)
  CLOUDFLARE_API_TOKEN                     Cloudflare REST API (Bearer). When absent the request
                                           carries no token, for a proxy that injects it.
  R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY  S3 API instead (needs boto3: pip install 'cfb[r2]')
  CFB_WAREHOUSE_LOCAL=<dir>                a folder that stands in for the bucket (tests, dry runs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from . import settings

PUBLISHED_DIRS = ("tables", "core", "derived", "features", "predictions", "reports")
SCHEMA = 1
KEEP_MANIFESTS = 3
DEFAULT_BUCKET = "cfb-warehouse"
DEFAULT_CONFIG = Path("/mnt/project-files/warehouse/r2.toml")
MANIFEST = "manifest.json"
INDEX = "manifests/index.json"
CHUNK = 1 << 20
WORKERS = 8
KEY_RE = re.compile(r"^(?P<season>\d{4})_(?P<season_type>[a-z]+)_(?P<week>\d+)$")


class WarehouseError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[warehouse {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ---- settings -------------------------------------------------------------------------------

@dataclass
class R2Settings:
    account_id: str | None = None
    bucket: str = DEFAULT_BUCKET
    api_token: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    local_bucket: Path | None = None

    @property
    def configured(self) -> bool:
        return self.local_bucket is not None or bool(self.account_id)

    @property
    def backend(self) -> str:
        if self.local_bucket is not None:
            return "local"
        if self.access_key_id and self.secret_access_key:
            return "s3"
        return "cloudflare-api"


def default_config() -> Path:
    pf = settings.project_files()
    return pf / "warehouse" / "r2.toml" if pf is not None else DEFAULT_CONFIG


def r2_settings(config_path: Path | None = None, env: dict | None = None) -> R2Settings:
    """Secrets via cfb.settings.secret (environment, then OS keyring); with `env` given (tests),
    only that dict. Then the config file for account_id and bucket."""
    if env is None:
        env, secret = os.environ, settings.secret
    else:
        secret = lambda name: env.get(settings.SECRET_ENV[name]) or None  # noqa: E731
    path = config_path or (Path(env["CFB_R2_CONFIG"]) if env.get("CFB_R2_CONFIG") else default_config())
    cfg: dict = {}
    if path.exists():
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    local = env.get("CFB_WAREHOUSE_LOCAL") or cfg.get("local_bucket")
    return R2Settings(
        account_id=secret("r2_account_id") or cfg.get("account_id"),
        bucket=secret("r2_bucket") or cfg.get("bucket") or DEFAULT_BUCKET,
        api_token=secret("cloudflare_api_token"),
        access_key_id=secret("r2_access_key_id"),
        secret_access_key=secret("r2_secret_access_key"),
        local_bucket=Path(local) if local else None,
    )


# ---- stores ---------------------------------------------------------------------------------

class Store(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/json") -> None: ...
    def put_file(self, key: str, path: Path) -> None: ...
    def download(self, key: str, dest: Path) -> None: ...
    def delete(self, key: str) -> None: ...


class LocalStore:
    """A folder standing in for the bucket."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        return self.root / key

    def get(self, key):
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def put_bytes(self, key, data, content_type="application/json"):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def put_file(self, key, path):
        self.put_bytes(key, Path(path).read_bytes())

    def download(self, key, dest):
        p = self._p(key)
        if not p.exists():
            raise WarehouseError(f"missing object {key}")
        shutil.copyfile(p, dest)

    def delete(self, key):
        self._p(key).unlink(missing_ok=True)


class CloudflareAPIStore:
    """R2 objects through the Cloudflare REST API (Bearer token; none when a proxy injects it)."""

    RETRY = {429, 500, 502, 503, 504}

    def __init__(self, account_id: str, bucket: str, token: str | None = None, session=None,
                 retries: int = 4, sleep: Callable[[float], None] = time.sleep, timeout: float = 300.0):
        import requests

        self.base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets/{bucket}/objects/"
        self._fixed = session
        self._local = threading.local()
        self._new = requests.Session
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.retries, self._sleep, self.timeout = retries, sleep, timeout

    @property
    def session(self):
        if self._fixed is not None:
            return self._fixed
        if not hasattr(self._local, "s"):  # one session per thread (uploads run in a pool)
            self._local.s = self._new()
        return self._local.s

    def _call(self, method: str, key: str, ok404: bool = False, **kw):
        last = ""
        headers = {**self.headers, **kw.pop("headers", {})}
        for attempt in range(self.retries + 1):
            data = kw.get("data")
            if hasattr(data, "seek"):
                data.seek(0)
            try:
                r = self.session.request(method, self.base + key, headers=headers,
                                         timeout=self.timeout, **kw)
            except Exception as exc:  # network errors: retry
                last = repr(exc)
            else:
                if r.status_code < 300 or (ok404 and r.status_code == 404):
                    return r
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in self.RETRY:
                    break
            if attempt < self.retries:
                self._sleep(2.0 * 2**attempt)
        raise WarehouseError(f"{method} {key} failed: {last}")

    def get(self, key):
        r = self._call("GET", key, ok404=True)
        return None if r.status_code == 404 else r.content

    def put_bytes(self, key, data, content_type="application/json"):
        self._call("PUT", key, data=data, headers={"Content-Type": content_type})

    def put_file(self, key, path):
        with open(path, "rb") as fh:
            self._call("PUT", key, data=fh, headers={"Content-Type": "application/octet-stream"})

    def download(self, key, dest):
        r = self._call("GET", key, stream=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)

    def delete(self, key):
        self._call("DELETE", key, ok404=True)


class S3Store:
    """R2 through its S3 API (boto3, SigV4 with the access key pair)."""

    def __init__(self, account_id: str, bucket: str, access_key_id: str, secret_access_key: str):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise WarehouseError("the S3 backend needs boto3: pip install 'cfb[r2]'") from exc
        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                               aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
                               region_name="auto")

    def get(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put_bytes(self, key, data, content_type="application/json"):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def put_file(self, key, path):
        self.s3.upload_file(str(path), self.bucket, key)

    def download(self, key, dest):
        self.s3.download_file(self.bucket, key, str(dest))

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def open_store(settings: R2Settings | None = None) -> Store:
    s = settings or r2_settings()
    if s.local_bucket is not None:
        return LocalStore(s.local_bucket)
    if not s.account_id:
        raise WarehouseError("no R2 account: set R2_ACCOUNT_ID or account_id in the r2.toml config")
    if s.backend == "s3":
        return S3Store(s.account_id, s.bucket, s.access_key_id, s.secret_access_key)
    return CloudflareAPIStore(s.account_id, s.bucket, s.api_token)


# ---- manifest -------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(data_dir: Path) -> dict[str, dict]:
    """relpath -> {size, sha256} for every published file under data_dir."""
    data_dir = Path(data_dir)
    out: dict[str, dict] = {}
    for top in PUBLISHED_DIRS:
        root = data_dir / top
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.startswith(".") or p.name.endswith((".tmp", ".part")):
                continue
            rel = p.relative_to(data_dir).as_posix()
            out[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def latest_report_key(data_dir: Path) -> str | None:
    root = Path(data_dir) / "reports"
    keys = [p for p in root.iterdir() if p.is_dir() and KEY_RE.match(p.name)] if root.is_dir() else []
    return max(keys, key=lambda p: p.stat().st_mtime).name if keys else None


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def cfbd_quota() -> dict | None:
    """CFBD /info (does not count as a call); None when it cannot be read."""
    try:
        from .client import CFBDClient

        info = CFBDClient(max_retries=1, timeout=20).get("info")
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    return {"tier": info.get("tierName"), "monthly_limit": info.get("monthlyLimit"),
            "remaining": info.get("remainingCalls"), "reset_at": info.get("resetAt")}


def make_manifest(files: dict[str, dict], key: str | None, now: datetime, quota: dict | None,
                  commit: str | None) -> dict:
    m = KEY_RE.match(key or "")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema": SCHEMA,
        "version": f"{key}-{stamp}" if key else stamp,
        "key": key,
        "season": int(m["season"]) if m else None,
        "season_type": m["season_type"] if m else None,
        "week": int(m["week"]) if m else None,
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "cfbd_quota": quota,
        "total_bytes": sum(f["size"] for f in files.values()),
        "files": files,
    }


def _json(data: dict) -> bytes:
    return json.dumps(data, indent=1, sort_keys=True).encode("utf-8")


def _load(store: Store, key: str) -> dict | None:
    raw = store.get(key)
    return json.loads(raw) if raw else None


# ---- publish --------------------------------------------------------------------------------

@dataclass
class PublishResult:
    version: str
    changed: bool
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    deleted_blobs: int = 0
    manifest: dict = field(default_factory=dict)


def publish(data_dir: str | Path, store: Store, key: str | None = None, *, now: datetime | None = None,
            quota: dict | None = None, commit: str | None = None, force: bool = False, dry_run: bool = False,
            workers: int = WORKERS, log: Callable[[str], None] = _log) -> PublishResult:
    """Upload the files that changed since the bucket's current manifest, then the manifest."""
    data_dir = Path(data_dir)
    now = now or datetime.now(timezone.utc)
    key = key or latest_report_key(data_dir)
    files = scan(data_dir)
    if not files:
        raise WarehouseError(f"nothing to publish under {data_dir} ({', '.join(PUBLISHED_DIRS)})")
    prev = _load(store, MANIFEST)
    if prev and not force and prev.get("files") == files and prev.get("key") == key:
        log(f"no changes since {prev['version']}: nothing uploaded")
        return PublishResult(prev["version"], False, manifest=prev)

    index = _load(store, INDEX) or {"versions": []}
    kept = [_load(store, f"manifests/{v}.json") for v in index["versions"]]
    kept = [k for k in kept if k]
    if prev and all(k["version"] != prev["version"] for k in kept):
        kept.append(prev)
    present = {f["sha256"] for k in kept for f in k["files"].values()}

    manifest = make_manifest(files, key, now, quota, commit)
    todo: dict[str, str] = {}
    for rel, f in files.items():
        if f["sha256"] not in present and f["sha256"] not in todo:
            todo[f["sha256"]] = rel
    nbytes = sum(files[rel]["size"] for rel in todo.values())
    log(f"{manifest['version']}: {len(files)} files, {manifest['total_bytes'] / 1e6:.1f} MB; "
        f"{len(todo)} new blobs, {nbytes / 1e6:.1f} MB to upload")
    if dry_run:
        return PublishResult(manifest["version"], True, len(todo), nbytes, manifest=manifest)

    t0, done, lock = time.monotonic(), [0, 0], threading.Lock()

    def upload(item):
        sha, rel = item
        store.put_file(f"blobs/{sha}", data_dir / rel)
        with lock:
            done[0] += 1
            done[1] += files[rel]["size"]
            if done[0] % 200 == 0 or done[0] == len(todo):
                log(f"  uploaded {done[0]}/{len(todo)} ({done[1] / 1e6:.1f} MB, {time.monotonic() - t0:.0f} s)")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(upload, todo.items()))  # re-raises the first failure; manifest not written

    store.put_bytes(f"manifests/{manifest['version']}.json", _json(manifest))
    store.put_bytes(MANIFEST, _json(manifest))
    versions = list(index["versions"])
    if prev and prev["version"] not in versions:  # index lost or written by hand: keep prev's blobs
        versions.append(prev["version"])
    versions = [v for v in versions if v != manifest["version"]] + [manifest["version"]]
    keep, drop = versions[-KEEP_MANIFESTS:], versions[:-KEEP_MANIFESTS]
    store.put_bytes(INDEX, _json({"versions": keep}))

    live = {f["sha256"] for k in kept if k["version"] in keep for f in k["files"].values()}
    live |= {f["sha256"] for f in files.values()}
    stale = {f["sha256"] for k in kept if k["version"] not in keep for f in k["files"].values()} - live
    for sha in stale:
        store.delete(f"blobs/{sha}")
    for v in drop:
        store.delete(f"manifests/{v}.json")
    log(f"published {manifest['version']}; pruned {len(stale)} blobs, {len(drop)} old manifests")
    return PublishResult(manifest["version"], True, len(todo), nbytes, len(stale), manifest)


# ---- sync -----------------------------------------------------------------------------------

@dataclass
class SyncResult:
    version: str
    changed: bool
    downloaded_files: int = 0
    downloaded_bytes: int = 0
    linked_files: int = 0
    data_dir: Path | None = None


def current_version(base: str | Path) -> str | None:
    p = Path(base) / "CURRENT"
    v = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return v or None


def current_data_dir(base: str | Path) -> Path | None:
    """The synced data dir in use (pass it to Warehouse), or None before the first sync."""
    v = current_version(base)
    d = Path(base) / "versions" / v if v else None
    return d if d is not None and d.is_dir() else None


def local_manifest(base: str | Path, version: str | None = None) -> dict | None:
    v = version or current_version(base)
    p = Path(base) / "versions" / v / "manifest.json" if v else None
    return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.exists() else None


class SyncBusy(WarehouseError):
    """Another process (or thread) holds base/sync.lock."""


class _Lock:
    """An OS lock on base/sync.lock (msvcrt on Windows, flock elsewhere). The OS drops it when the
    holder exits or crashes, so a closed app or killed sync never leaves a lock that blocks the
    next one. The file itself stays; only the lock on it matters."""

    def __init__(self, path: Path):
        self.path, self.fd = path, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            _os_lock(fd)
        except OSError as exc:
            os.close(fd)
            raise SyncBusy("another sync is running (another app window or `cfb sync`); "
                           "wait for it to finish") from exc
        self.fd = fd
        return self

    def __exit__(self, *exc):
        try:
            _os_unlock(self.fd)
        finally:
            os.close(self.fd)
            self.fd = None


if sys.platform == "win32":
    import msvcrt

    def _os_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _os_unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _os_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _os_unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _matches(path: Path, f: dict) -> bool:
    return path.exists() and path.stat().st_size == f["size"] and sha256_file(path) == f["sha256"]


def sync(base: str | Path, store: Store, *, keep_versions: int = 2, workers: int = WORKERS,
         log: Callable[[str], None] = _log) -> SyncResult:
    """Download the bucket's current version into base/versions/<version>/ and switch CURRENT to it."""
    base = Path(base)
    with _Lock(base / "sync.lock"):
        remote = _load(store, MANIFEST)
        if not remote:
            raise WarehouseError("the bucket has no manifest.json: nothing has been published")
        if remote.get("schema") != SCHEMA:
            raise WarehouseError(f"manifest schema {remote.get('schema')} is not {SCHEMA}: update the app")
        version = remote["version"]
        versions = base / "versions"
        if current_version(base) == version and current_data_dir(base) is not None:
            log(f"up to date ({version})")
            return SyncResult(version, False, data_dir=current_data_dir(base))

        cur_dir, cur = current_data_dir(base), local_manifest(base) or {"files": {}}
        by_sha = {f["sha256"]: cur_dir / rel for rel, f in cur["files"].items()} if cur_dir else {}
        staging = versions / f".staging-{version}"
        staging.mkdir(parents=True, exist_ok=True)
        got: dict[str, Path] = {}
        n_dl = n_link = nbytes = 0
        todo_bytes = sum(f["size"] for f in remote["files"].values() if f["sha256"] not in by_sha)
        log(f"syncing {version}: {len(remote['files'])} files, up to {todo_bytes / 1e6:.1f} MB to download")
        later: list[tuple[str, dict]] = []
        fetch: dict[str, str] = {}  # sha -> first relpath that needs a download
        for rel, f in sorted(remote["files"].items()):
            dst = staging / rel
            if _matches(dst, f):  # left by an interrupted sync
                got[f["sha256"]] = dst
                continue
            dst.unlink(missing_ok=True)
            src = by_sha.get(f["sha256"])
            if src is not None and src.exists():
                _link_or_copy(src, dst)
                got.setdefault(f["sha256"], dst)
                n_link += 1
            elif f["sha256"] in fetch or f["sha256"] in got:
                later.append((rel, f))
            else:
                fetch[f["sha256"]] = rel

        def download(item):
            sha, rel = item
            f, dst = remote["files"][rel], staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            store.download(f"blobs/{sha}", part)
            if not _matches(part, f):
                part.unlink(missing_ok=True)
                raise WarehouseError(f"checksum mismatch for {rel}; nothing was switched")
            os.replace(part, dst)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(download, fetch.items()))
        for sha, rel in fetch.items():
            got.setdefault(sha, staging / rel)
            n_dl += 1
            nbytes += remote["files"][rel]["size"]
        for rel, f in later:  # same content as another file of this version
            _link_or_copy(got[f["sha256"]], staging / rel)
            n_link += 1
        extra = [p for p in staging.rglob("*") if p.is_file()
                 and p.relative_to(staging).as_posix() not in remote["files"].keys() | {"manifest.json"}]
        for p in extra:
            p.unlink()

        final = versions / version
        if final.exists():
            shutil.rmtree(final)
        (staging / "manifest.json").write_bytes(_json(remote))
        os.replace(staging, final)
        tmp = base / "CURRENT.tmp"
        tmp.write_text(version + "\n", encoding="utf-8")
        os.replace(tmp, base / "CURRENT")
        log(f"switched to {version}: downloaded {n_dl} files ({nbytes / 1e6:.1f} MB), reused {n_link}")
        _prune_local(versions, version, keep_versions, log)
        return SyncResult(version, True, n_dl, nbytes, n_link, final)


def _prune_local(versions: Path, current: str, keep: int, log) -> None:
    dirs = sorted((p for p in versions.iterdir() if p.is_dir() and not p.name.startswith(".")),
                  key=lambda p: p.stat().st_mtime)
    old = [p for p in dirs if p.name != current][: max(0, len(dirs) - keep)]
    for p in old:
        try:
            shutil.rmtree(p)
        except OSError as exc:  # open files on Windows: try again next sync
            log(f"could not remove {p.name} yet ({exc.__class__.__name__})")


# ---- CLI ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.warehouse_sync", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="upload changed warehouse files and a new manifest (cloud)")
    p.add_argument("--data-dir", default=None, help="default: cfb.settings.data_dir()")
    p.add_argument("--key", help="report key <season>_<type>_<week> (default: newest under data/reports)")
    p.add_argument("--force", action="store_true", help="write a new manifest even when nothing changed")
    p.add_argument("--dry-run", action="store_true", help="show what would be uploaded")
    s = sub.add_parser("sync", help="download the current published version (PC)")
    s.add_argument("--base", default=None, help="local warehouse folder (default: cfb.settings.warehouse_dir())")
    st = sub.add_parser("status", help="show the bucket's and the local version")
    st.add_argument("--base", default=None, help="default: cfb.settings.warehouse_dir()")
    args = ap.parse_args(argv)

    settings = r2_settings()
    try:
        store = open_store(settings)
        if args.cmd == "publish":
            r = publish(args.data_dir or settings.data_dir(), store, args.key, quota=cfbd_quota(), commit=git_commit(),
                        force=args.force, dry_run=args.dry_run)
            print(json.dumps({"version": r.version, "changed": r.changed, "uploaded_files": r.uploaded_files,
                              "uploaded_bytes": r.uploaded_bytes, "deleted_blobs": r.deleted_blobs}))
        elif args.cmd == "sync":
            r = sync(args.base or settings.warehouse_dir(), store)
            print(json.dumps({"version": r.version, "changed": r.changed, "downloaded_files": r.downloaded_files,
                              "downloaded_bytes": r.downloaded_bytes, "data_dir": str(r.data_dir)}))
        else:
            remote = _load(store, MANIFEST)
            out = {"backend": settings.backend, "bucket": settings.bucket,
                   "remote_version": remote and remote["version"],
                   "remote_bytes": remote and remote["total_bytes"]}
            out["local_version"] = current_version(args.base or settings.warehouse_dir())
            print(json.dumps(out))
    except WarehouseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
