"""Step 0: download the four public catalogues with resume and checksum verification.

    python -m aionflow_data.fetch_catalogs [--config CONFIG] [--dry-run]

Each input in the config is downloaded into `paths.raw` as `<file>.part` with HTTP
Range resume, verified against the configured md5 or sha256 and, where the publisher
ships a checksum sidecar, against that too, then renamed into place. A file already
present is re-hashed against the configured checksum and makes no request. The
ledger `data/provenance/raw.json` records size, both digests, URL and retrieval
time per file.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import ensure_dirs, file_digests, load_config, step_parser, utc_now, write_ledger

STEP = "raw"
CHUNK = 1 << 20
USER_AGENT = "aionflow-data/0.1"
PERMANENT_HTTP = (400, 401, 403, 404, 410)


class FetchError(RuntimeError):
    pass


def _request(url: str, headers: dict | None = None, method: str = "GET", timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})},
                                 method=method)
    return urllib.request.urlopen(req, timeout=timeout)


# ----------------------------------------------------------------------------- download

def download(url: str, dest: Path, expected_bytes: int, *, retries: int = 6,
             backoff_s: float = 5.0, timeout_s: float = 60.0, log=print) -> None:
    """Fetch `url` into `dest`, resuming a partial `dest.part` with a Range request."""
    part = dest.with_name(dest.name + ".part")
    attempt = 0
    restarted = False
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have > expected_bytes:
            # A stale partial file may be oversize once; a server that sends more than
            # the configured size is a config error, not something to retry.
            if restarted:
                part.unlink()
                raise FetchError(f"{dest.name}: server sent {have:,} bytes, more than the "
                                 f"configured {expected_bytes:,}")
            log(f"[fetch] {dest.name}: partial file larger than expected; restarting")
            part.unlink()
            have = 0
            restarted = True
        if have == expected_bytes:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _request(url, headers, timeout=timeout_s) as resp:
                if have and resp.status == 206:
                    mode = "ab"
                elif resp.status == 200:
                    mode = "wb"          # no Range support: start over
                    if have:
                        log(f"[fetch] {dest.name}: server ignored Range; restarting")
                else:
                    raise FetchError(f"unexpected HTTP status {resp.status}")
                with open(part, mode) as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
        except urllib.error.HTTPError as exc:
            if exc.code in PERMANENT_HTTP:
                raise FetchError(f"{url}: HTTP {exc.code}") from exc
            attempt = _retry_or_raise(attempt, retries, backoff_s, dest, url, exc, log)
        except (urllib.error.URLError, TimeoutError, OSError, FetchError) as exc:
            attempt = _retry_or_raise(attempt, retries, backoff_s, dest, url, exc, log)
    size = part.stat().st_size
    if size != expected_bytes:
        raise FetchError(f"{dest.name}: downloaded {size:,} bytes, expected {expected_bytes:,}")
    part.replace(dest)


def _retry_or_raise(attempt, retries, backoff_s, dest, url, exc, log) -> int:
    attempt += 1
    if attempt > retries:
        raise FetchError(f"{url}: giving up after {retries} retries: {exc}") from exc
    wait = backoff_s * 2 ** (attempt - 1)
    log(f"[fetch] {dest.name}: {type(exc).__name__}: {exc}; retry {attempt}/{retries} "
        f"in {wait:.1f}s")
    time.sleep(wait)
    return attempt


# ----------------------------------------------------------------------------- checks

def verify(dest: Path, entry: dict) -> dict[str, str]:
    """Both digests of `dest`; raise if a configured one disagrees. Deletes a bad file."""
    digests = file_digests(dest)
    for algorithm in ("md5", "sha256"):
        want = entry.get(algorithm)
        if want and digests[algorithm] != want:
            dest.unlink()
            raise FetchError(f"{dest.name}: {algorithm} {digests[algorithm]} differs from "
                             f"the configured {want}; file removed")
    return digests


def publisher_sha256(sidecar_url: str, filename: str, timeout_s: float = 60.0) -> str | None:
    """The sha256 the publisher's checksum sidecar lists for `filename`, if any."""
    with _request(sidecar_url, timeout=timeout_s) as resp:
        text = resp.read().decode("utf-8", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*").rsplit("/", 1)[-1] == filename:
            return parts[0].lower()
    return None


def status(dest: Path, expected_bytes: int) -> str:
    if dest.is_file():
        return "present" if dest.stat().st_size == expected_bytes else "wrong-size"
    part = dest.with_name(dest.name + ".part")
    if part.is_file():
        return "partial"
    return "missing"


# ----------------------------------------------------------------------------- run

def run(cfg: dict, *, dry_run: bool = False, retries: int = 6, backoff_s: float = 5.0,
        timeout_s: float = 60.0, log=print) -> dict:
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])
    entries: dict[str, dict] = {}
    for name, entry in cfg["inputs"].items():
        dest = raw / entry["file"]
        state = status(dest, int(entry["bytes"]))
        log(f"[fetch] {name:10s} {state:10s} {dest}")
        if dry_run:
            entries[name] = {"path": str(dest), "status": state, "url": entry["url"]}
            continue
        if state == "wrong-size":
            log(f"[fetch] {name}: size differs from the configured {entry['bytes']:,}; "
                f"re-downloading")
            dest.unlink()
            state = "missing"
        if state != "present":
            download(entry["url"], dest, int(entry["bytes"]), retries=retries,
                     backoff_s=backoff_s, timeout_s=timeout_s, log=log)
            state = "downloaded"
        digests = verify(dest, entry)
        published = None
        if state == "downloaded" and entry.get("checksum_sidecar_url"):
            want = publisher_sha256(entry["checksum_sidecar_url"], entry["file"], timeout_s)
            if want is None:
                log(f"[fetch] {name}: publisher sidecar does not list {entry['file']}")
            elif want != digests["sha256"]:
                dest.unlink()
                raise FetchError(f"{name}: sha256 {digests['sha256']} differs from the "
                                 f"publisher's {want}; file removed")
            else:
                published = True
        entries[name] = {
            "path": str(dest), "bytes": dest.stat().st_size, "sha256": digests["sha256"],
            "md5": digests["md5"], "url": entry["url"], "retrieved_utc": utc_now(),
            "status": state, "publisher_sha256_verified": published,
        }
        log(f"[fetch] {name}: {state}, {entries[name]['bytes']:,} bytes, "
            f"sha256 {digests['sha256'][:16]}...")
    if not dry_run:
        write_ledger(STEP, cfg, inputs=entries, counts={"files": len(entries)})
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report state; download nothing")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--backoff", type=float, default=5.0, help="seconds, doubles per retry")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg, dry_run=args.dry_run, retries=args.retries, backoff_s=args.backoff,
            timeout_s=args.timeout)
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
