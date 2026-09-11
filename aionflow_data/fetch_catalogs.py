"""Step 0: download the four public catalogues with resume and checksum verification.

    python -m aionflow_data.fetch_catalogs [--config CONFIG] [--dry-run] [--only NAME ...]
    python -m aionflow_data.fetch_catalogs --clean-raw

Each input in the config is downloaded into `paths.raw` as `<file>.part` with HTTP
Range resume, verified against the configured md5 or sha256 and, where the publisher
ships a checksum sidecar, against that too, then renamed into place. The ledger
`data/provenance/raw.json` records size, both digests, URL and retrieval time per
file. A file already present with the recorded size and mtime is not re-hashed.

`--clean-raw` deletes `paths.raw` only when the crossmatch and labels ledgers exist
and record the same checksums, so the pipeline's outputs are provably derived from
the files being removed.
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import (
    ensure_dirs,
    file_digests,
    load_config,
    read_ledger,
    step_parser,
    utc_now,
    write_ledger,
)

STEP = "raw"
CHUNK = 1 << 20
USER_AGENT = "aionflow-data/0.1"
PERMANENT_HTTP = (400, 401, 403, 404, 410)
DOWNSTREAM_LEDGERS = ("crossmatch", "labels")


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


def _ledger_matches(prior: dict | None, name: str, dest: Path) -> dict | None:
    """The prior ledger entry for `name` if it still describes the file on disk."""
    if not prior:
        return None
    rec = (prior.get("inputs") or {}).get(name)
    if not rec:
        return None
    st = dest.stat()
    if rec.get("bytes") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns \
            and rec.get("path") == str(dest):
        return rec
    return None


# ----------------------------------------------------------------------------- run

def run(cfg: dict, *, only: list[str] | None = None, dry_run: bool = False,
        retries: int = 6, backoff_s: float = 5.0, timeout_s: float = 60.0, log=print) -> dict:
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])
    prior = read_ledger(STEP, cfg)
    entries: dict[str, dict] = {}
    for name, entry in cfg["inputs"].items():
        if only and name not in only:
            continue
        dest = raw / entry["file"]
        state = status(dest, int(entry["bytes"]))
        log(f"[fetch] {name:10s} {state:10s} {dest}")
        if dry_run:
            entries[name] = {"path": str(dest), "status": state, "url": entry["url"]}
            continue
        reuse = _ledger_matches(prior, name, dest) if state == "present" else None
        if reuse:
            entries[name] = dict(reuse, status="present")
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
        if entry.get("checksum_sidecar_url"):
            want = publisher_sha256(entry["checksum_sidecar_url"], entry["file"], timeout_s)
            if want is None:
                log(f"[fetch] {name}: publisher sidecar does not list {entry['file']}")
            elif want != digests["sha256"]:
                dest.unlink()
                raise FetchError(f"{name}: sha256 {digests['sha256']} differs from the "
                                 f"publisher's {want}; file removed")
            else:
                published = True
        st = dest.stat()
        entries[name] = {
            "path": str(dest), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha256": digests["sha256"], "md5": digests["md5"], "url": entry["url"],
            "retrieved_utc": utc_now(), "status": state,
            "publisher_sha256_verified": published,
        }
        log(f"[fetch] {name}: {state}, {st.st_size:,} bytes, sha256 {digests['sha256'][:16]}...")
    if not dry_run:
        # keep entries for inputs not selected by --only from the prior ledger
        if only and prior:
            for name, rec in (prior.get("inputs") or {}).items():
                entries.setdefault(name, rec)
        write_ledger(STEP, cfg, inputs=entries, counts={"files": len(entries)})
    return entries


def clean_raw(cfg: dict, log=print) -> int:
    """Delete `paths.raw` once every raw input is recorded, with the same sha256, by a
    downstream ledger. Returns the number of files removed."""
    raw_ledger = read_ledger(STEP, cfg)
    if raw_ledger is None:
        raise FetchError("refusing to clean: no raw ledger")
    downstream = {s: read_ledger(s, cfg) for s in DOWNSTREAM_LEDGERS}
    missing = [s for s, lg in downstream.items() if lg is None]
    if missing:
        raise FetchError(f"refusing to clean: ledgers not found for {missing}")
    for name, rec in raw_ledger["inputs"].items():
        seen = False
        for step, lg in downstream.items():
            ref = (lg.get("inputs") or {}).get(name)
            if ref is None:
                continue
            seen = True
            if ref.get("sha256") != rec.get("sha256"):
                raise FetchError(f"refusing to clean: {step} ledger records a different "
                                 f"sha256 for {name}")
        if not seen:
            raise FetchError(f"refusing to clean: no downstream ledger references {name}")
    raw = Path(cfg["paths"]["raw"])
    n = 0
    for rec in raw_ledger["inputs"].values():
        p = Path(rec["path"])
        if p.is_file():
            p.unlink()
            n += 1
            log(f"[clean-raw] removed {p}")
    if raw.is_dir() and not any(raw.iterdir()):
        shutil.rmtree(raw)
    return n


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report state; download nothing")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="subset of inputs")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--backoff", type=float, default=5.0, help="seconds, doubles per retry")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    parser.add_argument("--clean-raw", action="store_true",
                        help="delete raw files whose checksums downstream ledgers record")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        if args.clean_raw:
            n = clean_raw(cfg)
            print(f"[clean-raw] removed {n} file(s)")
            return 0
        run(cfg, only=args.only, dry_run=args.dry_run, retries=args.retries,
            backoff_s=args.backoff, timeout_s=args.timeout)
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
