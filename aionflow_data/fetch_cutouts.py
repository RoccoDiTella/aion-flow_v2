"""Step 4: fetch Legacy Survey DR10 griz cutouts, one FITS file per target.

    python -m aionflow_data.fetch_cutouts [--config CONFIG] [--limit N]

The cutout service is rate limited: a second concurrent request answers 429, so
requests are sequential with a pause between them. At roughly 5 s per cutout the
full sample takes about eight days, so the job is designed to be left running and
interrupted freely: one file per target under <work>/cutouts, written to a temp
name and renamed, resume by file existence. 429 and 5xx back off and retry. A
response under `min_bytes`, or one that does not parse as a (4, size, size) image
with the expected bands, is retried and then counted as failed; 404 is a
permanent failure. Failures do not block the pipeline: a target with no cutout
stages with a zero image and `has_image` false. The ledger records how many were
fetched, how many failed, and the failure list.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .common import ensure_dirs, load_config, step_parser, write_ledger
from .crossmatch import OUTPUT as CROSSMATCH_OUTPUT

STEP = "cutouts"
CUTOUT_DIR = "cutouts"
USER_AGENT = "aionflow-data/0.1"
RETRY_HTTP = (429, 500, 502, 503, 504)
MAX_LISTED_FAILURES = 200


class CutoutError(RuntimeError):
    pass


class BadCutout(ValueError):
    pass


def cutout_path(cutout_dir: Path, targetid) -> Path:
    return cutout_dir / f"{int(targetid)}.fits"


def read_cutout(path: Path, size: int, bands: str = "griz") -> np.ndarray:
    """The image as float32 (n_bands, size, size); BadCutout if it is not one."""
    with fits.open(path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32)
        got = str(hdul[0].header.get("BANDS", "")).lower()
    if image.shape != (len(bands), size, size):
        raise BadCutout(f"{path.name}: shape {image.shape}, expected {(len(bands), size, size)}")
    if got and got != bands:
        raise BadCutout(f"{path.name}: bands {got!r}, expected {bands!r}")
    if not np.isfinite(image).all():
        raise BadCutout(f"{path.name}: non-finite pixels")
    return image


def fetch_one(url: str, dest: Path, *, min_bytes: int, size: int, bands: str,
              attempts: int = 5, backoff_s: float = 5.0, max_backoff_s: float = 120.0,
              timeout_s: float = 120.0) -> str:
    """Returns 'ok' or 'skip' (already present); raises CutoutError after exhausting retries."""
    if dest.exists():
        return "skip"
    tmp = dest.with_name(dest.name + ".tmp")
    delay = backoff_s
    last: Exception = RuntimeError("no attempt made")
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                blob = resp.read()
            if len(blob) < min_bytes:
                raise BadCutout(f"response of {len(blob)} bytes")
            tmp.write_bytes(blob)
            read_cutout(tmp, size, bands)              # reject before the rename
            tmp.replace(dest)
            return "ok"
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRY_HTTP:
                raise CutoutError(f"HTTP {exc.code}") from exc   # 404 etc: permanent
        except Exception as exc:                        # timeout, truncated, unparsable
            last = exc
        tmp.unlink(missing_ok=True)
        time.sleep(delay)
        delay = min(delay * 2, max_backoff_s)
    raise CutoutError(f"giving up after {attempts} attempts: {type(last).__name__}: {last}")


def run(cfg: dict, *, limit: int | None = None, sleep_s: float | None = None,
        backoff_s: float | None = None, log=print) -> dict:
    ensure_dirs(cfg)
    work = Path(cfg["paths"]["work"])
    xm_path = work / CROSSMATCH_OUTPUT
    if not xm_path.is_file():
        raise CutoutError(f"missing input {xm_path}; run crossmatch first")
    c = cfg["cutouts"]
    template = cfg["archives"]["ls_cutout_url"]
    params = dict(layer=c["layer"], pixscale=c["pixscale"], size=int(c["size"]), bands=c["bands"])
    sleep_s = float(c["sleep_s"]) if sleep_s is None else sleep_s
    backoff_s = float(c["backoff_s"]) if backoff_s is None else backoff_s
    cutout_dir = work / CUTOUT_DIR
    cutout_dir.mkdir(parents=True, exist_ok=True)

    frame = (pd.read_parquet(xm_path, columns=["targetid", "target_ra", "target_dec"])
             .drop_duplicates("targetid").reset_index(drop=True))
    present = frame["targetid"].map(lambda t: cutout_path(cutout_dir, t).exists()).to_numpy()
    todo = frame[~present]
    if limit:
        todo = todo.head(limit)
    stats = {"targets": int(len(frame)), "present_before": int(present.sum()),
             "to_fetch": int(len(todo)), "fetched": 0, "failed": 0, "failures": [],
             "limit": limit}
    log(f"[cutouts] {stats['targets']:,} targets, {stats['present_before']:,} present, "
        f"{stats['to_fetch']:,} to fetch")
    t0 = time.time()
    for i, row in enumerate(todo.itertuples(index=False), 1):
        dest = cutout_path(cutout_dir, row.targetid)
        url = template.format(ra=float(row.target_ra), dec=float(row.target_dec), **params)
        try:
            if fetch_one(url, dest, min_bytes=int(c["min_bytes"]), size=params["size"],
                         bands=params["bands"], backoff_s=backoff_s,
                         max_backoff_s=float(c["max_backoff_s"]),
                         timeout_s=float(c["timeout_s"])) == "ok":
                stats["fetched"] += 1
        except CutoutError as exc:
            stats["failed"] += 1
            stats["failures"].append([int(row.targetid), str(exc)[:120]])
            log(f"[cutouts]   failed {int(row.targetid)}: {exc}")
        if sleep_s:
            time.sleep(sleep_s)                          # the service 429s without a gap
        if i % 100 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = stats["fetched"] / max(elapsed, 1e-9)
            eta_h = (len(todo) - i) / max(rate, 1e-9) / 3600
            log(f"[cutouts]   {i:,}/{len(todo):,} | {stats['fetched']:,} fetched | "
                f"{stats['failed']} failed | {rate * 3600:.0f}/h | ETA {eta_h:.1f} h")
    stats["present_after"] = int(sum(cutout_path(cutout_dir, t).exists()
                                     for t in frame["targetid"]))
    write_ledger(STEP, cfg, inputs={"crossmatch": xm_path},
                 counts={k: stats[k] for k in ("targets", "present_before", "to_fetch",
                                               "fetched", "failed", "present_after")},
                 extra={"url_template": template, "cutout": params,
                        "min_bytes": int(c["min_bytes"]), "limit": limit,
                        "failures": stats["failures"][:MAX_LISTED_FAILURES],
                        "failures_listed": min(len(stats["failures"]), MAX_LISTED_FAILURES)})
    log(f"[cutouts] done: {stats['fetched']:,} fetched, {stats['failed']} failed, "
        f"{stats['present_after']:,} of {stats['targets']:,} targets have a cutout")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="fetch at most N (a smoke)")
    parser.add_argument("--sleep", type=float, default=None, help="seconds between requests")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg, limit=args.limit, sleep_s=args.sleep)
    except CutoutError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
