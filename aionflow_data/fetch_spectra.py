"""Step 3: fetch DESI DR1 coadd spectra for the crossmatched targets.

    python -m aionflow_data.fetch_spectra [--config CONFIG] [--limit-groups N] [--workers N]
                                          [--no-merge | --merge-only]

A DESI healpix coadd is ~500 MB and holds a few thousand spectra; we want a
handful of them. Each coadd is opened lazily over HTTP (the archive honours
Range requests) and only the FIBERMAP and our rows of the camera flux and ivar
arrays are read. The B, R and Z cameras are combined by inverse-variance
weighting onto one uniform grid (overlaps coadded, not trimmed).

Output is sharded: one atomic `.npz` per (survey, program, healpix) under
<work>/spectra/shards, so a shard either exists complete or not at all, resume
is by file existence, and a kill cannot corrupt anything. A coadd the archive
does not have (confirmed by a HEAD request, since transient errors also surface
as "not found") is recorded as an empty shard. Transient failures leave no shard
and block the merge; rerun the same command to retry them.

When every group has a shard, the shards are merged into <work>/spectra/source.h5
(`desi_targetid`, `spectra`, `spectra_ivar`, `spectra_lambda`); a target observed
in two groups keeps the copy with more positive-ivar pixels. The merge streams
one shard at a time. The ledger data/provenance/spectra.json records the counts.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

from .common import describe_file, ensure_dirs, load_config, step_parser, write_ledger
from .crossmatch import OUTPUT as CROSSMATCH_OUTPUT

STEP = "spectra"
SHARD_DIR = "spectra/shards"
SOURCE = "spectra/source.h5"
CAMERAS = ("B", "R", "Z")
USER_AGENT = "aionflow-data/0.1"


class SpectraError(RuntimeError):
    pass


class MissingFile(Exception):
    """The coadd genuinely does not exist: never worth retrying."""


@dataclass(frozen=True)
class Grid:
    lam0: float
    dlam: float
    nbin: int

    @classmethod
    def from_config(cls, cfg: dict) -> Grid:
        s = cfg["spectra"]
        return cls(float(s["lam0_angstrom"]), float(s["dlam_angstrom"]), int(s["nbin"]))

    def wavelengths(self) -> np.ndarray:
        return self.lam0 + self.dlam * np.arange(self.nbin)


def coadd_url(template: str, survey: str, program: str, pix: int) -> str:
    return template.format(survey=survey, program=program, group=int(pix) // 100, pix=int(pix))


def shard_path(shard_dir: Path, survey: str, program: str, pix: int) -> Path:
    return shard_dir / f"{survey}__{program}__{int(pix)}.npz"


def _is_http(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def _open_coadd(url: str, timeout_s: float):
    if _is_http(url):
        import aiohttp

        # A timeout is mandatory: a dead connection otherwise blocks a worker forever,
        # no exception fires, and the job hangs holding stale sockets.
        kw = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=timeout_s,
                                                                 sock_read=timeout_s)}}
        return fits.open(url, use_fsspec=True, fsspec_kwargs=kw)
    return fits.open(url, memmap=True)


def _really_missing(url: str, timeout_s: float = 30.0) -> bool:
    """True only when the archive actually answers 404 (or a local path is absent)."""
    if not _is_http(url):
        return not Path(url).exists()
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 404
    except urllib.error.HTTPError as exc:
        return exc.code == 404
    except Exception:
        return False                                   # unreachable is not absent


# ----------------------------------------------------------------------------- one coadd

def fetch_once(url: str, want: np.ndarray, grid: Grid, timeout_s: float):
    """Read only our rows of one coadd and coadd the cameras onto the grid.

    Returns (targetids, flux, ivar) as float32 arrays of shape (n, nbin), or None
    when none of `want` is in the file.
    """
    with _open_coadd(url, timeout_s) as hdul:
        tid = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
        pos = {int(t): i for i, t in enumerate(tid)}
        rows = [(int(t), pos[int(t)]) for t in want if int(t) in pos]
        if not rows:
            return None
        idx = np.array([r for _, r in rows])
        order = np.argsort(idx)                        # ascending rows read faster
        idx = idx[order]
        keep = np.array([t for t, _ in rows], dtype=np.int64)[order]
        num = np.zeros((idx.size, grid.nbin), np.float64)
        den = np.zeros((idx.size, grid.nbin), np.float64)
        for cam in CAMERAS:
            wave = np.asarray(hdul[f"{cam}_WAVELENGTH"].data, dtype=np.float64)
            col = np.rint((wave - grid.lam0) / grid.dlam).astype(int)
            ok = (col >= 0) & (col < grid.nbin)
            col = col[ok]
            fsec, isec = hdul[f"{cam}_FLUX"].section, hdul[f"{cam}_IVAR"].section
            for k, r in enumerate(idx):
                f = np.asarray(fsec[r, :], dtype=np.float64)[ok]
                v = np.asarray(isec[r, :], dtype=np.float64)[ok]
                v = np.where(np.isfinite(v) & (v > 0) & np.isfinite(f), v, 0.0)
                num[k, col] += f * v
                den[k, col] += v
        flux = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return keep, flux.astype(np.float32), den.astype(np.float32)


def fetch_group(url: str, want: np.ndarray, grid: Grid, *, attempts: int = 4,
                backoff_s: float = 3.0, timeout_s: float = 120.0):
    """`fetch_once` with backoff. Raises MissingFile for a confirmed 404."""
    last: Exception | None = None
    for a in range(attempts):
        try:
            return fetch_once(url, want, grid, timeout_s)
        except FileNotFoundError as exc:
            # fsspec raises this for transient conditions too; confirm before believing it
            last = exc
            if _really_missing(url):
                raise MissingFile(str(exc)) from exc
        except Exception as exc:                       # 503, timeout, truncated read
            last = exc
        if a < attempts - 1:
            time.sleep(backoff_s * 2 ** a)
    assert last is not None
    raise last


def write_shard(path: Path, result, grid: Grid) -> int:
    """All-or-nothing: write under a temp name, then rename. Returns rows written."""
    # the temp name must end in .npz, or np.savez appends one and the rename fails
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    if result is None:
        np.savez(tmp, tid=np.empty(0, np.int64), flux=np.empty((0, grid.nbin), np.float32),
                 ivar=np.empty((0, grid.nbin), np.float32))
        n = 0
    else:
        np.savez(tmp, tid=result[0], flux=result[1], ivar=result[2])
        n = int(result[0].size)
    tmp.replace(path)
    return n


# ----------------------------------------------------------------------------- all groups

def plan_groups(frame: pd.DataFrame) -> list[tuple[str, str, int, np.ndarray]]:
    frame = frame.drop_duplicates("targetid")
    return [(str(s), str(p), int(h), g["targetid"].to_numpy(np.int64))
            for (s, p, h), g in frame.groupby(["survey", "program", "healpix"], sort=True)]


def fetch_all(groups, shard_dir: Path, template: str, grid: Grid, *, workers: int,
              timeout_s: float, backoff_s: float = 3.0, log=print) -> dict:
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in groups if not shard_path(shard_dir, g[0], g[1], g[2]).exists()]
    stats = {"groups": len(groups), "shards_present_before": len(groups) - len(todo),
             "groups_fetched": 0, "spectra_fetched": 0, "absent_coadds": 0,
             "transient_failures": 0, "absent": [], "failed": []}
    log(f"[spectra] {sum(g[3].size for g in groups):,} targets across {len(groups):,} coadd "
        f"files; {stats['shards_present_before']:,} shards present, {len(todo):,} to fetch")
    if not todo:
        return stats
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch_group, coadd_url(template, s, p, h), w, grid,
                               backoff_s=backoff_s, timeout_s=timeout_s): (s, p, h)
                   for s, p, h, w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                result = fut.result()
            except MissingFile:
                stats["absent_coadds"] += 1
                stats["absent"].append(list(key))
                write_shard(shard_path(shard_dir, *key), None, grid)  # never retried
                continue
            except Exception as exc:                   # transient: no shard, retried on rerun
                stats["transient_failures"] += 1
                stats["failed"].append([*key, f"{type(exc).__name__}: {str(exc)[:120]}"])
                log(f"[spectra]   retry later {key}: {type(exc).__name__}: {str(exc)[:90]}")
                continue
            try:
                n = write_shard(shard_path(shard_dir, *key), result, grid)
            except Exception as exc:
                stats["transient_failures"] += 1
                stats["failed"].append([*key, f"write: {exc}"])
                continue
            stats["groups_fetched"] += 1
            stats["spectra_fetched"] += n
            if i % 25 == 0 or i == len(todo):
                log(f"[spectra]   {i:,}/{len(todo):,} files | {stats['spectra_fetched']:,} "
                    f"spectra | {stats['transient_failures']} to retry | "
                    f"{stats['absent_coadds']} absent")
    return stats


# ----------------------------------------------------------------------------- merge

def merge(shard_dir: Path, out: Path, grid: Grid, log=print) -> dict:
    """Merge every shard into one HDF5, one shard in memory at a time.

    A target present in several shards keeps the copy with more positive-ivar
    pixels; ties go to the first shard in sorted order.
    """
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise SpectraError(f"no shards in {shard_dir}")
    best: dict[int, tuple[int, int, int]] = {}
    n_rows = 0
    for si, sp in enumerate(shards):
        with np.load(sp) as z:
            tid, ivar = z["tid"].astype(np.int64), z["ivar"]
        if ivar.ndim == 2 and ivar.shape[1] != grid.nbin:
            raise SpectraError(f"{sp.name}: {ivar.shape[1]} bins, expected {grid.nbin}")
        n_good = (ivar > 0).sum(axis=1) if tid.size else np.empty(0, int)
        for r, (t, g) in enumerate(zip(tid, n_good)):
            n_rows += 1
            cur = best.get(int(t))
            if cur is None or int(g) > cur[0]:
                best[int(t)] = (int(g), si, r)
    n_unique = len(best)
    log(f"[merge] {len(shards):,} shards, {n_rows:,} rows -> {n_unique:,} unique targets")
    if n_unique == 0:
        raise SpectraError("no spectra in any shard")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with h5py.File(tmp, "w") as h:
        d_tid = h.create_dataset("desi_targetid", shape=(n_unique,), dtype=np.int64)
        kw = dict(shape=(n_unique, grid.nbin), dtype=np.float32,
                  chunks=(min(64, n_unique), grid.nbin), compression="lzf")
        d_flux = h.create_dataset("spectra", **kw)
        d_ivar = h.create_dataset("spectra_ivar", **kw)
        pos = 0
        for si, sp in enumerate(shards):
            with np.load(sp) as z:
                tid, flux, ivar = z["tid"].astype(np.int64), z["flux"], z["ivar"]
            rows = np.array([r for r, t in enumerate(tid) if best[int(t)][1:] == (si, r)],
                            dtype=int)
            if rows.size == 0:
                continue
            d_tid[pos:pos + rows.size] = tid[rows]
            d_flux[pos:pos + rows.size] = flux[rows]
            d_ivar[pos:pos + rows.size] = ivar[rows]
            pos += rows.size
        assert pos == n_unique
        h.create_dataset("spectra_lambda", data=grid.wavelengths().astype(np.float32))
        h.attrs["n_spectra"] = n_unique
        h.attrs["n_shards"] = len(shards)
        h.attrs["lam0_angstrom"] = grid.lam0
        h.attrs["dlam_angstrom"] = grid.dlam
        h.attrs["nbin"] = grid.nbin
    tmp.replace(out)
    log(f"[merge] wrote {out} ({out.stat().st_size / 2 ** 20:.1f} MiB)")
    return {"shards": len(shards), "rows": n_rows, "unique_targets": n_unique,
            "duplicates_dropped": n_rows - n_unique}


# ----------------------------------------------------------------------------- run

def run(cfg: dict, *, limit_groups: int | None = None, workers: int | None = None,
        merge_only: bool = False, no_merge: bool = False, backoff_s: float = 3.0,
        log=print) -> dict:
    ensure_dirs(cfg)
    work = Path(cfg["paths"]["work"])
    xm_path = work / CROSSMATCH_OUTPUT
    if not xm_path.is_file():
        raise SpectraError(f"missing input {xm_path}; run crossmatch first")
    grid = Grid.from_config(cfg)
    template = cfg["archives"]["desi_coadd_url"]
    if not _is_http(template) and not Path(template).is_absolute():
        # a local coadd store (the test fixtures) resolves against the config file
        template = str(Path(cfg["_config_path"]).parent / template)
    shard_dir = work / SHARD_DIR
    source = work / SOURCE

    frame = pd.read_parquet(xm_path, columns=["targetid", "survey", "program", "healpix"])
    groups = plan_groups(frame)
    n_targets = int(frame["targetid"].nunique())
    if limit_groups:
        groups = groups[:limit_groups]
    stats: dict = {"targets": n_targets, "limit_groups": limit_groups}
    if not merge_only:
        stats["fetch"] = fetch_all(groups, shard_dir, template, grid,
                                   workers=workers or int(cfg["spectra"]["workers"]),
                                   timeout_s=float(cfg["spectra"]["timeout_s"]),
                                   backoff_s=backoff_s, log=log)
        if stats["fetch"]["transient_failures"]:
            raise SpectraError(f"{stats['fetch']['transient_failures']} coadd(s) failed "
                               f"transiently; rerun the same command to retry them")
    if not no_merge:
        stats["merge"] = merge(shard_dir, source, grid, log=log)
        with h5py.File(source, "r") as h:
            have = set(h["desi_targetid"][:].tolist())
        stats["targets_without_spectrum"] = int(
            (~frame.drop_duplicates("targetid")["targetid"].isin(have)).sum())
        write_ledger(STEP, cfg, inputs={"crossmatch": xm_path},
                     counts={"targets": n_targets, "coadd_groups": len(groups),
                             "spectra_merged": stats["merge"]["unique_targets"],
                             "targets_without_spectrum": stats["targets_without_spectrum"],
                             "absent_coadds": stats.get("fetch", {}).get("absent_coadds", 0)},
                     extra={"coadd_url_template": template,
                            "grid": {"lam0_angstrom": grid.lam0, "dlam_angstrom": grid.dlam,
                                     "nbin": grid.nbin},
                            "fetch": stats.get("fetch"), "merge": stats["merge"],
                            "limit_groups": limit_groups, "output": describe_file(source)})
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--limit-groups", type=int, default=None, help="fetch only N coadds")
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent coadds; the archive answers 503 when pushed")
    parser.add_argument("--no-merge", action="store_true", help="fetch shards only")
    parser.add_argument("--merge-only", action="store_true", help="merge existing shards only")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg, limit_groups=args.limit_groups, workers=args.workers,
            merge_only=args.merge_only, no_merge=args.no_merge)
    except SpectraError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
