"""Step 1: cross-match the eROSITA DR2 NWAY counterpart table with DESI DR1.

    python -m aionflow_data.crossmatch [--config CONFIG]

Rules, in order (PLAN.md section 3):
  NWAY   exact duplicate rows collapsed; NWAY_match_flag == 1 only; a DETUID that
         still has several primary rows keeps the highest NWAY_p_i, ties by
         NWAY_dist_post.
  DESI   zall-pix rows with ZCAT_PRIMARY, TARGETID > 0 and a finite position.
  match  every DESI target within `radius_arcsec` of the LS10 position; prefer a
         TARGETID whose release bits name a main-survey Legacy Survey release, then
         the nearest. No identity join: DESI DR1 indexes LS DR9, NWAY indexes DR10.
  cut    NWAY_p_any > NWAY_threshold6 where calibrated, else NWAY_p_any >= the
         configured floor.
  share  a DESI target adopted by several detections: X-ray positions all within
         `split_source_max_sep_arcsec` flag every row as a split source (kept here,
         excluded from the sample later); otherwise the highest NWAY_dist_post wins.

Output: <work>/crossmatch.parquet, one row per (detuid, targetid), and the ledger
data/provenance/crossmatch.json with every count.
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky

from .common import (
    FilterLedger,
    describe_file,
    ensure_dirs,
    load_config,
    read_fits_columns,
    step_parser,
    write_ledger,
)

STEP = "crossmatch"
OUTPUT = "crossmatch.parquet"

# FITS column -> output column. RA/DEC are the X-ray position; LS10_* the counterpart.
NWAY_COLUMNS = {
    "DETUID": "ero_detuid", "RA": "xray_ra", "DEC": "xray_dec",
    "LS10_RA": "ls10_ra", "LS10_DEC": "ls10_dec",
    "LS10_RELEASE": "ls10_release", "LS10_BRICKID": "ls10_brickid", "LS10_OBJID": "ls10_objid",
    "NWAY_p_any": "nway_p_any", "NWAY_p_i": "nway_p_i", "NWAY_p_single": "nway_p_single",
    "NWAY_match_flag": "nway_match_flag", "NWAY_threshold6": "nway_threshold6",
    "NWAY_dist_post": "nway_dist_post", "NWAY_dist_bayesfactor": "nway_dist_bayesfactor",
    "NWAY_Separation_LS10_ERO": "nway_sep_arcsec",
    "LS10_flux_w1": "ls10_flux_w1", "LS10_flux_w2": "ls10_flux_w2", "LS10_flux_w3": "ls10_flux_w3",
    "LS10_flux_ivar_w1": "ls10_flux_ivar_w1", "LS10_flux_ivar_w2": "ls10_flux_ivar_w2",
    "LS10_flux_ivar_w3": "ls10_flux_ivar_w3",
    "LS10_shape_r": "ls10_shape_r", "LS10_sersic": "ls10_sersic", "LS10_TYPE": "ls10_type",
    "LS10_Xray_proba": "ls10_xray_proba", "Exgal_prob_STAREX": "exgal_prob_starex",
    "class_gal_exgal": "class_gal_exgal", "simbad_known_galactic": "simbad_known_galactic",
}
DESI_COLUMNS = {
    "TARGET_RA": "target_ra", "TARGET_DEC": "target_dec",
    "MEAN_FIBER_RA": "fiber_ra", "MEAN_FIBER_DEC": "fiber_dec",
    "SURVEY": "survey", "PROGRAM": "program", "HEALPIX": "healpix", "SPECTYPE": "spectype",
    "Z": "z", "ZWARN": "zwarn", "DELTACHI2": "deltachi2",
}
MATCH_COLUMNS = ["sep_arcsec", "n_candidates", "preferred_over_nearest", "desi_release",
                 "is_main_survey", "reliability_branch", "split_source", "collision_group_size"]
OUTPUT_COLUMNS = (["targetid"] + list(NWAY_COLUMNS.values()) + list(DESI_COLUMNS.values())
                  + MATCH_COLUMNS)

# desitarget.targetmask.encode_targetid: objid bits 0-21, brickid 22-41, release 42-57
OBJID_BITS, BRICKID_BITS, RELEASE_BITS = 22, 20, 16


class CrossmatchError(RuntimeError):
    pass


def decode_release(targetid: np.ndarray) -> np.ndarray:
    """The Legacy Survey release encoded in a DESI TARGETID (0 for non-LS targets)."""
    tid = np.asarray(targetid, dtype=np.int64)
    return (tid >> (OBJID_BITS + BRICKID_BITS)) & ((1 << RELEASE_BITS) - 1)


def _strings(a: np.ndarray) -> np.ndarray:
    return np.char.strip(a.astype(str))


def decompress_if_needed(path: Path, work: Path, log=print) -> Path:
    """A gzipped FITS table cannot be memory-mapped; inflate it once into `work`."""
    if path.suffix != ".gz":
        return path
    out = work / path.name[:-3]
    if out.is_file() and out.stat().st_size > 0:
        log(f"[nway] using inflated {out}")
        return out
    log(f"[nway] inflating {path} -> {out}")
    tmp = out.with_name(out.name + ".part")
    with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 24)
    tmp.replace(out)
    return out


# ----------------------------------------------------------------------------- inputs

def load_nway(path: Path, led: FilterLedger | None = None,
              log=print) -> tuple[pd.DataFrame, FilterLedger]:
    cols = read_fits_columns(path, list(NWAY_COLUMNS))
    frame = pd.DataFrame({NWAY_COLUMNS[k]: (_strings(v) if v.dtype.kind in "SU" else v)
                          for k, v in cols.items()})
    del cols
    led = led or FilterLedger(len(frame))
    log(f"[nway] {len(frame):,} rows read from {path.name}")
    frame = frame[led.apply("nway_exact_duplicates_collapsed",
                            ~frame.duplicated(keep="first").to_numpy())]
    frame = frame[led.apply("nway_match_flag==1", frame["nway_match_flag"].to_numpy() == 1)]
    # one primary row per detection: highest p_i, then highest dist_post
    codes, _ = pd.factorize(frame["ero_detuid"].to_numpy())
    order = np.lexsort((-frame["nway_dist_post"].to_numpy(np.float64),
                        -frame["nway_p_i"].to_numpy(np.float64), codes))
    frame = frame.iloc[order]
    frame = frame[led.apply("nway_one_primary_per_detuid",
                            ~frame["ero_detuid"].duplicated(keep="first").to_numpy())]
    finite = np.isfinite(frame["ls10_ra"].to_numpy()) & np.isfinite(frame["ls10_dec"].to_numpy())
    frame = frame[led.apply("nway_finite_ls10_position", finite)]
    return frame.reset_index(drop=True), led


def load_desi(path: Path, main_releases: list[int], log=print) -> tuple[pd.DataFrame, dict]:
    head = read_fits_columns(path, ["TARGETID", "ZCAT_PRIMARY"])
    tid = head["TARGETID"].astype(np.int64)
    primary = head["ZCAT_PRIMARY"].astype(bool)
    del head
    keep = primary & (tid > 0)
    counts = {"desi_rows_raw": int(tid.size), "desi_primary_rows": int(primary.sum()),
              "desi_primary_targetid_le0": int((primary & ~(tid > 0)).sum())}
    cols = read_fits_columns(path, list(DESI_COLUMNS), rows=keep)
    frame = pd.DataFrame({DESI_COLUMNS[k]: (_strings(v) if v.dtype.kind in "SU" else v)
                          for k, v in cols.items()})
    del cols
    frame.insert(0, "targetid", tid[keep])
    finite = (np.isfinite(frame["target_ra"].to_numpy())
              & np.isfinite(frame["target_dec"].to_numpy()))
    counts["desi_primary_bad_position"] = int((~finite).sum())
    frame = frame[finite].reset_index(drop=True)
    counts["desi_rows_kept"] = int(len(frame))
    if frame["targetid"].duplicated().any():
        raise CrossmatchError("ZCAT_PRIMARY does not give one row per TARGETID; "
                              "the DESI catalogue is not the expected zall-pix product")
    frame["desi_release"] = decode_release(frame["targetid"].to_numpy())
    frame["is_main_survey"] = np.isin(frame["desi_release"].to_numpy(), main_releases)
    log(f"[desi] {counts['desi_rows_raw']:,} rows, {counts['desi_rows_kept']:,} primary targets "
        f"with a position; {int(frame['is_main_survey'].sum()):,} main-survey encoded")
    return frame, counts


# ----------------------------------------------------------------------------- matching

def match(nway: pd.DataFrame, desi: pd.DataFrame, radius_arcsec: float,
          log=print) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """One chosen DESI target per NWAY row that has any inside the radius.

    Returns the matched rows, the match statistics, and the boolean mask over `nway`
    of the rows that matched.
    """
    c_nway = SkyCoord(nway["ls10_ra"].to_numpy() * u.deg, nway["ls10_dec"].to_numpy() * u.deg)
    c_desi = SkyCoord(desi["target_ra"].to_numpy() * u.deg, desi["target_dec"].to_numpy() * u.deg)
    i_n, i_d, sep2d, _ = search_around_sky(c_nway, c_desi, radius_arcsec * u.arcsec)
    if i_n.size == 0:
        raise CrossmatchError(f"no DESI target within {radius_arcsec}\" of any NWAY position")
    sep = sep2d.arcsec
    is_main = desi["is_main_survey"].to_numpy()[i_d]
    n_cand = np.bincount(i_n, minlength=len(nway))

    def first_per_row(order):
        rows = i_n[order]
        first = np.ones(rows.size, bool)
        first[1:] = rows[1:] != rows[:-1]
        return order[first]

    pref = first_per_row(np.lexsort((sep, ~is_main, i_n)))
    near = first_per_row(np.lexsort((sep, i_n)))
    assert np.array_equal(i_n[pref], i_n[near])
    flipped = i_d[pref] != i_d[near]
    rows = nway.iloc[i_n[pref]].reset_index(drop=True)
    chosen = desi.iloc[i_d[pref]].reset_index(drop=True)
    out = pd.concat([chosen[["targetid"]], rows, chosen.drop(columns=["targetid"])], axis=1)
    out["sep_arcsec"] = sep[pref]
    out["n_candidates"] = n_cand[i_n[pref]]
    out["preferred_over_nearest"] = flipped
    stats = {"candidate_pairs": int(i_n.size),
             "nway_rows_matched": int(pref.size),
             "nway_rows_with_2plus_candidates": int((n_cand >= 2).sum()),
             "flipped_by_main_survey_preference": int(flipped.sum()),
             "median_sep_arcsec": float(np.median(sep[pref])),
             "p99_sep_arcsec": float(np.percentile(sep[pref], 99))}
    log(f"[match] {stats['nway_rows_matched']:,} NWAY rows matched ({stats['candidate_pairs']:,} "
        f"candidate pairs, {stats['nway_rows_with_2plus_candidates']:,} with 2+, "
        f"{stats['flipped_by_main_survey_preference']:,} flipped to a main-survey TARGETID)")
    matched = np.zeros(len(nway), bool)
    matched[i_n[pref]] = True
    return out, stats, matched


def reliability_mask(frame: pd.DataFrame,
                     uncalibrated_min: float) -> tuple[np.ndarray, np.ndarray, dict]:
    p_any = frame["nway_p_any"].to_numpy(np.float64)
    thr = frame["nway_threshold6"].to_numpy(np.float64)
    calibrated = np.isfinite(thr)
    keep = np.where(calibrated, p_any > thr, p_any >= uncalibrated_min)
    branch = np.where(calibrated, "calibrated", "uncalibrated")
    stats = {"calibrated_kept": int((calibrated & keep).sum()),
             "calibrated_dropped": int((calibrated & ~keep).sum()),
             "uncalibrated_kept": int((~calibrated & keep).sum()),
             "uncalibrated_dropped": int((~calibrated & ~keep).sum()),
             "uncalibrated_p_any_min": uncalibrated_min}
    return keep, branch, stats


def resolve_shared_targets(frame: pd.DataFrame, split_max_sep_arcsec: float,
                           log=print) -> tuple[np.ndarray, dict]:
    """Flag split-source groups and pick a winner in collision groups.

    Adds `split_source` and `collision_group_size`; returns the keep mask and stats.
    """
    n = len(frame)
    split = np.zeros(n, bool)
    size = np.ones(n, np.int64)
    keep = np.ones(n, bool)
    xra = frame["xray_ra"].to_numpy(np.float64)
    xdec = frame["xray_dec"].to_numpy(np.float64)
    dist_post = frame["nway_dist_post"].to_numpy(np.float64)
    p_any = frame["nway_p_any"].to_numpy(np.float64)
    sep_cp = frame["sep_arcsec"].to_numpy(np.float64)
    stats = {"shared_target_groups": 0, "split_source_groups": 0, "split_source_rows": 0,
             "collision_groups": 0, "collision_rows_dropped": 0, "largest_group": 1,
             "split_source_max_sep_arcsec": split_max_sep_arcsec}
    for rows in frame.groupby("targetid", sort=False).indices.values():
        if rows.size < 2:
            continue
        stats["shared_target_groups"] += 1
        stats["largest_group"] = max(stats["largest_group"], int(rows.size))
        size[rows] = rows.size
        coords = SkyCoord(xra[rows] * u.deg, xdec[rows] * u.deg)
        pair_max = max(coords[i].separation(coords[j]).arcsec
                       for i in range(rows.size) for j in range(i + 1, rows.size))
        if pair_max <= split_max_sep_arcsec:
            split[rows] = True
            stats["split_source_groups"] += 1
            stats["split_source_rows"] += int(rows.size)
        else:
            order = np.lexsort((sep_cp[rows], -p_any[rows], -dist_post[rows]))
            losers = rows[order[1:]]
            keep[losers] = False
            stats["collision_groups"] += 1
            stats["collision_rows_dropped"] += int(losers.size)
    frame["split_source"] = split
    frame["collision_group_size"] = size
    log(f"[shared] {stats['shared_target_groups']:,} targets under several detections: "
        f"{stats['split_source_groups']:,} split-source groups flagged, "
        f"{stats['collision_groups']:,} collisions resolved "
        f"({stats['collision_rows_dropped']:,} rows dropped)")
    return keep, stats


# ----------------------------------------------------------------------------- run

def run(cfg: dict, log=print) -> pd.DataFrame:
    ensure_dirs(cfg)
    raw, work = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["work"])
    cm = cfg["crossmatch"]
    nway_src = raw / cfg["inputs"]["nway"]["file"]
    desi_src = raw / cfg["inputs"]["desi_zcat"]["file"]
    for p in (nway_src, desi_src):
        if not p.is_file():
            raise CrossmatchError(f"missing input {p}; run fetch_catalogs first")
    nway_path = decompress_if_needed(nway_src, work, log)

    nway, led = load_nway(nway_path, log=log)
    desi, desi_counts = load_desi(desi_src, list(cm["main_survey_releases"]), log=log)

    frame, match_stats, matched = match(nway, desi, float(cm["radius_arcsec"]), log=log)
    del nway, desi
    led.apply("nway_matched_within_radius", matched)
    # `frame` is in NWAY row order restricted to matched rows, so the ledger masks below
    # index it directly

    keep, branch, rel_stats = reliability_mask(frame, float(cm["uncalibrated_p_any_min"]))
    frame["reliability_branch"] = branch
    frame = frame[led.apply("nway_p_any_reliability", keep)].reset_index(drop=True)

    keep, share_stats = resolve_shared_targets(frame, float(cm["split_source_max_sep_arcsec"]),
                                               log=log)
    frame = frame[led.apply("collision_losers_dropped", keep)].reset_index(drop=True)

    frame = frame[OUTPUT_COLUMNS].sort_values("ero_detuid").reset_index(drop=True)
    census = frame["spectype"].value_counts().to_dict()
    log(f"[out] {len(frame):,} rows; census {census}")

    out = work / OUTPUT
    frame.to_parquet(out, index=False)
    extra = {"radius_arcsec": cm["radius_arcsec"],
             "main_survey_releases": list(cm["main_survey_releases"]),
             "match": match_stats, "reliability": rel_stats, "shared_targets": share_stats,
             "census": {k: int(v) for k, v in census.items()},
             "output": describe_file(out)}
    if nway_path != nway_src:
        extra["nway_inflated"] = describe_file(nway_path)
    write_ledger(STEP, cfg,
                 inputs={"nway": nway_src, "desi_zcat": desi_src},
                 counts={"nway_rows_raw": led.n_in, **desi_counts, "rows_out": int(len(frame)),
                         "split_source_rows": share_stats["split_source_rows"]},
                 filters=led.rows, extra=extra)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg)
    except CrossmatchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
