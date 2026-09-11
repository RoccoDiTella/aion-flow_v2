"""Step 7: validate the staged files, the split and the label table before training.

    python -m aionflow_data.validate [--config CONFIG]

Every check is named and recorded in data/provenance/validate.json with its
verdict and a detail line; the exit code is non-zero if any check fails.
Checks: the three files exist; each carries exactly the contract datasets with
the contract dtypes and consistent shapes; no label dataset is staged; every
dataset is chunked along rows only; targetids are unique within and across
splits and equal split.csv; flags, redshift and source_row agree with the
manifest; the flags agree with the content (a zero frame iff has_image is
false, every staged row has a spectrum, has_z follows the redshift rule);
values are physical (finite spectra wherever ivar > 0, non-negative ivar,
finite images, positions on the sky); the split fractions are within the
configured tolerance; and labels.csv carries the trainer's columns for every
staged target, with its finite counts reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .common import ensure_dirs, load_config, step_parser, write_ledger
from .labels import OUTPUT as LABELS_OUTPUT
from .manifest_split import MANIFEST, PRESENCE_FLAGS, SPLIT, SPLITS
from .stage import IMAGE_BANDS, SPLIT_FILE

STEP = "validate"
CONTRACT = {
    "source_row": "int64", "desi_targetid": "int64", "spectra": "float32",
    "spectra_ivar": "float32", "spectra_lambda": "float32", "redshift": "float32",
    "flux_w1": "float32", "flux_w2": "float32", "flux_w3": "float32",
    "target_ra": "float32", "target_dec": "float32", "image_flux": "float32",
    "has_spectrum": "bool", "has_z": "bool", "has_wise": "bool", "has_image": "bool",
}
FORBIDDEN = ("log_ml_flux_1", "log_lx", "ml_flux_1", "log_sfr", "logmstar_cigale",
             "flux_sig_lo", "flux_sig_hi", "det_like_0", "ape_cts_1", "hr32_u", "logmstar")
TRAINER_COLUMNS = (
    ["targetid", "log_ml_flux_1", "flux_sig_lo", "flux_sig_hi", "log_lx", "det_like_0", "z",
     "spectype", "ero_detuid", "log_sfr", "log_sfr_sig_lo", "log_sfr_sig_hi",
     "logmstar_cigale", "logmstar_cigale_sig_lo", "logmstar_cigale_sig_hi"]
    + [f"log_flux_p{b}{s}" for b in (1, 2, 3, 4) for s in ("", "_sig_lo", "_sig_hi")]
    + [f"det_like_p{b}" for b in (1, 2, 3, 4)]
    + [f"ape_{q}_{b}" for b in ("1", "p1", "p2", "p3", "p4") for q in ("cts", "bkg", "exp")]
)
LABEL_COUNT_COLUMNS = ("log_ml_flux_1", "log_lx", "log_flux_p1", "log_flux_p2", "log_flux_p3",
                       "log_flux_p4", "log_sfr", "logmstar_cigale")


def _by_target(manifest: pd.DataFrame) -> pd.DataFrame:
    """The sample rows indexed by targetid (unique by construction; the full manifest is
    not, since a split-source pair shares one target)."""
    return manifest[manifest["in_sample"].astype(bool)].set_index("targetid")


class Report:
    def __init__(self, log=print) -> None:
        self.checks: list[dict] = []
        self.log = log

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "passed": bool(ok), "detail": detail})
        self.log(f"[validate] {'PASS' if ok else 'FAIL'}  {name:36s} {detail}")
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    @property
    def failed(self) -> list[str]:
        return [c["check"] for c in self.checks if not c["passed"]]


# ----------------------------------------------------------------------------- checks

def check_files(rep: Report, staged: Path) -> dict[str, h5py.File]:
    handles = {}
    missing, parts = [], sorted(p.name for p in staged.glob("*.part"))
    for split in SPLITS:
        path = staged / SPLIT_FILE.format(split=split)
        if path.is_file():
            handles[split] = h5py.File(path, "r")
        else:
            missing.append(path.name)
    rep.check("files_present", not missing and not parts,
              f"missing {missing}" if missing else f"leftover {parts}" if parts else
              ", ".join(SPLIT_FILE.format(split=s) for s in SPLITS))
    return handles


def check_schema(rep: Report, handles: dict, cfg: dict) -> None:
    nbin, size = int(cfg["spectra"]["nbin"]), int(cfg["cutouts"]["size"])
    problems = []
    for split, h in handles.items():
        keys = set(h.keys())
        if keys != set(CONTRACT):
            problems.append(f"{split}: datasets {sorted(keys ^ set(CONTRACT))}")
            continue
        n = h["desi_targetid"].shape[0]
        for key, dtype in CONTRACT.items():
            if str(h[key].dtype) != dtype:
                problems.append(f"{split}/{key}: dtype {h[key].dtype} != {dtype}")
        if h["spectra"].shape != (n, nbin) or h["spectra_ivar"].shape != (n, nbin):
            problems.append(f"{split}: spectra shape {h['spectra'].shape} != ({n}, {nbin})")
        if h["spectra_lambda"].shape != (nbin,):
            problems.append(f"{split}: spectra_lambda shape {h['spectra_lambda'].shape}")
        if h["image_flux"].shape != (n, len(IMAGE_BANDS), size, size):
            problems.append(f"{split}: image shape {h['image_flux'].shape}")
        for key in CONTRACT:
            if key != "spectra_lambda" and h[key].shape[0] != n:
                problems.append(f"{split}/{key}: {h[key].shape[0]} rows != {n}")
        bands = [b.decode() if isinstance(b, bytes) else str(b) for b in h.attrs["image_bands"]]
        if bands != list(IMAGE_BANDS) or int(h.attrs["image_size"]) != size:
            problems.append(f"{split}: image attrs {bands}, {h.attrs.get('image_size')}")
    rep.check("schema", not problems, "; ".join(problems[:4]) or
              f"{len(CONTRACT)} datasets, nbin {nbin}, image {size} px")
    forbidden = [f"{s}/{k}" for s, h in handles.items() for k in FORBIDDEN if k in h]
    rep.check("no_labels_staged", not forbidden, "; ".join(forbidden) or "inputs only")
    bad_chunks = []
    for split, h in handles.items():
        for key, ds in h.items():
            if key == "spectra_lambda" or ds.shape[0] == 0:
                continue
            if ds.chunks is None or tuple(ds.chunks[1:]) != tuple(ds.shape[1:]):
                bad_chunks.append(f"{split}/{key}: {ds.chunks}")
    rep.check("row_aligned_chunks", not bad_chunks, "; ".join(bad_chunks[:4]) or "every dataset")


def check_split(rep: Report, handles: dict, split_frame: pd.DataFrame, cfg: dict) -> None:
    staged = {s: h["desi_targetid"][:] for s, h in handles.items()}
    problems = []
    seen: set[int] = set()
    for split, tids in staged.items():
        if np.unique(tids).size != tids.size:
            problems.append(f"{split}: duplicate targetids")
        overlap = seen & set(tids.tolist())
        if overlap:
            problems.append(f"{split}: {len(overlap)} targetids also in another split")
        seen |= set(tids.tolist())
        expected = set(split_frame.loc[split_frame["split"] == split, "targetid"].astype(int))
        if expected != set(tids.tolist()):
            problems.append(f"{split}: {len(expected ^ set(tids.tolist()))} targetids differ "
                            f"from split.csv")
    rep.check("targetids_unique_and_match_split", not problems, "; ".join(problems[:4]) or
              f"{len(seen):,} targets staged once each")
    n = sum(t.size for t in staged.values())
    drift = {s: staged[s].size / max(n, 1) - f for s, f in zip(SPLITS, cfg["split"]["fractions"])}
    tol = float(cfg["split"]["tolerance"])
    rep.check("split_fractions", all(abs(d) <= tol for d in drift.values()),
              " ".join(f"{s} {staged[s].size:,} ({drift[s]:+.3f})" for s in SPLITS))


def check_manifest_agreement(rep: Report, handles: dict, manifest: pd.DataFrame) -> None:
    man = _by_target(manifest)
    problems = []
    for split, h in handles.items():
        tids = h["desi_targetid"][:]
        rows = man.reindex(tids)
        if rows["in_sample"].isna().any():
            problems.append(f"{split}: staged targetids missing from the manifest")
            continue
        if not (rows["split"] == split).all():
            problems.append(f"{split}: manifest split disagrees")
        if not np.array_equal(h["source_row"][:], rows["source_row"].to_numpy(np.int64)):
            problems.append(f"{split}: source_row disagrees")
        for flag in PRESENCE_FLAGS:
            if not np.array_equal(h[flag][:], rows[flag].to_numpy(bool)):
                problems.append(f"{split}: {flag} disagrees")
        if not np.array_equal(h["redshift"][:], rows["z"].to_numpy(np.float64).astype(np.float32)):
            problems.append(f"{split}: redshift disagrees")
    rep.check("manifest_agreement", not problems, "; ".join(problems[:4]) or
              "flags, redshift and source_row match")


def check_content(rep: Report, handles: dict, manifest: pd.DataFrame, cfg: dict) -> None:
    man = _by_target(manifest)
    lam0, dlam, nbin = (float(cfg["spectra"]["lam0_angstrom"]),
                        float(cfg["spectra"]["dlam_angstrom"]), int(cfg["spectra"]["nbin"]))
    grid = (lam0 + dlam * np.arange(nbin)).astype(np.float32)
    flag_problems, value_problems = [], []
    for split, h in handles.items():
        n = h["desi_targetid"].shape[0]
        has_image = h["has_image"][:]
        nonzero = np.zeros(n, bool)
        for lo in range(0, n, 256):
            block = h["image_flux"][lo:lo + 256]
            nonzero[lo:lo + block.shape[0]] = block.reshape(block.shape[0], -1).any(axis=1)
            if not np.isfinite(block).all():
                value_problems.append(f"{split}: non-finite image pixels")
        if not np.array_equal(nonzero, has_image):
            flag_problems.append(f"{split}: has_image disagrees with image content on "
                                 f"{int((nonzero != has_image).sum())} rows")
        if not h["has_spectrum"][:].all():
            flag_problems.append(f"{split}: has_spectrum false on a staged row")
        z = h["redshift"][:]
        zwarn = man.reindex(h["desi_targetid"][:])["zwarn"].to_numpy(np.float64)
        rule = np.isfinite(z) & (z > 0) & (zwarn == 0)
        if not np.array_equal(rule, h["has_z"][:]):
            flag_problems.append(f"{split}: has_z disagrees with the redshift rule")
        flux, ivar = h["spectra"][:], h["spectra_ivar"][:]
        if (ivar < 0).any() or not np.isfinite(ivar).all():
            value_problems.append(f"{split}: negative or non-finite ivar")
        if not np.isfinite(flux[ivar > 0]).all():
            value_problems.append(f"{split}: non-finite spectra where ivar > 0")
        if not np.array_equal(h["spectra_lambda"][:], grid):
            value_problems.append(f"{split}: spectra_lambda is not the configured grid")
        ra, dec = h["target_ra"][:], h["target_dec"][:]
        if not (np.isfinite(ra).all() and np.isfinite(dec).all() and (ra >= 0).all()
                and (ra < 360).all() and (np.abs(dec) <= 90).all()):
            value_problems.append(f"{split}: positions off the sky")
        if not (np.isfinite(z).all() and (np.abs(z) < 10).all()):
            value_problems.append(f"{split}: implausible redshift")
    rep.check("flags_vs_content", not flag_problems, "; ".join(flag_problems[:4]) or
              "zero frames iff has_image false; has_z follows the rule")
    rep.check("value_ranges", not value_problems, "; ".join(value_problems[:4]) or
              "finite spectra where ivar > 0, finite images, positions on the sky")


def check_sidecar(rep: Report, handles: dict, labels_path: Path, cfg: dict) -> dict:
    if not labels_path.is_file():
        rep.check("sidecar", False, f"{labels_path} missing")
        return {}
    header = pd.read_csv(labels_path, nrows=0).columns
    missing = [c for c in TRAINER_COLUMNS if c not in header]
    if missing:
        rep.check("sidecar", False, f"labels.csv lacks {missing[:6]}")
        return {}
    use = ["targetid", "z", "det_like_0"] + list(LABEL_COUNT_COLUMNS)
    labels = pd.read_csv(labels_path, usecols=use).drop_duplicates("targetid").set_index("targetid")
    staged = np.concatenate([h["desi_targetid"][:] for h in handles.values()])
    covered = labels.index.isin(staged)
    absent = int((~pd.Index(staged).isin(labels.index)).sum())
    counts = {"staged_targets": int(staged.size), "labelled_targets": int(covered.sum())}
    sub = labels[covered]
    for col in LABEL_COUNT_COLUMNS:
        counts[f"{col}_finite"] = int(np.isfinite(sub[col]).sum())
    det_min = float(cfg["labels"]["det_like_min"])
    counts[f"det_like_0_gt_{det_min:g}"] = int((sub["det_like_0"] > det_min).sum())
    rep.check("sidecar", absent == 0,
              f"{absent} staged targets without a label row" if absent else
              f"{len(TRAINER_COLUMNS)} trainer columns; log_ml_flux_1 finite "
              f"{counts['log_ml_flux_1_finite']:,}, log_sfr finite {counts['log_sfr_finite']:,}, "
              f"logmstar_cigale finite {counts['logmstar_cigale_finite']:,}")
    return counts


# ----------------------------------------------------------------------------- run

def run(cfg: dict, log=print) -> dict:
    ensure_dirs(cfg)
    work, staged = Path(cfg["paths"]["work"]), Path(cfg["paths"]["staged"])
    rep = Report(log=log)
    handles = check_files(rep, staged)
    manifest_path, split_path, labels_path = work / MANIFEST, work / SPLIT, work / LABELS_OUTPUT
    inputs_ok = manifest_path.is_file() and split_path.is_file()
    rep.check("manifest_and_split_present", inputs_ok, f"{manifest_path.name}, {split_path.name}")
    counts: dict = {}
    census: dict = {}
    if handles and len(handles) == len(SPLITS) and inputs_ok:
        manifest = pd.read_csv(manifest_path)
        split_frame = pd.read_csv(split_path)
        try:
            check_schema(rep, handles, cfg)
            if rep.passed:
                check_split(rep, handles, split_frame, cfg)
                check_manifest_agreement(rep, handles, manifest)
                check_content(rep, handles, manifest, cfg)
                counts = check_sidecar(rep, handles, labels_path, cfg)
                man = _by_target(manifest)
                for split, h in handles.items():
                    census[split] = (man.reindex(h["desi_targetid"][:])["spectype"]
                                     .value_counts().to_dict())
                    counts[f"rows_{split}"] = int(h["desi_targetid"].shape[0])
        finally:
            for h in handles.values():
                h.close()
    elif handles:
        for h in handles.values():
            h.close()
    verdict = {"passed": rep.passed, "failed": rep.failed, "checks": rep.checks,
               "census_by_split": census}
    log(f"[validate] {'ALL PASSED' if rep.passed else 'FAILED: ' + ', '.join(rep.failed)}")
    inputs = {name: p for name, p in (("manifest", manifest_path), ("split", split_path),
                                      ("labels", labels_path)) if p.is_file()}
    for split in SPLITS:
        p = staged / SPLIT_FILE.format(split=split)
        if p.is_file():
            inputs[f"staged_{split}"] = p
    write_ledger(STEP, cfg, inputs=inputs, counts=counts, extra=verdict)
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = step_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    verdict = run(cfg)
    if not verdict["passed"]:
        print(f"FAIL: {', '.join(verdict['failed'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
