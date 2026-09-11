#!/usr/bin/env python
"""Generate the committed test fixtures: tiny synthetic catalogues and archives.

Deterministic and seeded. Every situation the pipeline must handle is planted
once, and `planted.json` records what was planted and what each step must
produce from it, so the tests assert against construction rather than against
a previous run.

    python tests/fixtures/make_fixtures.py            # writes into tests/fixtures
    python tests/fixtures/make_fixtures.py --out DIR  # elsewhere (the tests use this)

Files written: nway.fits, main.fits, zall_pix.fits, cigale.fits,
coadd/coadd-<survey>-<program>-<pix>.fits, cutouts/<targetid>.fits,
planted.json, config.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED = 20260910

RA0, DEC0, SPREAD_DEG = 150.0, 20.0, 0.4
MAIN_RELEASE = 9010
LS10_RELEASE = 10000
GAIA_BIT = 1 << 61          # backup / Gaia-based TARGETIDs carry no LS release
C_KMS = 299792.458
SIGMA_V_KMS = 250.0
CUTOUT_SIZE = 32            # the fixture config sets cutouts.size to this; the paper uses 160

# rest-frame vacuum wavelengths of the planted lines, and doublet ratios
LINES = {
    "nev_3426": [(3426.85, 1.0)],
    "oii_3727": [(3727.09, 1.0), (3729.88, 1.4)],
    "hbeta_4862": [(4862.683, 1.0)],
    "oiii_4960": [(4960.295, 0.335)],       # the physical doublet ratio the fitter fixes
    "oiii_5008": [(5008.240, 1.0)],
    "halpha_6564": [(6564.610, 1.0)],
}
# DESI camera grids, 0.8 A, overlapping
CAMERAS = {"B": (3600.0, 5800.0), "R": (5760.0, 7620.0), "Z": (7520.0, 9824.0)}
EEF = {"1": 0.8836025, "P1": 0.89230239, "P2": 0.88694167, "P3": 0.8836025, "P4": 0.85624605}
BAND_FRACTION = {"1": 1.0, "P1": 0.10, "P2": 0.35, "P3": 0.45, "P4": 0.40}

NWAY_SPEC = [
    ("DETUID", "32A"), ("RA", "D"), ("DEC", "D"), ("LS10_RA", "D"), ("LS10_DEC", "D"),
    ("LS10_RELEASE", "I"), ("LS10_BRICKID", "J"), ("LS10_OBJID", "J"),
    ("NWAY_p_any", "E"), ("NWAY_p_i", "E"), ("NWAY_p_single", "E"), ("NWAY_match_flag", "I"),
    ("NWAY_threshold6", "D"), ("NWAY_dist_post", "E"), ("NWAY_dist_bayesfactor", "E"),
    ("NWAY_Separation_LS10_ERO", "E"),
    ("LS10_flux_w1", "E"), ("LS10_flux_w2", "E"), ("LS10_flux_w3", "E"),
    ("LS10_flux_ivar_w1", "E"), ("LS10_flux_ivar_w2", "E"), ("LS10_flux_ivar_w3", "E"),
    ("LS10_shape_r", "E"), ("LS10_sersic", "E"), ("LS10_TYPE", "3A"), ("LS10_Xray_proba", "D"),
    ("Exgal_prob_STAREX", "D"), ("class_gal_exgal", "J"), ("simbad_known_galactic", "L"),
    ("DET_LIKE_0", "E"), ("ML_FLUX_1", "E"),
]
ZALL_SPEC = [
    ("TARGETID", "K"), ("TARGET_RA", "D"), ("TARGET_DEC", "D"), ("ZCAT_PRIMARY", "L"),
    ("SURVEY", "7A"), ("PROGRAM", "6A"), ("HEALPIX", "J"), ("SPECTYPE", "6A"), ("Z", "D"),
    ("ZWARN", "K"), ("DELTACHI2", "D"), ("RELEASE", "I"), ("BRICKID", "J"), ("BRICK_OBJID", "J"),
]
CIGALE_SPEC = [
    ("TARGETID", "K"), ("SURVEY", "7A"), ("PROGRAM", "7A"), ("HEALPIX", "J"), ("SPECTYPE", "7A"),
    ("RA", "D"), ("DEC", "D"), ("RELEASE", "I"), ("Z", "D"), ("CHI2", "D"),
    ("LOGM", "D"), ("LOGM_ERR", "D"), ("LOGSFR", "D"), ("LOGSFR_ERR", "D"),
    ("AGNLUM", "D"), ("AGNFRAC", "D"), ("AGNPSY", "D"), ("FLAG_MASSPDF", "D"), ("FLAG_SFRPDF", "D"),
]


def main_spec() -> list[tuple[str, str]]:
    spec = [("DETUID", "32A"), ("RA", "D"), ("DEC", "D"), ("DET_LIKE_0", "E")]
    spec += [(f"DET_LIKE_P{b}", "E") for b in (1, 2, 3, 4)]
    for b in ("1", "P1", "P2", "P3", "P4"):
        spec += [(f"ML_FLUX_{b}", "E"), (f"ML_FLUX_LOWERR_{b}", "E"), (f"ML_FLUX_UPERR_{b}", "E"),
                 (f"APE_CTS_{b}", "I"), (f"APE_BKG_{b}", "E"), (f"APE_EXP_{b}", "E"),
                 (f"APE_RADIUS_{b}", "E"), (f"APE_POIS_{b}", "E"),
                 (f"ML_CTS_{b}", "E"), (f"ML_RATE_{b}", "E"), (f"ML_EXP_{b}", "E"),
                 (f"ML_EEF_{b}", "E")]
    return spec


# ----------------------------------------------------------------------------- helpers

def encode_targetid(release: int, brickid: int, objid: int) -> int:
    return (release << 42) | (brickid << 22) | objid


def backup_targetid(gaia_low: int) -> int:
    return GAIA_BIT | gaia_low


def offset(ra: float, dec: float, dra_arcsec: float, ddec_arcsec: float) -> tuple[float, float]:
    return (ra + dra_arcsec / 3600.0 / math.cos(math.radians(dec)), dec + ddec_arcsec / 3600.0)


def write_table(path: Path, rows: list[dict], spec: list[tuple[str, str]]) -> None:
    cols = []
    for name, fmt in spec:
        values = [r.get(name, _default(fmt)) for r in rows]
        if fmt.endswith("A"):
            arr = np.array(values, dtype=f"U{fmt[:-1]}")
        elif fmt == "L":
            arr = np.array(values, dtype=bool)
        elif fmt in ("I", "J", "K"):
            arr = np.array(values, dtype={"I": np.int16, "J": np.int32, "K": np.int64}[fmt])
        else:
            arr = np.array(values, dtype={"E": np.float32, "D": np.float64}[fmt])
        cols.append(fits.Column(name=name, format=fmt, array=arr))
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.name = "CATALOG"
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _default(fmt: str):
    if fmt.endswith("A"):
        return ""
    if fmt == "L":
        return False
    if fmt in ("I", "J", "K"):
        return 0
    return np.nan


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- builder

class Builder:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.rng = np.random.default_rng(SEED)
        self.nway: list[dict] = []
        self.desi: list[dict] = []
        self.main: list[dict] = []
        self.cigale: list[dict] = []
        self.spectra: dict[tuple[str, str, int], list[dict]] = {}
        self.cutouts: list[tuple[int, float, float]] = []
        self.planted: dict = {"seed": SEED, "sigma_v_kms": SIGMA_V_KMS, "scenarios": {},
                              "line_flux": {}, "coadd_groups": {}, "cutout_targetids": []}
        self.expected_rows: list[dict] = []      # crossmatch output, by construction
        self._n_det = 0
        self._n_obj = 0
        self._n_gaia = 0

    # -- primitives ---------------------------------------------------------
    def sky(self) -> tuple[float, float]:
        return (RA0 + self.rng.uniform(-SPREAD_DEG, SPREAD_DEG),
                DEC0 + self.rng.uniform(-SPREAD_DEG, SPREAD_DEG))

    def detuid(self) -> str:
        self._n_det += 1
        return f"sm03_150020_020_ML{self._n_det:05d}_001_c030"

    def ls10_object(self) -> tuple[int, int]:
        self._n_obj += 1
        return 300000 + self._n_obj // 7, 1000 + self._n_obj

    def main_targetid(self, brickid: int, objid: int) -> int:
        return encode_targetid(MAIN_RELEASE, brickid, objid)

    def gaia_targetid(self) -> int:
        self._n_gaia += 1
        return backup_targetid(4_294_000_000 + self._n_gaia)

    def add_desi(self, targetid: int, ra: float, dec: float, spectype: str, z: float, *,
                 survey="main", program="dark", healpix=1234, zwarn=0, primary=True,
                 brickid=0, objid=0, release=MAIN_RELEASE) -> dict:
        row = dict(TARGETID=targetid, TARGET_RA=ra, TARGET_DEC=dec, ZCAT_PRIMARY=primary,
                   SURVEY=survey, PROGRAM=program, HEALPIX=healpix, SPECTYPE=spectype, Z=z,
                   ZWARN=zwarn, DELTACHI2=float(self.rng.uniform(20, 500)),
                   RELEASE=release if brickid else (0 if targetid & GAIA_BIT else -1),
                   BRICKID=brickid, BRICK_OBJID=objid)
        self.desi.append(row)
        return row

    def add_nway(self, detuid: str, xra: float, xdec: float, lra: float, ldec: float,
                 brickid: int, objid: int, *, p_any=None, threshold6=None, match_flag=1,
                 dist_post=None, p_i=None, flux_w3=None, det_like_0=None) -> dict:
        sep = math.hypot((lra - xra) * math.cos(math.radians(xdec)), ldec - xdec) * 3600.0
        w1 = float(self.rng.uniform(20, 400))
        u = self.rng.uniform
        row = dict(
            DETUID=detuid, RA=xra, DEC=xdec, LS10_RA=lra, LS10_DEC=ldec,
            LS10_RELEASE=LS10_RELEASE, LS10_BRICKID=brickid, LS10_OBJID=objid,
            NWAY_p_any=float(u(0.9, 1.0)) if p_any is None else p_any,
            NWAY_p_i=float(u(0.85, 1.0)) if p_i is None else p_i,
            NWAY_p_single=float(u(0.5, 1.0)),
            NWAY_match_flag=match_flag,
            NWAY_threshold6=float(u(0.03, 0.06)) if threshold6 is None else threshold6,
            NWAY_dist_post=float(u(0.9, 1.0)) if dist_post is None else dist_post,
            NWAY_dist_bayesfactor=float(u(1, 6)),
            NWAY_Separation_LS10_ERO=sep,
            LS10_flux_w1=w1, LS10_flux_w2=w1 * 0.8,
            LS10_flux_w3=w1 * 0.5 if flux_w3 is None else flux_w3,
            LS10_flux_ivar_w1=1.0, LS10_flux_ivar_w2=0.8, LS10_flux_ivar_w3=0.01,
            LS10_shape_r=float(u(0.2, 3.0)), LS10_sersic=float(u(0.5, 4)),
            LS10_TYPE=str(self.rng.choice(["PSF", "REX", "EXP", "DEV", "SER"])),
            LS10_Xray_proba=float(self.rng.uniform(0.2, 1.0)),
            Exgal_prob_STAREX=float(self.rng.uniform(0.5, 1.0)), class_gal_exgal=1,
            simbad_known_galactic=False,
            DET_LIKE_0=float(self.rng.uniform(6.5, 60)) if det_like_0 is None else det_like_0,
            ML_FLUX_1=float(10 ** self.rng.uniform(-13.8, -12.2)),
        )
        self.nway.append(row)
        return row

    def add_main(self, detuid: str, xra: float, xdec: float, flux1: float,
                 det_like_0: float, **overrides) -> dict:
        row = dict(DETUID=detuid, RA=xra, DEC=xdec, DET_LIKE_0=det_like_0)
        for i, b in enumerate(("1", "P1", "P2", "P3", "P4")):
            f = flux1 * BAND_FRACTION[b]
            exp = float(self.rng.uniform(150, 600))
            rate = f / 1.2e-12
            bkg = float(self.rng.uniform(2, 6))
            cts = int(round(rate * exp * EEF[b] + bkg))
            row[f"ML_FLUX_{b}"] = f
            row[f"ML_FLUX_LOWERR_{b}"] = f * float(self.rng.uniform(0.08, 0.3))
            row[f"ML_FLUX_UPERR_{b}"] = f * float(self.rng.uniform(0.1, 0.35))
            row[f"APE_CTS_{b}"] = cts
            row[f"APE_BKG_{b}"] = bkg
            row[f"APE_EXP_{b}"] = exp
            row[f"APE_RADIUS_{b}"] = 8.5
            row[f"APE_POIS_{b}"] = float(self.rng.uniform(1e-6, 1e-2)) if cts > 0 else -9.99
            row[f"ML_CTS_{b}"] = rate * exp
            row[f"ML_RATE_{b}"] = rate
            row[f"ML_EXP_{b}"] = exp
            row[f"ML_EEF_{b}"] = EEF[b]
            if b != "1":
                row[f"DET_LIKE_{b}"] = float(self.rng.uniform(0, 40))
        row.update(overrides)
        self.main.append(row)
        return row

    def add_cigale(self, targetid: int, desi: dict, **overrides) -> dict:
        u = self.rng.uniform
        row = dict(TARGETID=targetid, SURVEY=desi["SURVEY"], PROGRAM=desi["PROGRAM"],
                   HEALPIX=desi["HEALPIX"], SPECTYPE=desi["SPECTYPE"], RA=desi["TARGET_RA"],
                   DEC=desi["TARGET_DEC"], RELEASE=MAIN_RELEASE, Z=desi["Z"],
                   CHI2=float(u(0.5, 3.0)),
                   LOGM=float(u(9.5, 11.5)), LOGM_ERR=float(u(0.05, 0.3)),
                   LOGSFR=float(u(-1.0, 2.0)), LOGSFR_ERR=float(u(0.05, 0.4)),
                   AGNLUM=float(10 ** u(36, 39)), AGNFRAC=float(u(0, 0.6)),
                   AGNPSY=float(self.rng.choice([30.0, 70.0])),
                   FLAG_MASSPDF=float(u(0.6, 1.6)),
                   FLAG_SFRPDF=float(u(0.6, 1.6)))
        row.update(overrides)
        self.cigale.append(row)
        return row

    def add_spectrum(self, desi: dict, line_scale: float | None) -> None:
        key = (desi["SURVEY"], desi["PROGRAM"], int(desi["HEALPIX"]))
        self.spectra.setdefault(key, []).append(dict(desi=desi, line_scale=line_scale))

    def expect(self, nway_row: dict, targetid: int, spectype: str, *, split_source=False,
               has_spectrum=True) -> None:
        self.expected_rows.append(dict(detuid=nway_row["DETUID"], targetid=int(targetid),
                                       spectype=spectype, split_source=split_source,
                                       has_spectrum=has_spectrum))

    # -- one ordinary source ------------------------------------------------
    def clean_source(self, spectype: str, *, program="dark", healpix=1234, z=None, zwarn=0,
                     with_spectrum=True, with_cutout=True, flux_w3=None, line_scale=None,
                     main_overrides=None) -> dict:
        xra, xdec = self.sky()
        u = self.rng.uniform
        lra, ldec = offset(xra, xdec, u(-6, 6), u(-6, 6))
        tra, tdec = offset(lra, ldec, u(-0.03, 0.03), u(-0.03, 0.03))
        brickid, objid = self.ls10_object()
        tid = self.main_targetid(brickid, objid)
        if z is None:
            z = {"QSO": self.rng.uniform(0.3, 2.5), "GALAXY": self.rng.uniform(0.05, 0.8),
                 "STAR": 0.0002}[spectype]
        det = self.detuid()
        nrow = self.add_nway(det, xra, xdec, lra, ldec, brickid, objid, flux_w3=flux_w3)
        drow = self.add_desi(tid, tra, tdec, spectype, float(z), program=program,
                             healpix=healpix, zwarn=zwarn, brickid=brickid, objid=objid)
        self.add_main(det, xra, xdec, nrow["ML_FLUX_1"], nrow["DET_LIKE_0"],
                      **(main_overrides or {}))
        if with_spectrum:
            self.add_spectrum(drow, line_scale)
        if with_cutout:
            self.cutouts.append((tid, tra, tdec))
        self.expect(nrow, tid, spectype, has_spectrum=with_spectrum)
        return dict(nway=nrow, desi=drow, targetid=tid, detuid=det)

    # -- scenarios ----------------------------------------------------------
    def build(self) -> dict:
        S = self.planted["scenarios"]
        clean = []
        spectypes = ["QSO"] * 22 + ["GALAXY"] * 6 + ["STAR"] * 2
        for i, st in enumerate(spectypes):
            kw: dict = {}
            if i == 1:
                kw["zwarn"] = 4
            if i == 2:
                kw["flux_w3"] = -5.0
            if i == 3:
                kw["with_cutout"] = False
            if i == 4:
                kw.update(program="backup", healpix=9999, with_spectrum=False)
            if i == 6:
                kw["main_overrides"] = {"APE_CTS_1": -30000}
            if i == 7:
                kw["main_overrides"] = {"APE_BKG_P2": -0.5}
            if i in (9, 10, 11):
                kw["main_overrides"] = {"APE_CTS_P4": 0, "DET_LIKE_P4": 0.0, "APE_POIS_P4": -9.99}
            if i == 29:
                kw["z"] = -0.0015
            if i in (22, 23, 24):
                kw.update(program="bright", healpix=1235)
            if i in (12, 13, 14, 15, 16, 17, 18, 20, 21):
                kw["line_scale"] = float(self.rng.uniform(15, 50))
            clean.append(self.clean_source(st, **kw))
        # a flux consistent with zero in P4 on clean[8]: LOWERR >= FLUX
        m8 = next(m for m in self.main if m["DETUID"] == clean[8]["detuid"])
        m8["ML_FLUX_LOWERR_P4"] = m8["ML_FLUX_P4"] * 1.2
        S["clean"] = {"detuids": [c["detuid"] for c in clean],
                      "targetids": [c["targetid"] for c in clean]}
        S["zwarn_nonzero"] = {"targetid": clean[1]["targetid"], "zwarn": 4}
        S["w3_nonpositive"] = {"targetid": clean[2]["targetid"], "flux_w3": -5.0}
        S["no_cutout"] = {"targetid": clean[3]["targetid"]}
        S["no_spectrum"] = {"targetid": clean[4]["targetid"], "group": ["main", "backup", 9999]}
        S["ape_cts_wrapped"] = {"detuid": clean[6]["detuid"], "band": "1", "value": -30000}
        S["ape_bkg_negative"] = {"detuid": clean[7]["detuid"], "band": "P2", "value": -0.5}
        S["flux_consistent_with_zero"] = {"detuid": clean[8]["detuid"], "band": "P4"}
        S["zero_counts_p4"] = {"detuids": [clean[i]["detuid"] for i in (9, 10, 11)]}
        S["z_nonpositive"] = {"targetid": clean[29]["targetid"], "z": -0.0015}

        # a non-primary duplicate observation of clean[5] (sv3), dropped by ZCAT_PRIMARY
        d5 = clean[5]["desi"]
        self.add_desi(d5["TARGETID"], d5["TARGET_RA"], d5["TARGET_DEC"], d5["SPECTYPE"], d5["Z"],
                      survey="sv3", program="dark", healpix=d5["HEALPIX"], primary=False,
                      brickid=d5["BRICKID"], objid=d5["BRICK_OBJID"])
        S["nonprimary_duplicate"] = {"targetid": clean[5]["targetid"], "rows_in_zall_pix": 2}

        # tie: two main-survey targets inside 1", nearest wins, no flip
        xra, xdec = self.sky()
        lra, ldec = offset(xra, xdec, 3.0, -2.0)
        b, o = self.ls10_object()
        near = self.main_targetid(b, o)
        b2, o2 = self.ls10_object()
        far = self.main_targetid(b2, o2)
        det = self.detuid()
        n = self.add_nway(det, xra, xdec, lra, ldec, b, o)
        dn = self.add_desi(near, *offset(lra, ldec, 0.3, 0.0), "QSO", 1.1, brickid=b, objid=o)
        self.add_desi(far, *offset(lra, ldec, 0.0, 0.6), "QSO", 1.3, brickid=b2, objid=o2)
        self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
        self.add_spectrum(dn, None)
        self.cutouts.append((near, dn["TARGET_RA"], dn["TARGET_DEC"]))
        self.expect(n, near, "QSO")
        S["tie_two_main"] = {"detuid": det, "chosen_targetid": near, "other_targetid": far}

        # tie: backup target nearer than the main-survey target, main wins (a flip)
        xra, xdec = self.sky()
        lra, ldec = offset(xra, xdec, -4.0, 1.0)
        b, o = self.ls10_object()
        main_tid = self.main_targetid(b, o)
        gaia_tid = self.gaia_targetid()
        det = self.detuid()
        n = self.add_nway(det, xra, xdec, lra, ldec, b, o)
        dm = self.add_desi(main_tid, *offset(lra, ldec, 0.5, 0.5), "GALAXY", 0.21,
                           brickid=b, objid=o, program="bright", healpix=1235)
        self.add_desi(gaia_tid, *offset(lra, ldec, 0.1, 0.1), "GALAXY", 0.21, program="backup")
        self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
        self.add_spectrum(dm, float(self.rng.uniform(15, 50)))
        self.cutouts.append((main_tid, dm["TARGET_RA"], dm["TARGET_DEC"]))
        self.expect(n, main_tid, "GALAXY")
        S["tie_main_vs_backup"] = {"detuid": det, "chosen_targetid": main_tid,
                                   "nearest_targetid": gaia_tid}

        # only a backup-encoded target inside 1": it is the match
        xra, xdec = self.sky()
        lra, ldec = offset(xra, xdec, 2.0, 2.0)
        b, o = self.ls10_object()
        gaia_tid = self.gaia_targetid()
        det = self.detuid()
        n = self.add_nway(det, xra, xdec, lra, ldec, b, o)
        dg = self.add_desi(gaia_tid, *offset(lra, ldec, 0.02, -0.02), "QSO", 0.9,
                           program="backup")
        self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
        self.add_spectrum(dg, None)
        self.cutouts.append((gaia_tid, dg["TARGET_RA"], dg["TARGET_DEC"]))
        self.expect(n, gaia_tid, "QSO")
        S["backup_only"] = {"detuid": det, "targetid": gaia_tid}

        # a secondary candidate row (match_flag 2) for clean[0]'s detection, with its own
        # DESI target inside 1" that must NOT be adopted
        c0 = clean[0]["nway"]
        lra, ldec = offset(c0["LS10_RA"], c0["LS10_DEC"], 5.0, 5.0)
        b, o = self.ls10_object()
        self.add_nway(c0["DETUID"], c0["RA"], c0["DEC"], lra, ldec, b, o, match_flag=2,
                      p_i=0.3, p_any=c0["NWAY_p_any"], threshold6=c0["NWAY_threshold6"])
        sec_tid = self.main_targetid(b, o)
        self.add_desi(sec_tid, *offset(lra, ldec, 0.01, 0.01), "GALAXY", 0.4, brickid=b, objid=o)
        S["secondary_flag2"] = {"detuid": c0["DETUID"], "rejected_targetid": sec_tid}

        # an exact duplicate row of clean[20]
        self.nway.append(dict(clean[20]["nway"]))
        S["exact_duplicate"] = {"detuid": clean[20]["detuid"]}

        # one DETUID with two primary rows naming different counterparts: higher p_i wins
        xra, xdec = self.sky()
        det = self.detuid()
        la, lda = offset(xra, xdec, 3.0, 3.0)
        lb, ldb = offset(xra, xdec, -3.0, -3.0)
        ba, oa = self.ls10_object()
        bb, ob = self.ls10_object()
        ta, tb = self.main_targetid(ba, oa), self.main_targetid(bb, ob)
        na = self.add_nway(det, xra, xdec, la, lda, ba, oa, p_i=0.8, dist_post=0.7)
        self.add_nway(det, xra, xdec, lb, ldb, bb, ob, p_i=0.2, dist_post=0.9,
                      p_any=na["NWAY_p_any"], threshold6=na["NWAY_threshold6"])
        da = self.add_desi(ta, *offset(la, lda, 0.01, 0.0), "QSO", 1.7, brickid=ba, objid=oa)
        self.add_desi(tb, *offset(lb, ldb, 0.01, 0.0), "QSO", 0.6, brickid=bb, objid=ob)
        self.add_main(det, xra, xdec, na["ML_FLUX_1"], na["DET_LIKE_0"])
        self.add_spectrum(da, None)
        self.cutouts.append((ta, da["TARGET_RA"], da["TARGET_DEC"]))
        self.expect(na, ta, "QSO")
        S["repeat_detuid_two_primaries"] = {"detuid": det, "kept_targetid": ta,
                                            "dropped_targetid": tb}

        # uncalibrated threshold6: kept at p_any 0.06, dropped at 0.04
        for name, p_any, keep in (("uncalibrated_keep", 0.06, True),
                                  ("uncalibrated_drop", 0.04, False)):
            xra, xdec = self.sky()
            lra, ldec = offset(xra, xdec, 1.0, -1.0)
            b, o = self.ls10_object()
            tid = self.main_targetid(b, o)
            det = self.detuid()
            n = self.add_nway(det, xra, xdec, lra, ldec, b, o, p_any=p_any, threshold6=np.nan)
            d = self.add_desi(tid, *offset(lra, ldec, 0.01, 0.01), "GALAXY", 0.33, brickid=b,
                              objid=o, program="bright", healpix=1235)
            self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
            if keep:
                self.add_spectrum(d, None)
                self.cutouts.append((tid, d["TARGET_RA"], d["TARGET_DEC"]))
                self.expect(n, tid, "GALAXY")
            S[name] = {"detuid": det, "targetid": tid, "p_any": p_any}

        # calibrated row below its own threshold
        xra, xdec = self.sky()
        lra, ldec = offset(xra, xdec, 1.0, 1.0)
        b, o = self.ls10_object()
        tid = self.main_targetid(b, o)
        det = self.detuid()
        n = self.add_nway(det, xra, xdec, lra, ldec, b, o, p_any=0.03, threshold6=0.045)
        self.add_desi(tid, *offset(lra, ldec, 0.01, 0.01), "QSO", 2.0, brickid=b, objid=o)
        self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
        S["below_threshold6"] = {"detuid": det, "targetid": tid, "p_any": 0.03,
                                 "threshold6": 0.045}

        # collision: two detections 45" apart share one counterpart; higher dist_post wins
        xra, xdec = self.sky()
        xra2, xdec2 = offset(xra, xdec, 45.0, 0.0)
        lra, ldec = offset(xra, xdec, 20.0, 3.0)
        b, o = self.ls10_object()
        tid = self.main_targetid(b, o)
        d1, d2 = self.detuid(), self.detuid()
        n1 = self.add_nway(d1, xra, xdec, lra, ldec, b, o, dist_post=0.95, p_any=0.99)
        n2 = self.add_nway(d2, xra2, xdec2, lra, ldec, b, o, dist_post=0.80, p_any=0.7)
        dd = self.add_desi(tid, *offset(lra, ldec, 0.01, 0.01), "QSO", 1.4, brickid=b, objid=o)
        self.add_main(d1, xra, xdec, n1["ML_FLUX_1"], n1["DET_LIKE_0"])
        self.add_main(d2, xra2, xdec2, n2["ML_FLUX_1"], n2["DET_LIKE_0"])
        self.add_spectrum(dd, None)
        self.cutouts.append((tid, dd["TARGET_RA"], dd["TARGET_DEC"]))
        self.expect(n1, tid, "QSO")
        S["collision"] = {"kept_detuid": d1, "dropped_detuid": d2, "targetid": tid,
                          "xray_sep_arcsec": 45.0}

        # split source: two detections 6.3" apart share one counterpart; both flagged,
        # both excluded from the sample
        xra, xdec = self.sky()
        xra2, xdec2 = offset(xra, xdec, 6.3, 0.0)
        lra, ldec = offset(xra, xdec, 3.0, 1.0)
        b, o = self.ls10_object()
        tid = self.main_targetid(b, o)
        d1, d2 = self.detuid(), self.detuid()
        n1 = self.add_nway(d1, xra, xdec, lra, ldec, b, o, dist_post=0.9)
        n2 = self.add_nway(d2, xra2, xdec2, lra, ldec, b, o, dist_post=0.85)
        ds = self.add_desi(tid, *offset(lra, ldec, 0.01, 0.01), "GALAXY", 0.12, brickid=b, objid=o,
                           program="bright", healpix=1235)
        self.add_main(d1, xra, xdec, n1["ML_FLUX_1"], n1["DET_LIKE_0"])
        self.add_main(d2, xra2, xdec2, n2["ML_FLUX_1"], n2["DET_LIKE_0"])
        self.add_spectrum(ds, None)
        self.cutouts.append((tid, ds["TARGET_RA"], ds["TARGET_DEC"]))
        self.expect(n1, tid, "GALAXY", split_source=True)
        self.expect(n2, tid, "GALAXY", split_source=True)
        S["split_source"] = {"detuids": [d1, d2], "targetid": tid, "xray_sep_arcsec": 6.3}

        # a sky fibre (TARGETID -1) is the only thing inside 1": no match
        xra, xdec = self.sky()
        lra, ldec = offset(xra, xdec, 2.0, -2.0)
        b, o = self.ls10_object()
        det = self.detuid()
        n = self.add_nway(det, xra, xdec, lra, ldec, b, o)
        self.add_desi(-1, *offset(lra, ldec, 0.01, 0.01), "", 0.0)
        self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
        S["sky_fibre_near_detection"] = {"detuid": det}

        # detections with nothing inside 1"
        far = []
        for _ in range(5):
            xra, xdec = self.sky()
            lra, ldec = offset(xra, xdec, 2.0, 2.0)
            b, o = self.ls10_object()
            det = self.detuid()
            n = self.add_nway(det, xra, xdec, lra, ldec, b, o)
            self.add_main(det, xra, xdec, n["ML_FLUX_1"], n["DET_LIKE_0"])
            far.append(det)
        S["no_desi_within_radius"] = {"detuids": far}

        # background: DESI targets and Main detections unrelated to any NWAY row
        for _ in range(10):
            b, o = self.ls10_object()
            self.add_desi(self.main_targetid(b, o), *self.sky(), "GALAXY",
                          float(self.rng.uniform(0.1, 1.0)), brickid=b, objid=o)
        for _ in range(10):
            xra, xdec = self.sky()
            self.add_main(self.detuid(), xra, xdec, 10 ** self.rng.uniform(-14, -12.5),
                          float(self.rng.uniform(6.5, 30)))

        # ---- CIGALE ------------------------------------------------------
        cig_targets = {r["targetid"]: r["spectype"] for r in self.expected_rows}
        desi_by_tid = {d["TARGETID"]: d for d in self.desi if d["ZCAT_PRIMARY"]}
        special = {
            clean[12]["targetid"]: ("cigale_failed", {"LOGM": 0.0, "LOGSFR": 0.0}),
            clean[13]["targetid"]: ("cigale_sentinel", {"LOGM": -99.0, "LOGM_ERR": -99.0}),
            clean[14]["targetid"]: ("cigale_broad_mass", {"FLAG_MASSPDF": 8.0, "FLAG_SFRPDF": 1.2}),
            clean[15]["targetid"]: ("cigale_broad_sfr", {"FLAG_MASSPDF": 1.1, "FLAG_SFRPDF": 0.1}),
            clean[16]["targetid"]: ("cigale_wide_err", {"LOGSFR_ERR": 3.5}),
            clean[18]["targetid"]: ("cigale_missing", None),
        }
        for tid, st in cig_targets.items():
            if st == "STAR":
                continue
            d = desi_by_tid[tid]
            if tid in special:
                name, over = special[tid]
                if over is None:
                    S[name] = {"targetid": tid}
                    continue
                self.add_cigale(tid, d, **over)
                S[name] = {"targetid": tid, **over}
            else:
                self.add_cigale(tid, d)
        # two fits for clean[17]: the main/dark one (same observation) must win even
        # though the sv1 one has the better chi2
        t17 = clean[17]["targetid"]
        d17 = desi_by_tid[t17]
        self.add_cigale(t17, d17, SURVEY="sv1", PROGRAM="dark", CHI2=0.3, LOGM=8.0, LOGSFR=-2.0)
        chosen = next(r for r in self.cigale if r["TARGETID"] == t17 and r["SURVEY"] == "main")
        S["cigale_two_fits"] = {"targetid": t17, "chosen": ["main", "dark"],
                                "chosen_logm": chosen["LOGM"], "other": ["sv1", "dark"]}
        for _ in range(10):
            b, o = self.ls10_object()
            d = dict(SURVEY="main", PROGRAM="dark", HEALPIX=4321, SPECTYPE="GALAXY",
                     TARGET_RA=RA0 + 5.0, TARGET_DEC=DEC0, Z=0.3)
            self.add_cigale(self.main_targetid(b, o), d)

        # ---- expected counts ---------------------------------------------
        def census(rows):
            out = {"QSO": 0, "GALAXY": 0, "STAR": 0}
            for r in rows:
                out[r["spectype"]] += 1
            return out

        sample = [r for r in self.expected_rows if not r["split_source"] and r["has_spectrum"]]
        self.planted["expected"] = {
            "crossmatch_rows": len(self.expected_rows),
            "crossmatch_census": census(self.expected_rows),
            "crossmatch_rows_by_detuid": {r["detuid"]: r["targetid"] for r in self.expected_rows},
            "split_source_detuids": [r["detuid"] for r in self.expected_rows if r["split_source"]],
            "sample_rows": len(sample),
            "sample_census": census(sample),
            "sample_targetids": sorted(r["targetid"] for r in sample),
            "n_flipped_by_preference": 1,
            "n_uncalibrated_kept": 1,
            "n_uncalibrated_dropped": 1,
        }
        self.planted["n_rows"] = {"nway": len(self.nway), "main": len(self.main),
                                  "zall_pix": len(self.desi), "cigale": len(self.cigale)}
        return self.planted

    # -- archives -----------------------------------------------------------
    def write_coadds(self) -> None:
        for (survey, program, pix), entries in sorted(self.spectra.items()):
            path = self.out / "coadd" / f"coadd-{survey}-{program}-{pix}.fits"
            # one foreign target per file, so a reader must select rows by TARGETID
            foreign = dict(TARGETID=9_000_000_000_000 + pix, TARGET_RA=RA0 + 3, TARGET_DEC=DEC0,
                           Z=0.5)
            rows = [e["desi"] for e in entries] + [foreign]
            scales = [e["line_scale"] for e in entries] + [None]
            self._write_coadd(path, rows, scales)
            self.planted["coadd_groups"][f"{survey}-{program}-{pix}"] = [
                int(e["desi"]["TARGETID"]) for e in entries]

    def _write_coadd(self, path: Path, rows: list[dict], scales: list) -> None:
        hdus = [fits.PrimaryHDU()]
        for cam, (lo, hi) in CAMERAS.items():
            wave = lo + 0.8 * np.arange(int(round((hi - lo) / 0.8)) + 1)
            flux = np.zeros((len(rows), wave.size), np.float32)
            ivar = np.zeros_like(flux)
            for i, (row, scale) in enumerate(zip(rows, scales)):
                flux[i] = self._model(wave, row, scale)
                ivar[i] = {"B": 9.0, "R": 12.0, "Z": 7.0}[cam]
                ivar[i, 100:103] = 0.0          # a few masked pixels per camera
                flux[i, 100:103] = 0.0
            hdus.append(fits.ImageHDU(wave, name=f"{cam}_WAVELENGTH"))
            hdus.append(fits.ImageHDU(flux, name=f"{cam}_FLUX"))
            hdus.append(fits.ImageHDU(ivar, name=f"{cam}_IVAR"))
        fm = fits.BinTableHDU.from_columns([
            fits.Column(name="TARGETID", format="K",
                        array=np.array([r["TARGETID"] for r in rows], np.int64)),
            fits.Column(name="TARGET_RA", format="D",
                        array=np.array([r["TARGET_RA"] for r in rows])),
            fits.Column(name="TARGET_DEC", format="D",
                        array=np.array([r["TARGET_DEC"] for r in rows])),
        ])
        fm.name = "FIBERMAP"
        hdus.append(fm)
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(hdus).writeto(path, overwrite=True)

    def _model(self, wave: np.ndarray, row: dict, scale: float | None) -> np.ndarray:
        z = float(row["Z"])
        tid = int(row["TARGETID"])
        level = 4.0 + (tid % 7)
        flux = level * (wave / 6000.0) ** -0.3
        if scale is None:
            return flux.astype(np.float32)
        record = self.planted["line_flux"].setdefault(str(tid), {})
        for name, components in LINES.items():
            total = 0.0
            for rest, ratio in components:
                lam = rest * (1.0 + z)
                sigma = lam * SIGMA_V_KMS / C_KMS
                amp = scale * ratio
                flux += amp * np.exp(-0.5 * ((wave - lam) / sigma) ** 2)
                total += amp * sigma * math.sqrt(2.0 * math.pi)
            record[name] = total
        return flux.astype(np.float32)

    def write_cutouts(self) -> None:
        for tid, ra, dec in self.cutouts:
            path = self.out / "cutouts" / f"{tid}.fits"
            n = CUTOUT_SIZE
            y, x = np.mgrid[0:n, 0:n]
            data = np.zeros((4, n, n), np.float32)
            centre = (n - 1) / 2.0
            for b in range(4):
                sigma = 3.0 + 0.5 * b
                data[b] = 0.002 + (0.5 + 0.2 * b) * np.exp(-((x - centre) ** 2 + (y - centre) ** 2)
                                                           / (2 * sigma ** 2))
            header = fits.Header()
            header["BANDS"] = "griz"
            for i, band in enumerate("griz"):
                header[f"BAND{i}"] = band
            header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
            header["CRVAL1"], header["CRVAL2"] = ra, dec
            header["CRPIX1"], header["CRPIX2"] = n / 2 + 0.5, n / 2 + 0.5
            header["CD1_1"], header["CD2_2"] = -0.262 / 3600.0, 0.262 / 3600.0
            header["IMAGETYP"], header["SURVEY"] = "IMAGE", "LegacySurvey"
            path.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(data, header).writeto(path, overwrite=True)
            self.planted["cutout_targetids"].append(int(tid))

    # -- outputs ------------------------------------------------------------
    def write_all(self) -> dict:
        self.build()
        write_table(self.out / "nway.fits", self.nway, NWAY_SPEC)
        write_table(self.out / "main.fits", self.main, main_spec())
        write_table(self.out / "zall_pix.fits", self.desi, ZALL_SPEC)
        write_table(self.out / "cigale.fits", self.cigale, CIGALE_SPEC)
        self.write_coadds()
        self.write_cutouts()
        (self.out / "planted.json").write_text(
            json.dumps(self.planted, indent=1, sort_keys=True) + "\n")
        self.write_config()
        return self.planted

    def write_config(self) -> None:
        cfg = yaml.safe_load((REPO / "config.yaml").read_text())
        files = {"nway": "nway.fits", "main": "main.fits", "desi_zcat": "zall_pix.fits",
                 "cigale": "cigale.fits"}
        for name, fname in files.items():
            path = self.out / fname
            cfg["inputs"][name] = {"file": fname, "url": f"fixture://{fname}",
                                   "bytes": path.stat().st_size, "md5": md5(path)}
        # outputs land under the repository's gitignored data/fixture_run when a step is
        # run on this config by hand; the tests redirect them to a temp directory
        cfg["paths"] = {"raw": ".", "work": "../../data/fixture_run/work",
                        "staged": "../../data/fixture_run/staged",
                        "provenance": "../../data/fixture_run/provenance"}
        cfg["archives"]["desi_coadd_url"] = "coadd/coadd-{survey}-{program}-{pix}.fits"
        cfg["cutouts"]["size"] = CUTOUT_SIZE
        cfg["spectra"]["workers"] = 2
        cfg["split"]["tolerance"] = 0.5          # 35 rows: one row is 3% of the sample
        (self.out / "config.yaml").write_text(
            "# Generated by make_fixtures.py from the repository config; do not edit.\n"
            + yaml.safe_dump(cfg, sort_keys=False))


def build(out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.fits")) + list(out.glob("coadd/*.fits")) + \
            list(out.glob("cutouts/*.fits")):
        stale.unlink()
    return Builder(out).write_all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    planted = build(args.out)
    n = planted["n_rows"]
    e = planted["expected"]
    print(f"wrote fixtures to {args.out}: nway {n['nway']}, main {n['main']}, "
          f"zall_pix {n['zall_pix']}, cigale {n['cigale']}; "
          f"crossmatch {e['crossmatch_rows']} rows, sample {e['sample_rows']} rows "
          f"{e['sample_census']}")


if __name__ == "__main__":
    main()
