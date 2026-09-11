"""Cut 1: the committed fixtures carry every planted scenario and regenerate identically."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
from astropy.io import fits

from aionflow_data import common
from tests.conftest import FIXTURES


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_fixtures", FIXTURES / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(name: str) -> fits.FITS_rec:
    with fits.open(FIXTURES / name) as handle:
        return handle[1].data.copy()


# ----------------------------------------------------------------------------- inventory

def test_row_counts_match_planted(planted):
    for name, fname in (("nway", "nway.fits"), ("main", "main.fits"),
                        ("zall_pix", "zall_pix.fits"), ("cigale", "cigale.fits")):
        assert len(_table(fname)) == planted["n_rows"][name]


def test_nway_scenarios(planted):
    S = planted["scenarios"]
    t = _table("nway.fits")
    det = np.char.strip(t["DETUID"].astype(str))
    flag = t["NWAY_match_flag"]
    thr = t["NWAY_threshold6"]
    # exactly one secondary-candidate row, on the planted detection
    assert (flag == 2).sum() == 1
    assert det[flag == 2][0] == S["secondary_flag2"]["detuid"]
    # the exact duplicate: two identical primary rows
    dup = np.flatnonzero((det == S["exact_duplicate"]["detuid"]) & (flag == 1))
    assert dup.size == 2
    for name in t.columns.names:
        a, b = t[name][dup[0]], t[name][dup[1]]
        assert a == b or (np.isnan(a) and np.isnan(b))
    # the repeated DETUID with two different primary counterparts
    rep = np.flatnonzero(det == S["repeat_detuid_two_primaries"]["detuid"])
    assert rep.size == 2 and t["LS10_OBJID"][rep[0]] != t["LS10_OBJID"][rep[1]]
    # uncalibrated rows carry NaN thresholds on both sides of 0.05
    for name in ("uncalibrated_keep", "uncalibrated_drop"):
        i = np.flatnonzero(det == S[name]["detuid"])[0]
        assert np.isnan(thr[i]) and t["NWAY_p_any"][i] == pytest.approx(S[name]["p_any"])
    i = np.flatnonzero(det == S["below_threshold6"]["detuid"])[0]
    assert t["NWAY_p_any"][i] < thr[i]
    # collision and split source share a counterpart; X-ray separations as planted
    for name in ("collision", "split_source"):
        ids = ([S[name]["kept_detuid"], S[name]["dropped_detuid"]] if name == "collision"
               else S[name]["detuids"])
        rows = [np.flatnonzero(det == d)[0] for d in ids]
        assert t["LS10_OBJID"][rows[0]] == t["LS10_OBJID"][rows[1]]
        cosdec = np.cos(np.radians(t["DEC"][rows[0]]))
        sep = np.hypot((t["RA"][rows[0]] - t["RA"][rows[1]]) * cosdec,
                       t["DEC"][rows[0]] - t["DEC"][rows[1]]) * 3600
        assert sep == pytest.approx(S[name]["xray_sep_arcsec"], abs=0.05)
    assert (t["LS10_flux_w3"] <= 0).sum() == 1


def test_desi_scenarios(planted):
    S = planted["scenarios"]
    t = _table("zall_pix.fits")
    tid = t["TARGETID"]
    assert (tid <= 0).sum() == 1
    assert (~t["ZCAT_PRIMARY"]).sum() == 1
    assert (tid == S["nonprimary_duplicate"]["targetid"]).sum() == 2
    i = np.flatnonzero(tid == S["zwarn_nonzero"]["targetid"])[0]
    assert t["ZWARN"][i] == 4
    i = np.flatnonzero(tid == S["z_nonpositive"]["targetid"])[0]
    assert t["Z"][i] < 0
    # main-survey TARGETIDs decode to release 9010; backup ones to release 0
    rel = (tid.astype(np.int64) >> 42) & 0xFFFF
    for name in ("tie_two_main",):
        assert rel[tid == S[name]["chosen_targetid"]][0] == 9010
    assert rel[tid == S["tie_main_vs_backup"]["nearest_targetid"]][0] == 0
    assert rel[tid == S["backup_only"]["targetid"]][0] == 0
    # decoded triples agree with the columns on LS-encoded rows
    ls = t["BRICKID"] > 0
    assert np.array_equal((tid[ls] >> 22) & 0xFFFFF, t["BRICKID"][ls])
    assert np.array_equal(tid[ls] & 0x3FFFFF, t["BRICK_OBJID"][ls])


def test_main_scenarios(planted):
    S = planted["scenarios"]
    t = _table("main.fits")
    det = np.char.strip(t["DETUID"].astype(str))
    assert t["APE_CTS_1"].dtype.kind == "i" and t["APE_CTS_1"].dtype.itemsize == 2
    i = np.flatnonzero(det == S["ape_cts_wrapped"]["detuid"])[0]
    assert t["APE_CTS_1"][i] == -30000
    assert (t["APE_CTS_1"] < 0).sum() == 1
    i = np.flatnonzero(det == S["ape_bkg_negative"]["detuid"])[0]
    assert t["APE_BKG_P2"][i] < 0
    i = np.flatnonzero(det == S["flux_consistent_with_zero"]["detuid"])[0]
    assert t["ML_FLUX_LOWERR_P3"][i] >= t["ML_FLUX_P3"][i]
    zero = [np.flatnonzero(det == d)[0] for d in S["zero_counts_p2"]["detuids"]]
    assert all(t["APE_CTS_P2"][z] == 0 for z in zero)
    assert all(t["APE_POIS_P2"][z] == pytest.approx(-9.99) for z in zero)
    # every NWAY detection has a Main row
    nway_det = set(np.char.strip(_table("nway.fits")["DETUID"].astype(str)))
    assert nway_det <= set(det)
    assert len(set(det)) > len(nway_det)


def test_cigale_scenarios(planted):
    S = planted["scenarios"]
    t = _table("cigale.fits")
    tid = t["TARGETID"]
    r = np.flatnonzero(tid == S["cigale_failed"]["targetid"])[0]
    assert t["LOGM"][r] == 0.0 and t["LOGSFR"][r] == 0.0
    r = np.flatnonzero(tid == S["cigale_sentinel"]["targetid"])[0]
    assert t["LOGM"][r] <= -90
    r = np.flatnonzero(tid == S["cigale_broad_mass"]["targetid"])[0]
    assert t["FLAG_MASSPDF"][r] > 5 and 0.2 < t["FLAG_SFRPDF"][r] < 5
    r = np.flatnonzero(tid == S["cigale_wide_err"]["targetid"])[0]
    assert t["LOGSFR_ERR"][r] > 3
    assert (tid == S["cigale_missing"]["targetid"]).sum() == 0
    two = np.flatnonzero(tid == S["cigale_two_fits"]["targetid"])
    assert two.size == 2
    surveys = sorted(np.char.strip(t["SURVEY"][two].astype(str)))
    assert surveys == ["main", "sv1"]
    stars = {c for c in planted["scenarios"]["clean"]["targetids"]}
    assert len(stars) == 30


def test_archives_match_planted(planted):
    S = planted["scenarios"]
    # coadd files exist for every planted group, not for the no-spectrum group
    for group, targetids in planted["coadd_groups"].items():
        path = FIXTURES / "coadd" / f"coadd-{group}.fits"
        with fits.open(path) as handle:
            names = [h.name for h in handle]
            for cam in "BRZ":
                assert f"{cam}_WAVELENGTH" in names and f"{cam}_FLUX" in names \
                    and f"{cam}_IVAR" in names
            fm = handle["FIBERMAP"].data["TARGETID"]
            assert set(targetids) < set(fm.tolist())          # plus one foreign target
            assert handle["B_FLUX"].data.shape[0] == fm.size
            b_wave = handle["B_WAVELENGTH"].data
            assert b_wave[0] == 3600.0 and np.allclose(np.diff(b_wave), 0.8)
            assert handle["Z_WAVELENGTH"].data[-1] == pytest.approx(9824.0)
    s, p, pix = S["no_spectrum"]["group"]
    assert not (FIXTURES / "coadd" / f"coadd-{s}-{p}-{pix}.fits").exists()
    assert S["no_spectrum"]["targetid"] not in {
        t for ids in planted["coadd_groups"].values() for t in ids}
    # cutouts exist for the planted list and not for the cutout-less target; the fixture
    # config sets a small cutout size so the committed files stay small
    size = common.load_config(FIXTURES / "config.yaml")["cutouts"]["size"]
    for tid in planted["cutout_targetids"]:
        with fits.open(FIXTURES / "cutouts" / f"{tid}.fits") as handle:
            assert handle[0].data.shape == (4, size, size)
            assert handle[0].header["BANDS"] == "griz"
            assert (FIXTURES / "cutouts" / f"{tid}.fits").stat().st_size >= 10_000
    assert not (FIXTURES / "cutouts" / f"{S['no_cutout']['targetid']}.fits").exists()
    # planted line fluxes are positive and on targets that have spectra
    with_spectra = {t for ids in planted["coadd_groups"].values() for t in ids}
    for tid, lines in planted["line_flux"].items():
        assert int(tid) in with_spectra
        assert set(lines) == {"nev_3426", "oii_3727", "hbeta_4862", "oiii_4960",
                              "oiii_5008", "halpha_6564"}
        assert all(v > 0 for v in lines.values())


def test_expected_counts_are_self_consistent(planted):
    e = planted["expected"]
    assert e["crossmatch_rows"] == sum(e["crossmatch_census"].values())
    assert e["sample_rows"] == sum(e["sample_census"].values()) == len(e["sample_targetids"])
    assert len(e["split_source_detuids"]) == 2
    # two split-source rows, one row without a spectrum, one without a cutout
    assert e["sample_rows"] == e["crossmatch_rows"] - 2 - 1 - 1


def test_fixture_config_loads_and_describes_the_files(fx_cfg):
    for name in common.INPUT_NAMES:
        entry = fx_cfg["inputs"][name]
        path = FIXTURES / entry["file"]
        assert path.stat().st_size == entry["bytes"]
        assert common.file_digests(path)["md5"] == entry["md5"]
    assert fx_cfg["paths"]["raw"] == str(FIXTURES)
    assert fx_cfg["split"] == {"seed": 42, "fractions": [0.8, 0.1, 0.1]}


# ----------------------------------------------------------------------------- determinism

def test_regeneration_is_byte_identical(tmp_path):
    generator = _load_generator()
    generator.build(tmp_path)
    committed = sorted(p for p in FIXTURES.rglob("*")
                       if p.is_file() and p.suffix in (".fits", ".json", ".yaml"))
    assert committed, "no fixtures committed"
    for path in committed:
        rel = path.relative_to(FIXTURES)
        fresh = tmp_path / rel
        assert fresh.is_file(), f"{rel} not regenerated"
        if path.suffix == ".json":
            assert json.loads(fresh.read_text()) == json.loads(path.read_text())
        else:
            assert common.sha256(fresh) == common.sha256(path), f"{rel} differs on regeneration"
