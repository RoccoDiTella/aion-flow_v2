"""Cut 4: X-ray and CIGALE labels, formulas checked by hand and gates by construction."""

from __future__ import annotations

import shutil

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.cosmology import Planck18
from astropy.io import fits

from aionflow_data import common, crossmatch, labels
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a: None)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = make_fx_cfg(tmp_path_factory.mktemp("labels"))
    crossmatch.run(cfg, **QUIET)
    frame = labels.run(cfg, **QUIET)
    return cfg, frame, common.read_ledger("labels", cfg)


@pytest.fixture(scope="module")
def main_table():
    with fits.open(FIXTURES / "main.fits") as handle:
        return handle[1].data.copy()


def _main_row(main_table, detuid):
    det = np.char.strip(main_table["DETUID"].astype(str))
    return main_table[np.flatnonzero(det == detuid)[0]]


def _row(frame, detuid):
    rows = frame[frame["ero_detuid"] == detuid]
    assert len(rows) == 1
    return rows.iloc[0]


def _by_tid(frame, tid):
    rows = frame[frame["targetid"] == tid]
    assert len(rows) >= 1
    return rows.iloc[0]


# ----------------------------------------------------------------------------- formulas

def test_asymmetric_errors_by_hand():
    f, lo, hi = 2e-13, 3e-14, 5e-14
    lf, slo, shi = labels.log_with_asym_errors([f], [lo], [hi], cap_dex=1.5)
    assert lf[0] == pytest.approx(np.log10(f))
    assert slo[0] == pytest.approx(-np.log10(1 - lo / f))
    assert shi[0] == pytest.approx(np.log10(1 + hi / f))
    # lower error swallowing the flux, a cap violation, and a non-measurement are all NaN
    lf, slo, shi = labels.log_with_asym_errors([f, f, 0.0, -1.0], [1.2 * f, lo, lo, lo],
                                               [hi, 40 * f, hi, hi], cap_dex=1.5)
    assert np.isnan(lf).all() and np.isnan(slo).all() and np.isnan(shi).all()


def test_log_luminosity_matches_astropy():
    z = np.array([0.1, 0.9, 2.3, -0.001, 0.0])
    lf = np.full(5, -13.0)
    lx, n_bad = labels.log_luminosity(lf, z)
    dl = Planck18.luminosity_distance(z[:3]).to(u.cm).value
    assert np.allclose(lx[:3], -13.0 + np.log10(4 * np.pi * dl ** 2), rtol=0, atol=1e-10)
    assert np.isnan(lx[3:]).all() and n_bad == 2


# ----------------------------------------------------------------------------- X-ray

def test_row_set_and_order_follow_the_crossmatch(run, planted):
    cfg, frame, ledger = run
    xm = pd.read_parquet(common.ledger_path("labels", cfg).parent.parent / "work"
                         / crossmatch.OUTPUT)
    assert len(frame) == len(xm) == planted["expected"]["crossmatch_rows"]
    assert list(frame["ero_detuid"]) == list(xm["ero_detuid"])
    assert list(frame["targetid"]) == list(xm["targetid"])
    assert ledger["counts"]["rows_out"] == len(frame)


def test_fluxes_and_luminosity_against_the_catalogue(run, main_table, planted):
    _, frame, _ = run
    for detuid in planted["scenarios"]["clean"]["detuids"][:5]:
        m = _main_row(main_table, detuid)
        r = _row(frame, detuid)
        f, lo, hi = (float(m["ML_FLUX_1"]), float(m["ML_FLUX_LOWERR_1"]),
                     float(m["ML_FLUX_UPERR_1"]))
        assert r["log_flux_1"] == pytest.approx(np.log10(f), rel=1e-6)
        assert r["log_flux_1_sig_lo"] == pytest.approx(-np.log10(1 - lo / f), rel=1e-5)
        assert r["log_flux_1_sig_hi"] == pytest.approx(np.log10(1 + hi / f), rel=1e-5)
        assert r["det_like_0"] == pytest.approx(float(m["DET_LIKE_0"]))
        for b in (2, 3):
            assert r[f"det_like_p{b}"] == pytest.approx(float(m[f"DET_LIKE_P{b}"]))
            assert r[f"log_flux_p{b}"] == pytest.approx(np.log10(float(m[f"ML_FLUX_P{b}"])),
                                                        rel=1e-6)
        if r["z"] > 0:
            dl = Planck18.luminosity_distance(r["z"]).to(u.cm).value
            assert r["log_lx"] == pytest.approx(r["log_flux_1"] + np.log10(4 * np.pi * dl ** 2),
                                                rel=1e-9)
    assert not any(c.endswith(("_p1", "_p4")) for c in frame.columns)


def test_luminosity_is_missing_at_nonpositive_redshift(run, planted):
    _, frame, ledger = run
    tid = planted["scenarios"]["z_nonpositive"]["targetid"]
    r = _by_tid(frame, tid)
    assert np.isnan(r["log_lx"]) and np.isfinite(r["log_flux_1"])
    assert ledger["extra"]["xray"]["log_lx"]["z_le_0"] == 1
    assert ledger["counts"]["log_lx_finite"] == len(frame) - 1


def test_counts_integrity(run, main_table, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    # wrapped counts: the band-1 triple is missing, other bands intact
    r = _row(frame, S["ape_cts_wrapped"]["detuid"])
    assert pd.isna(r["ape_cts_1"]) and np.isnan(r["ape_bkg_1"]) and np.isnan(r["ape_exp_1"])
    assert pd.notna(r["ape_cts_p2"]) and np.isfinite(r["ape_exp_p2"])
    assert ledger["extra"]["xray"]["1"]["ape_cts_wrapped"] == 1
    assert ledger["counts"]["ape_triple_1_complete"] == len(frame) - 1
    # negative background: clipped to zero, flagged, counted, row kept
    r = _row(frame, S["ape_bkg_negative"]["detuid"])
    assert r["ape_bkg_p2"] == 0.0 and r["ape_bkg_negative_p2"] and pd.notna(r["ape_cts_p2"])
    assert ledger["extra"]["xray"]["p2"]["ape_bkg_negative_clipped"] == 1
    assert frame["ape_bkg_negative_p2"].sum() == 1
    # zero counts are ordinary data; the band's flux is gated by its detection likelihood
    for detuid in S["zero_counts_p2"]["detuids"]:
        r = _row(frame, detuid)
        assert r["ape_cts_p2"] == 0 and r["det_like_p2"] == 0.0
    assert ledger["extra"]["xray"]["p2"]["ape_cts_zero"] == 3
    # counts are integers, never "45.0"
    assert str(frame["ape_cts_p3"].dtype) == "Int64"
    m = _main_row(main_table, S["clean"]["detuids"][0])
    r = _row(frame, S["clean"]["detuids"][0])
    assert r["ape_cts_p3"] == int(m["APE_CTS_P3"])
    assert r["ape_exp_p3"] == pytest.approx(float(m["APE_EXP_P3"]))


def test_flux_consistent_with_zero_is_not_a_measurement(run, planted):
    _, frame, ledger = run
    r = _row(frame, planted["scenarios"]["flux_consistent_with_zero"]["detuid"])
    assert np.isnan(r["log_flux_p3"]) and np.isnan(r["log_flux_p3_sig_lo"])
    assert np.isfinite(r["log_flux_p2"])
    assert ledger["extra"]["xray"]["p3"]["flux_not_a_measurement"] == 1
    assert ledger["extra"]["xray"]["1"]["flux_not_a_measurement"] == 0


def test_missing_detuid_in_main_is_an_error(tmp_path, planted):
    cfg = make_fx_cfg(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("nway.fits", "zall_pix.fits", "cigale.fits"):
        shutil.copy(FIXTURES / name, raw / name)
    with fits.open(FIXTURES / "main.fits") as handle:
        data = handle[1].data
        det = np.char.strip(data["DETUID"].astype(str))
        keep = det != planted["scenarios"]["clean"]["detuids"][3]
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data[keep])]).writeto(raw / "main.fits")
    text = yaml.safe_load(open(cfg["_config_path"]))
    text["paths"]["raw"] = str(raw)
    with open(cfg["_config_path"], "w") as fh:
        yaml.safe_dump(text, fh)
    cfg = common.load_config(cfg["_config_path"])
    crossmatch.run(cfg, **QUIET)
    with pytest.raises(labels.LabelsError, match="absent"):
        labels.run(cfg, **QUIET)
    assert labels.main([f"--config={cfg['_config_path']}"]) == 1


# ----------------------------------------------------------------------------- CIGALE

def test_cigale_gates_by_construction(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    c = ledger["extra"]["cigale"]
    r = _by_tid(frame, S["cigale_failed"]["targetid"])
    assert np.isnan(r["logmstar_cigale"]) and np.isnan(r["log_sfr"])
    r = _by_tid(frame, S["cigale_sentinel"]["targetid"])
    assert np.isnan(r["logmstar_cigale"]) and np.isnan(r["log_sfr"])
    r = _by_tid(frame, S["cigale_broad_mass"]["targetid"])
    assert np.isnan(r["logmstar_cigale"]) and np.isfinite(r["log_sfr"])
    r = _by_tid(frame, S["cigale_broad_sfr"]["targetid"])
    assert np.isfinite(r["logmstar_cigale"]) and np.isnan(r["log_sfr"])
    r = _by_tid(frame, S["cigale_wide_err"]["targetid"])
    assert np.isfinite(r["logmstar_cigale"]) and np.isnan(r["log_sfr"])
    assert np.isnan(r["log_sfr_sig_lo"])
    r = _by_tid(frame, S["cigale_missing"]["targetid"])
    assert np.isnan(r["logmstar_cigale"]) and np.isnan(r["log_sfr"])
    assert c["failed_fit"] == 1 and c["sentinel"] == 1
    assert c["broad_mass_pdf"] == 1 and c["broad_sfr_pdf"] == 1
    assert c["log_sfr"]["removed_by_max_sigma"] == 1
    assert c["logmstar_cigale"]["removed_by_max_sigma"] == 0
    stars = frame[frame["spectype"] == "STAR"]
    assert len(stars) == 2 and stars["logmstar_cigale"].isna().all()
    assert ledger["counts"]["log_sfr_finite"] == int(np.isfinite(frame["log_sfr"]).sum())


def test_cigale_duplicate_fit_prefers_the_same_observation(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]["cigale_two_fits"]
    r = _by_tid(frame, S["targetid"])
    assert r["logmstar_cigale"] == pytest.approx(S["chosen_logm"])
    assert ledger["extra"]["cigale"]["duplicate_fits_resolved"] == 1


# ----------------------------------------------------------------------------- contract

def test_output_carries_the_label_contract_and_is_deterministic(run):
    cfg, frame, ledger = run
    assert not [c for c in labels.LABEL_COLUMNS if c not in frame.columns]
    for c in crossmatch.OUTPUT_COLUMNS:
        assert c in frame.columns
    out = common.ledger_path("labels", cfg).parent.parent / "work" / labels.OUTPUT
    reread = pd.read_csv(out)
    assert len(reread) == len(frame) and list(reread.columns) == list(frame.columns)
    # counts are written as integers, never "45.0", and a wrapped count as an empty field
    text = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert not text["ape_cts_p2"].str.contains(r"\.").any()
    assert (text["ape_cts_1"] == "").sum() == 1
    assert text.loc[text["ape_cts_1"] != "", "ape_cts_1"].str.fullmatch(r"-?\d+").all()
    first = common.sha256(out)
    labels.run(cfg, **QUIET)
    assert common.sha256(out) == first
    assert set(ledger["inputs"]) == {"crossmatch", "main", "cigale"}
    assert ledger["inputs"]["main"]["sha256"] == common.sha256(FIXTURES / "main.fits")
    assert ledger["extra"]["gates"]["cigale_max_sigma_dex"] == 3.0
