"""Cut 10: the baseline's line fluxes, recovered from planted lines by construction."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from aionflow_data import common, crossmatch, fetch_spectra, line_features, linefit, manifest_split
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a, **k: None)
C = linefit.C_KMS


def synthetic(name: str, amp: float, sigma_v: float, noise: float = 0.05, z: float = 0.0):
    """A rest-frame window spectrum with the complex's primary species planted."""
    lo, hi = linefit.COMPLEXES[name]["window"]
    lam = np.arange(lo - 20, hi + 20, 0.8 / (1 + z))
    flux = 3.0 + 0.001 * (lam - lam.mean())
    for sp, lines in linefit.COMPLEXES[name]["species"]:
        if sp != linefit.COMPLEXES[name]["primary"][0]:
            continue
        for lam0, ratio in lines:
            sigma = lam0 * sigma_v / C
            flux += amp * ratio * np.exp(-0.5 * ((lam - lam0) / sigma) ** 2)
    rng = np.random.default_rng(1)
    flux += rng.normal(0, noise, lam.size)
    ivar = np.full(lam.size, 1 / noise ** 2)
    return lam, flux, ivar


# ----------------------------------------------------------------------------- fitter

@pytest.mark.parametrize("name", ["oiii", "hbeta", "halpha", "nev"])
def test_fit_one_recovers_a_planted_line(name):
    amp, sigma_v = 40.0, 250.0
    lam, flux, ivar = synthetic(name, amp, sigma_v)
    fit = linefit.fit_one(name, lam, flux, ivar)
    assert fit["status"] == "ok"
    assert fit["amp"] == pytest.approx(amp, rel=0.01)
    assert fit["sigma_kms"] == pytest.approx(sigma_v, rel=0.02)
    assert abs(fit["v_kms"]) < 10
    lam0 = linefit.COMPLEXES[name]["primary"][1]
    expected = amp * (lam0 * sigma_v / C) * np.sqrt(2 * np.pi)
    assert fit["flux"] == pytest.approx(expected, rel=0.02)
    assert 0 < fit["flux_err"] < 0.05 * fit["flux"]
    # at redshift z the same rest-frame fit integrates over (1 + z) times the wavelength
    shifted = linefit.fit_one(name, lam, flux, ivar, z=0.5)
    assert shifted["flux"] == pytest.approx(1.5 * fit["flux"], rel=1e-9)
    assert fit["an"] == pytest.approx(amp / 0.05, rel=0.1)
    assert fit["c0"] == pytest.approx(3.0, abs=0.05) and not fit["at_bound"]


def test_fit_one_status_codes():
    lam, flux, ivar = synthetic("oiii", 40.0, 250.0)
    assert linefit.fit_one("oiii", lam[:30], flux[:30], ivar[:30])["status"] == "off_grid"
    ivar_bad = ivar.copy()
    ivar_bad[::2] = 0.0
    assert linefit.fit_one("oiii", lam, flux, ivar_bad)["status"] == "window_ivar"
    assert linefit.fit_one("oiii", lam, np.full_like(flux, np.nan), ivar)["status"] == "window_ivar"


def test_in_window_rule():
    z = np.array([-0.1, 0.0, 0.05, 0.5, 0.9, 1.2, 1.9])
    lo, hi = 3600.0, 9824.0
    F, T = False, True
    assert list(linefit.in_window("oiii", z, lo, hi)) == [F, F, T, T, T, F, F]
    assert list(linefit.in_window("nev", z, lo, hi)) == [F, F, F, T, T, T, F]
    assert list(linefit.in_window("halpha", z, lo, hi)) == [F, F, T, F, F, F, F]


# ----------------------------------------------------------------------------- pipeline

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = make_fx_cfg(tmp_path_factory.mktemp("lines"))
    crossmatch.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    work = common.ledger_path("spectra", cfg).parent.parent / "work"
    shutil.copytree(FIXTURES / "cutouts", work / manifest_split.CUTOUT_DIR)
    manifest_split.run(cfg, **QUIET)
    features = line_features.run(cfg, nproc=2, **QUIET)
    return cfg, work, features, common.read_ledger("line_features", cfg)


def test_every_sample_row_gets_a_row(run, planted):
    _, work, features, ledger = run
    assert len(features) == planted["expected"]["sample_rows"]
    assert sorted(features["targetid"]) == planted["expected"]["sample_targetids"]
    for _, stem in line_features.LINES:
        assert f"{stem}_flux" in features.columns
        assert (features[f"{stem}_flux"] >= 0).all()
        assert features[f"{stem}_flux"].notna().all()
    z_neg = planted["scenarios"]["z_nonpositive"]["targetid"]
    row = features[features["targetid"] == z_neg].iloc[0]
    assert row["oiii_5007_status"] == "no_redshift" and row["oiii_5007_flux"] == 0.0
    reread = pd.read_csv(work / line_features.FEATURES)
    assert list(reread.columns) == list(features.columns)
    assert ledger["counts"]["sample_rows"] == len(features)


def test_planted_line_fluxes_are_recovered_in_the_observed_frame(run, planted):
    cfg, _, features, _ = run
    s = cfg["spectra"]
    lo = float(s["lam0_angstrom"])
    hi = lo + float(s["dlam_angstrom"]) * (int(s["nbin"]) - 1)
    planted_key = {"oiii_5007": "oiii_5008", "nev_3426": "nev_3426", "halpha": "halpha_6564",
                   "hbeta": "hbeta_4862"}
    checked = 0
    for tid, lines in planted["line_flux"].items():
        rows = features[features["targetid"] == int(tid)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for name, stem in line_features.LINES:
            expected = lines[planted_key[stem]]
            if linefit.in_window(name, np.array([row["z"]]), lo, hi)[0]:
                assert row[f"{stem}_status"] == "ok", (tid, stem)
                assert row[f"{stem}_flux"] == pytest.approx(expected, rel=0.03), (tid, stem)
                assert row[f"{stem}_an"] > 20
                checked += 1
            else:
                assert row[f"{stem}_status"] == "not_in_window", (tid, stem)
                assert row[f"{stem}_flux"] == 0.0
    assert checked >= 8


def test_lineless_spectra_measure_nothing_significant(run, planted):
    _, _, features, _ = run
    with_lines = {int(t) for t in planted["line_flux"]}
    quiet = features[~features["targetid"].isin(with_lines)]
    assert len(quiet) > 5
    for _, stem in line_features.LINES:
        ok = quiet[quiet[f"{stem}_status"] == "ok"]
        assert (ok[f"{stem}_an"] < 5).all()


def test_balmer_decrement_and_ledger(run):
    _, work, features, ledger = run
    # planted H-alpha and H-beta share an amplitude, so their flux ratio is the ratio
    # of their observed widths, i.e. of their wavelengths
    both = features[(features["halpha_status"] == "ok") & (features["hbeta_status"] == "ok")
                    & (features["halpha_an"] > 5) & (features["hbeta_an"] > 5)]
    assert len(both) >= 1
    ratio = (both["halpha_flux"] / both["hbeta_flux"]).to_numpy()
    assert np.allclose(ratio, 6564.614 / 4862.683, rtol=0.03)
    counts = ledger["counts"]
    assert counts["halpha_measured"] >= 1 and counts["oiii_5007_measured"] >= 3
    fits = pd.read_csv(work / line_features.FITS)
    assert {"targetid", "line", "status", "flux", "flux_err", "amp", "sigma_kms"} <= set(fits)
    assert counts["fits"] == len(fits)


def test_rerun_is_deterministic_and_cli_works(run):
    cfg, work, _, _ = run
    first = common.sha256(work / line_features.FEATURES)
    assert line_features.main(["--config", cfg["_config_path"], "--nproc", "1"]) == 0
    assert common.sha256(work / line_features.FEATURES) == first


def test_missing_inputs(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    with pytest.raises(line_features.LineFeaturesError, match="missing input"):
        line_features.run(cfg, **QUIET)
    assert line_features.main(["--config", cfg["_config_path"]]) == 1
