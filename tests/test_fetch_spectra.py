"""Cut 5: DESI coadd fetching, the camera coadd, shards, resume and the merge."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml
from astropy.io import fits

from aionflow_data import common, crossmatch, fetch_spectra
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a: None)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = make_fx_cfg(tmp_path_factory.mktemp("spectra"))
    crossmatch.run(cfg, **QUIET)
    stats = fetch_spectra.run(cfg, **QUIET)
    return cfg, stats, common.read_ledger("spectra", cfg)


def _shard(cfg, group: str):
    survey, program, pix = group.split("-")
    path = fetch_spectra.shard_path(common.ledger_path("spectra", cfg).parent.parent / "work"
                                    / fetch_spectra.SHARD_DIR, survey, program, int(pix))
    with np.load(path) as z:
        return {k: z[k] for k in ("tid", "flux", "ivar")}


def _source(cfg):
    return h5py.File(common.ledger_path("spectra", cfg).parent.parent / "work"
                     / fetch_spectra.SOURCE, "r")


def _expected_coadd(coadd_path, targetid, grid):
    """An independent implementation of the inverse-variance camera coadd."""
    with fits.open(coadd_path) as h:
        row = int(np.flatnonzero(h["FIBERMAP"].data["TARGETID"] == targetid)[0])
        num = np.zeros(grid.nbin)
        den = np.zeros(grid.nbin)
        for cam in "BRZ":
            wave = h[f"{cam}_WAVELENGTH"].data.astype(float)
            f = h[f"{cam}_FLUX"].data[row].astype(float)
            v = h[f"{cam}_IVAR"].data[row].astype(float)
            for w, fi, vi in zip(wave, f, v):
                c = int(round((w - grid.lam0) / grid.dlam))
                if 0 <= c < grid.nbin and vi > 0:
                    num[c] += fi * vi
                    den[c] += vi
    flux = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    return flux.astype(np.float32), den.astype(np.float32)


# ----------------------------------------------------------------------------- shards

def test_shards_hold_exactly_our_rows(run, planted):
    cfg, stats, _ = run
    all_ours = set()
    for group, targetids in planted["coadd_groups"].items():
        shard = _shard(cfg, group)
        assert sorted(shard["tid"].tolist()) == sorted(targetids)
        assert shard["flux"].shape == (len(targetids), cfg["spectra"]["nbin"])
        assert shard["flux"].dtype == np.float32 and shard["ivar"].dtype == np.float32
        all_ours |= set(targetids)
    absent = planted["scenarios"]["no_spectrum"]
    shard = _shard(cfg, "-".join(str(x) for x in absent["group"]))
    assert shard["tid"].size == 0 and shard["flux"].shape == (0, cfg["spectra"]["nbin"])
    assert stats["fetch"]["absent_coadds"] == 1
    survey, program, pix = absent["group"]
    assert stats["fetch"]["absent"] == [[survey, program, pix]]
    shard_dir = common.ledger_path("spectra", cfg).parent.parent / "work" / fetch_spectra.SHARD_DIR
    assert not list(shard_dir.glob("*.tmp.npz"))
    assert len(list(shard_dir.glob("*.npz"))) == stats["fetch"]["groups"]


def test_camera_coadd_matches_an_independent_computation(run, planted):
    cfg, _, _ = run
    grid = fetch_spectra.Grid.from_config(cfg)
    for group, targetids in planted["coadd_groups"].items():
        shard = _shard(cfg, group)
        tid = targetids[0]
        k = int(np.flatnonzero(shard["tid"] == tid)[0])
        flux, ivar = _expected_coadd(FIXTURES / "coadd" / f"coadd-{group}.fits", tid, grid)
        assert np.array_equal(shard["ivar"][k], ivar)
        assert np.allclose(shard["flux"][k], flux, rtol=1e-6, atol=1e-6)


def test_overlaps_are_coadded_and_masks_respected(run, planted):
    cfg, _, _ = run
    grid = fetch_spectra.Grid.from_config(cfg)
    group, targetids = next(iter(planted["coadd_groups"].items()))
    shard = _shard(cfg, group)
    k = int(np.flatnonzero(shard["tid"] == targetids[0])[0])
    lam = grid.wavelengths()
    ivar, flux = shard["ivar"][k], shard["flux"][k]
    # B and R overlap on 5760-5800 A (51 pixels): ivar adds, 9 + 12
    overlap = (lam >= 5760.0) & (lam <= 5800.0)
    assert overlap.sum() == 51 and np.allclose(ivar[overlap], 21.0)
    # single-camera stretches carry that camera's ivar
    assert np.allclose(ivar[(lam > 3700.0) & (lam < 5700.0)], 9.0)
    assert np.allclose(ivar[(lam > 5900.0) & (lam < 7400.0)], 12.0)
    assert np.allclose(ivar[(lam > 7700.0) & (lam < 9800.0)], 7.0)
    # pixels masked in B (3680-3681.6 A) and in R (5840-5841.6 A) have no other camera
    for start in (3600.0, 5760.0):
        masked = np.isin(np.round(lam, 3), np.round(start + 0.8 * np.arange(100, 103), 3))
        assert masked.sum() == 3
        assert (ivar[masked] == 0).all() and (flux[masked] == 0).all()
    # pixels masked in Z (7600-7601.6 A) lie inside the R/Z overlap and are filled by R
    z_masked = np.isin(np.round(lam, 3), np.round(7520.0 + 0.8 * np.arange(100, 103), 3))
    assert z_masked.sum() == 3 and np.allclose(ivar[z_masked], 12.0)
    assert (flux[z_masked] != 0).all()


# ----------------------------------------------------------------------------- source.h5

def test_source_file_contract(run, planted):
    cfg, stats, ledger = run
    grid = fetch_spectra.Grid.from_config(cfg)
    with_spectra = {t for ids in planted["coadd_groups"].values() for t in ids}
    with _source(cfg) as h:
        assert set(h.keys()) == {"targetid", "spectra", "spectra_ivar", "spectra_lambda"}
        tid = h["targetid"][:]
        assert set(tid.tolist()) == with_spectra and np.unique(tid).size == tid.size
        assert h["spectra"].shape == (len(with_spectra), grid.nbin)
        assert h["spectra"].dtype == np.float32 and h["spectra_ivar"].dtype == np.float32
        assert h["spectra"].chunks[1] == grid.nbin
        assert np.array_equal(h["spectra_lambda"][:], grid.wavelengths().astype(np.float32))
        assert h["spectra_lambda"][0] == 3600.0 and h["spectra_lambda"][-1] == pytest.approx(9824.0)
        assert h.attrs["n_spectra"] == len(with_spectra)
        # every row is finite wherever ivar > 0
        flux, ivar = h["spectra"][:], h["spectra_ivar"][:]
        assert np.isfinite(flux[ivar > 0]).all()
    assert stats["merge"]["duplicates_dropped"] == 0
    assert stats["targets_without_spectrum"] == 1
    assert ledger["counts"]["spectra_merged"] == len(with_spectra)
    assert ledger["counts"]["targets_without_spectrum"] == 1
    assert ledger["counts"]["absent_coadds"] == 1
    assert ledger["extra"]["grid"]["nbin"] == grid.nbin
    assert planted["scenarios"]["no_spectrum"]["targetid"] not in with_spectra


def test_rerun_skips_every_existing_shard(run, monkeypatch):
    cfg, _, _ = run

    def boom(*a, **k):
        raise AssertionError("fetch_group must not be called on a resume")

    monkeypatch.setattr(fetch_spectra, "fetch_group", boom)
    stats = fetch_spectra.run(cfg, **QUIET)
    assert stats["fetch"]["shards_present_before"] == stats["fetch"]["groups"]
    assert stats["fetch"]["groups_fetched"] == 0 and stats["merge"]["unique_targets"] > 0


def test_transient_failure_blocks_the_merge(tmp_path, monkeypatch):
    cfg = make_fx_cfg(tmp_path)
    crossmatch.run(cfg, **QUIET)

    def flaky(url, want, grid, timeout_s):
        raise OSError("connection reset")

    monkeypatch.setattr(fetch_spectra, "fetch_once", flaky)
    with pytest.raises(fetch_spectra.SpectraError, match="rerun"):
        fetch_spectra.run(cfg, backoff_s=0.001, **QUIET)
    work = tmp_path / "work"
    assert not (work / fetch_spectra.SOURCE).exists()
    # every group failed transiently, so no shard was written and nothing is recorded
    assert list((work / fetch_spectra.SHARD_DIR).glob("*.npz")) == []
    assert common.read_ledger("spectra", cfg) is None


def test_transient_failure_is_retried_then_succeeds(tmp_path, monkeypatch, planted):
    cfg = make_fx_cfg(tmp_path)
    crossmatch.run(cfg, **QUIET)
    real = fetch_spectra.fetch_once
    calls = {"n": 0}

    def flaky_once(url, want, grid, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("503")
        return real(url, want, grid, timeout_s)

    monkeypatch.setattr(fetch_spectra, "fetch_once", flaky_once)
    stats = fetch_spectra.run(cfg, workers=1, backoff_s=0.001, **QUIET)
    assert stats["fetch"]["transient_failures"] == 0
    n_planted = sum(len(v) for v in planted["coadd_groups"].values())
    assert stats["fetch"]["spectra_fetched"] == n_planted


def test_merge_keeps_the_copy_with_more_good_pixels(tmp_path):
    grid = fetch_spectra.Grid(3600.0, 0.8, 50)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    flux_a = np.full((1, 50), 1.0, np.float32)
    ivar_a = np.full((1, 50), 2.0, np.float32)
    ivar_a[0, :30] = 0.0                      # 20 good pixels
    flux_b = np.full((1, 50), 5.0, np.float32)
    ivar_b = np.full((1, 50), 1.0, np.float32)  # 50 good pixels
    np.savez(shard_dir / "main__dark__1.npz", tid=np.array([7], np.int64), flux=flux_a,
             ivar=ivar_a)
    np.savez(shard_dir / "main__bright__2.npz", tid=np.array([7, 8], np.int64),
             flux=np.vstack([flux_b, flux_a]), ivar=np.vstack([ivar_b, ivar_a]))
    np.savez(shard_dir / "sv1__dark__3.npz", tid=np.empty(0, np.int64),
             flux=np.empty((0, 50), np.float32), ivar=np.empty((0, 50), np.float32))
    out = tmp_path / "source.h5"
    stats = fetch_spectra.merge(shard_dir, out, grid, **QUIET)
    assert stats == {"shards": 3, "rows": 3, "unique_targets": 2, "duplicates_dropped": 1}
    with h5py.File(out) as h:
        tid = h["targetid"][:]
        row7 = int(np.flatnonzero(tid == 7)[0])
        assert h["spectra"][row7, 0] == 5.0 and h["spectra_ivar"][row7, 0] == 1.0
        assert h["spectra"].shape == (2, 50)
    np.savez(shard_dir / "bad__grid__4.npz", tid=np.array([9], np.int64),
             flux=np.zeros((1, 49), np.float32), ivar=np.zeros((1, 49), np.float32))
    with pytest.raises(fetch_spectra.SpectraError, match="49 bins"):
        fetch_spectra.merge(shard_dir, out, grid, **QUIET)


def test_url_and_shard_naming():
    template = "https://x/{survey}/{program}/{group}/{pix}/coadd-{survey}-{program}-{pix}.fits"
    assert fetch_spectra.coadd_url(template, "main", "dark", 10280) == \
        "https://x/main/dark/102/10280/coadd-main-dark-10280.fits"
    assert fetch_spectra.shard_path(FIXTURES, "sv1", "bright", 5).name == "sv1__bright__5.npz"
    assert fetch_spectra._really_missing(str(FIXTURES / "coadd" / "nope.fits"))
    assert not fetch_spectra._really_missing(str(FIXTURES / "nway.fits"))


def test_cli_limit_groups_and_merge_only(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    # a relative local template resolves against the config file, as paths do
    (tmp_path / "coadd").symlink_to(FIXTURES / "coadd", target_is_directory=True)
    text = yaml.safe_load(open(cfg["_config_path"]))
    text["archives"]["desi_coadd_url"] = "coadd/coadd-{survey}-{program}-{pix}.fits"
    with open(cfg["_config_path"], "w") as fh:
        yaml.safe_dump(text, fh)
    cfg = common.load_config(cfg["_config_path"])
    crossmatch.run(cfg, **QUIET)
    rc = fetch_spectra.main(["--config", cfg["_config_path"], "--limit-groups", "1",
                             "--no-merge"])
    assert rc == 0
    shards = list((tmp_path / "work" / fetch_spectra.SHARD_DIR).glob("*.npz"))
    assert len(shards) == 1
    assert fetch_spectra.main(["--config", cfg["_config_path"], "--merge-only"]) == 0
    with h5py.File(tmp_path / "work" / fetch_spectra.SOURCE) as h:
        assert h.attrs["n_shards"] == 1
