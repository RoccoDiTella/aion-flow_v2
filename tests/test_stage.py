"""Cut 8: the staged HDF5 files carry exactly the contract, row-aligned and inputs only."""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

from aionflow_data import common, crossmatch, fetch_spectra, manifest_split, stage
from aionflow_data.fetch_cutouts import read_cutout
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a, **k: None)
CONTRACT = {
    "source_row": np.int64, "desi_targetid": np.int64, "spectra": np.float32,
    "spectra_ivar": np.float32, "spectra_lambda": np.float32, "redshift": np.float32,
    "flux_w1": np.float32, "flux_w2": np.float32, "flux_w3": np.float32,
    "target_ra": np.float32, "target_dec": np.float32, "image_flux": np.float32,
    "has_spectrum": np.bool_, "has_z": np.bool_, "has_wise": np.bool_, "has_image": np.bool_,
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = make_fx_cfg(tmp_path_factory.mktemp("stage"))
    crossmatch.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    work = common.ledger_path("spectra", cfg).parent.parent / "work"
    shutil.copytree(FIXTURES / "cutouts", work / manifest_split.CUTOUT_DIR)
    manifest = manifest_split.run(cfg, **QUIET)
    summary = stage.run(cfg, **QUIET)
    staged = common.ledger_path("stage", cfg).parent.parent / "staged"
    return cfg, work, staged, manifest, summary, common.read_ledger("stage", cfg)


def _open(staged, split):
    return h5py.File(staged / stage.SPLIT_FILE.format(split=split), "r")


# ----------------------------------------------------------------------------- units

def test_row_chunks_are_row_aligned():
    assert stage.row_chunks((100_000, 7781), 4, 256 * 1024) == (8, 7781)
    assert stage.row_chunks((500, 4, 32, 32), 4, 256 * 1024) == (16, 4, 32, 32)
    assert stage.row_chunks((500, 4, 160, 160), 4, 256 * 1024) == (1, 4, 160, 160)
    assert stage.row_chunks((10, 7781), 4, 256 * 1024) == (8, 7781)
    assert stage.row_chunks((10,), 8, 256 * 1024) == (10,)
    assert stage.row_chunks((0, 7781), 4, 256 * 1024) is None


# ----------------------------------------------------------------------------- contract

def test_datasets_dtypes_shapes_and_attributes(run):
    cfg, _, staged, manifest, summary, _ = run
    nbin, size = cfg["spectra"]["nbin"], cfg["cutouts"]["size"]
    for split in manifest_split.SPLITS:
        with _open(staged, split) as h:
            assert set(h.keys()) == set(CONTRACT)
            n = h["desi_targetid"].shape[0]
            assert n == summary["splits"][split]["rows"] > 0
            for key, dtype in CONTRACT.items():
                assert h[key].dtype == dtype, key
            assert h["spectra"].shape == h["spectra_ivar"].shape == (n, nbin)
            assert h["spectra_lambda"].shape == (nbin,)
            assert h["image_flux"].shape == (n, 4, size, size)
            for key in CONTRACT:
                if key == "spectra_lambda":
                    continue
                assert h[key].shape[0] == n, key
            assert [b.decode() for b in h.attrs["image_bands"]] == list(stage.IMAGE_BANDS)
            assert h.attrs["image_size"] == size and h.attrs["split"] == split


def test_no_label_dataset_is_staged(run):
    _, _, staged, _, _, _ = run
    forbidden = {"log_ml_flux_1", "log_lx", "ml_flux_1", "log_sfr", "logmstar_cigale",
                 "flux_sig_lo", "det_like_0", "ape_cts_1"}
    for split in manifest_split.SPLITS:
        with _open(staged, split) as h:
            assert not (set(h.keys()) & forbidden)


def test_rows_follow_the_manifest_in_source_row_order(run):
    _, work, staged, manifest, _, _ = run
    sample = manifest[manifest["in_sample"]]
    seen = []
    with h5py.File(work / fetch_spectra.SOURCE) as src:
        for split in manifest_split.SPLITS:
            rows = sample[sample["split"] == split].sort_values("source_row")
            with _open(staged, split) as h:
                src_rows = h["source_row"][:]
                assert (np.diff(src_rows) > 0).all()
                assert np.array_equal(src_rows, rows["source_row"].to_numpy())
                assert np.array_equal(h["desi_targetid"][:], rows["targetid"].to_numpy())
                assert np.array_equal(h["spectra"][:], src["spectra"][src_rows])
                assert np.array_equal(h["spectra_ivar"][:], src["spectra_ivar"][src_rows])
                assert np.array_equal(h["spectra_lambda"][:], src["spectra_lambda"][:])
                for key, col in stage.SCALARS.items():
                    assert np.array_equal(h[key][:], rows[col].to_numpy(np.float64)
                                          .astype(np.float32)), key
                for flag in manifest_split.PRESENCE_FLAGS:
                    assert np.array_equal(h[flag][:], rows[flag].to_numpy(bool)), flag
                seen += h["desi_targetid"][:].tolist()
    assert sorted(seen) == sorted(sample["targetid"])
    assert len(seen) == len(set(seen))


def test_images_are_the_cutouts_or_a_zero_frame(run, planted):
    cfg, work, staged, manifest, summary, ledger = run
    size = cfg["cutouts"]["size"]
    missing = planted["scenarios"]["no_cutout"]["targetid"]
    found_missing = False
    for split in manifest_split.SPLITS:
        with _open(staged, split) as h:
            tids = h["desi_targetid"][:]
            has = h["has_image"][:]
            for i, tid in enumerate(tids):
                img = h["image_flux"][i]
                if tid == missing:
                    found_missing = True
                    assert not has[i] and not img.any()
                else:
                    assert has[i] and img.any()
                    expected = read_cutout(work / manifest_split.CUTOUT_DIR / f"{tid}.fits", size)
                    assert np.array_equal(img, expected)
        assert summary["splits"][split]["zero_images"] == int((~has).sum())
        assert ledger["counts"][f"has_image_{split}"] == int(has.sum())
    assert found_missing


def test_chunking_and_compression(run):
    cfg, _, staged, _, _, _ = run
    target = cfg["stage"]["chunk_target_bytes"]
    for split in manifest_split.SPLITS:
        with _open(staged, split) as h:
            for key, ds in h.items():
                if key == "spectra_lambda":
                    continue
                assert ds.chunks is not None, key
                assert ds.chunks[1:] == ds.shape[1:], key
                row_bytes = ds.dtype.itemsize * int(np.prod(ds.shape[1:]))
                assert ds.chunks[0] == min(ds.shape[0], max(1, target // row_bytes)), key
            assert h["spectra"].compression == "gzip" and h["spectra_ivar"].compression == "gzip"
            assert h["image_flux"].compression is None


def test_summary_and_ledger(run):
    cfg, _, staged, manifest, summary, ledger = run
    on_disk = json.loads((staged / stage.SUMMARY).read_text())
    assert on_disk["splits"] == summary["splits"]
    sample = manifest[manifest["in_sample"]]
    assert sum(s["rows"] for s in summary["splits"].values()) == len(sample)
    assert ledger["counts"]["sample_rows"] == len(sample)
    assert set(ledger["inputs"]) == {"manifest", "spectra_source"}
    assert set(ledger["extra"]["outputs"]) == set(manifest_split.SPLITS)
    assert ledger["extra"]["image_size"] == cfg["cutouts"]["size"]
    assert not list(staged.glob("*.part"))


def test_restaging_reproduces_the_same_content(run):
    cfg, _, staged, _, _, _ = run
    before = {}
    for split in manifest_split.SPLITS:
        with _open(staged, split) as h:
            before[split] = {k: h[k][:] for k in h}
    stage.run(cfg, **QUIET)
    for split in manifest_split.SPLITS:
        with _open(staged, split) as h:
            for k, v in before[split].items():
                assert np.array_equal(h[k][:], v), (split, k)


def test_missing_inputs_and_cli(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    with pytest.raises(stage.StageError, match="missing input"):
        stage.run(cfg, **QUIET)
    assert stage.main(["--config", cfg["_config_path"]]) == 1
    crossmatch.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    shutil.copytree(FIXTURES / "cutouts", tmp_path / "work" / manifest_split.CUTOUT_DIR)
    manifest_split.run(cfg, **QUIET)
    assert stage.main(["--config", cfg["_config_path"]]) == 0
    assert (tmp_path / "staged" / "desi_train.hdf5").is_file()
    pd.read_csv(tmp_path / "work" / manifest_split.SPLIT)
