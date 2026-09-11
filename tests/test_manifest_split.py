"""Cut 7: presence flags, the sample, connected-component grouping and the keyed split."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from aionflow_data import common, crossmatch, fetch_spectra, manifest_split
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a: None)


def prepare(cfg):
    """Run the steps the manifest needs; cutouts are the fixture files, copied."""
    crossmatch.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    work = common.ledger_path("spectra", cfg).parent.parent / "work"
    shutil.copytree(FIXTURES / "cutouts", work / manifest_split.CUTOUT_DIR)
    return work


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = make_fx_cfg(tmp_path_factory.mktemp("manifest"))
    work = prepare(cfg)
    manifest = manifest_split.run(cfg, **QUIET)
    return cfg, work, manifest, common.read_ledger("manifest_split", cfg)


def _by_tid(frame, tid):
    rows = frame[frame["targetid"] == tid]
    assert len(rows) >= 1
    return rows.iloc[0]


# ----------------------------------------------------------------------------- units

def test_hash_is_keyed_deterministic_and_uniform_enough():
    keys = [f"sm03_{i:06d}" for i in range(2000)]
    u = manifest_split.hash_unit(keys, "salt-a")
    assert np.array_equal(u, manifest_split.hash_unit(keys, "salt-a"))
    assert not np.array_equal(u, manifest_split.hash_unit(keys, "salt-b"))
    assert (u >= 0).all() and (u < 1).all() and 0.45 < u.mean() < 0.55
    split = manifest_split.assign(keys, "salt-a", [0.8, 0.1, 0.1])
    frac = {s: (split == s).mean() for s in manifest_split.SPLITS}
    assert abs(frac["train"] - 0.8) < 0.03 and abs(frac["test"] - 0.1) < 0.03
    with pytest.raises(manifest_split.ManifestError, match="sum to 1"):
        manifest_split.assign(keys, "s", [0.5, 0.3, 0.3])


def test_components_join_shared_detections_and_targets():
    det = np.array(["d1", "d1", "d2", "d3", "d4"])
    tid = np.array([1, 2, 2, 3, 3])
    comp = manifest_split.components(det, tid)
    # d1-t1, d1-t2, d2-t2 form one component keyed by its smallest DETUID; d3-t3, d4-t3 another
    assert list(comp) == ["d1", "d1", "d1", "d3", "d3"]
    # order independence
    order = np.array([4, 2, 0, 3, 1])
    assert list(manifest_split.components(det[order], tid[order])) == list(comp[order])


# ----------------------------------------------------------------------------- flags

def test_presence_flags_by_construction(run, planted):
    _, _, manifest, ledger = run
    S = planted["scenarios"]
    assert not _by_tid(manifest, S["zwarn_nonzero"]["targetid"])["has_z"]
    assert not _by_tid(manifest, S["z_nonpositive"]["targetid"])["has_z"]
    r = _by_tid(manifest, S["w3_nonpositive"]["targetid"])
    assert not r["has_w3"] and r["has_w1"] and r["has_wise"]
    assert not _by_tid(manifest, S["no_cutout"]["targetid"])["has_image"]
    assert not _by_tid(manifest, S["no_spectrum"]["targetid"])["has_spectrum"]
    assert _by_tid(manifest, S["no_spectrum"]["targetid"])["source_row"] == -1
    sample = manifest[manifest["in_sample"]]
    assert sample["has_spectrum"].all()
    assert int(sample["has_image"].sum()) == len(set(planted["cutout_targetids"])
                                                 & set(sample["targetid"]))
    assert int((~sample["has_z"]).sum()) == 2
    assert int((~sample["has_w3"]).sum()) == 1 and sample["has_wise"].all()
    p = ledger["extra"]
    assert p["presence"]["cutouts_unreadable"] == 0
    assert p["presence"]["z_flagged_by_zwarn"] == 1
    assert p["presence_in_sample"]["has_image"] == int(sample["has_image"].sum())
    assert p["presence_fraction_in_sample"]["has_spectrum"] == 1.0


def test_source_row_indexes_the_spectra_file(run):
    _, work, manifest, _ = run
    with h5py.File(work / fetch_spectra.SOURCE) as h:
        tid = h["desi_targetid"][:]
    has = manifest[manifest["has_spectrum"]]
    assert np.array_equal(tid[has["source_row"].to_numpy()], has["targetid"].to_numpy())


# ----------------------------------------------------------------------------- sample and split

def test_sample_matches_planted(run, planted):
    _, _, manifest, ledger = run
    e = planted["expected"]
    sample = manifest[manifest["in_sample"]]
    assert len(sample) == e["sample_rows"]
    assert sample["spectype"].value_counts().to_dict() == e["sample_census"]
    assert sorted(sample["targetid"]) == e["sample_targetids"]
    assert not sample["targetid"].duplicated().any()
    excluded = manifest[manifest["split_source"]]
    assert sorted(excluded["ero_detuid"]) == sorted(e["split_source_detuids"])
    assert (~excluded["in_sample"]).all() and excluded["split"].isna().all()
    filters = {f["filter"]: f for f in ledger["filters"]}
    assert filters["split_source_pairs_excluded"]["dropped"] == 2
    assert filters["has_spectrum"]["dropped"] == 1
    assert ledger["counts"]["sample_rows"] == e["sample_rows"]
    assert ledger["counts"]["components"] == e["sample_rows"]
    assert ledger["counts"]["largest_component_rows"] == 1


def test_split_files_agree_and_cover_the_sample(run):
    _, work, manifest, ledger = run
    split = pd.read_csv(work / manifest_split.SPLIT)
    assert list(split.columns) == ["targetid", "split"]
    sample = manifest[manifest["in_sample"]]
    assert len(split) == len(sample)
    merged = sample.merge(split, on="targetid", suffixes=("", "_file"))
    assert (merged["split"] == merged["split_file"]).all()
    assert set(split["split"]) <= set(manifest_split.SPLITS)
    counts = split["split"].value_counts().to_dict()
    assert counts == {k: ledger["counts"][f"rows_{k}"] for k in counts}
    assert sum(counts.values()) == len(sample)
    reread = pd.read_csv(work / manifest_split.MANIFEST)
    assert list(reread.columns) == manifest_split.MANIFEST_COLUMNS
    assert reread["in_sample"].sum() == len(sample)
    assert reread["split"].isna().sum() == len(reread) - len(sample)


def test_no_component_crosses_splits_and_assignment_is_stable(run):
    cfg, work, manifest, _ = run
    sample = manifest[manifest["in_sample"]]
    per_component = sample.groupby("component")["split"].nunique()
    assert (per_component == 1).all()
    first = pd.read_csv(work / manifest_split.SPLIT)
    manifest_split.run(cfg, **QUIET)
    again = pd.read_csv(work / manifest_split.SPLIT)
    pd.testing.assert_frame_equal(first, again)
    # the assignment depends on the key only, not on row order
    keys = sample["component"].to_numpy()
    shuffled = np.random.default_rng(0).permutation(len(keys))
    a = manifest_split.assign(keys, cfg["split"]["hash"]["salt"], cfg["split"]["fractions"])
    b = manifest_split.assign(keys[shuffled], cfg["split"]["hash"]["salt"],
                              cfg["split"]["fractions"])
    assert np.array_equal(a[shuffled], b)
    assert not np.array_equal(a, manifest_split.assign(keys, "another-salt",
                                                       cfg["split"]["fractions"]))


def test_drift_check_refuses_a_bad_split(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    prepare(cfg)
    text = yaml.safe_load(open(cfg["_config_path"]))
    text["split"]["tolerance"] = 1e-4
    with open(cfg["_config_path"], "w") as fh:
        yaml.safe_dump(text, fh)
    cfg = common.load_config(cfg["_config_path"])
    with pytest.raises(manifest_split.ManifestError, match="drift"):
        manifest_split.run(cfg, **QUIET)
    assert common.read_ledger("manifest_split", cfg) is None
    assert manifest_split.main(["--config", cfg["_config_path"]]) == 1


def test_missing_inputs_are_reported(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    crossmatch.run(cfg, **QUIET)
    with pytest.raises(manifest_split.ManifestError, match="missing input"):
        manifest_split.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    with pytest.raises(manifest_split.ManifestError, match="cutout directory"):
        manifest_split.run(cfg, **QUIET)
