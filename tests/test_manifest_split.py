"""Cut 7: presence flags, the sample, and the seeded permutation split."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

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

def test_assign_is_a_seeded_permutation_cut_at_the_fractions():
    tids = np.arange(1000, 3000) * 7
    split = manifest_split.assign(tids, 42, [0.8, 0.1, 0.1])
    counts = {s: int((split == s).sum()) for s in manifest_split.SPLITS}
    assert counts == {"train": 1600, "val": 200, "test": 200}
    # the same seed gives the same split whatever the row order; another seed does not
    order = np.random.default_rng(0).permutation(tids.size)
    assert np.array_equal(manifest_split.assign(tids[order], 42, [0.8, 0.1, 0.1]), split[order])
    assert not np.array_equal(manifest_split.assign(tids, 43, [0.8, 0.1, 0.1]), split)
    # it is the permutation the paper states: first 80% of RandomState(seed).permutation
    perm = np.random.RandomState(42).permutation(tids.size)
    assert (split[np.sort(tids).searchsorted(tids)][perm[:1600]] == "train").all()
    with pytest.raises(manifest_split.ManifestError, match="sum to 1"):
        manifest_split.assign(tids, 42, [0.5, 0.3, 0.3])
    with pytest.raises(manifest_split.ManifestError, match="unique"):
        manifest_split.assign([1, 1, 2], 42, [0.8, 0.1, 0.1])
    # rounding at the cumulative edges: 34 rows cut at round(27.2) and round(30.6)
    small = manifest_split.assign(np.arange(34), 1, [0.8, 0.1, 0.1])
    assert [int((small == s).sum()) for s in manifest_split.SPLITS] == [27, 4, 3]


# ----------------------------------------------------------------------------- flags

def test_presence_flags_by_construction(run, planted):
    _, _, manifest, ledger = run
    S = planted["scenarios"]
    assert not _by_tid(manifest, S["zwarn_nonzero"]["targetid"])["has_z"]
    assert not _by_tid(manifest, S["z_nonpositive"]["targetid"])["has_z"]
    assert _by_tid(manifest, S["w3_nonpositive"]["targetid"])["has_wise"]
    assert not _by_tid(manifest, S["no_cutout"]["targetid"])["has_image"]
    assert not _by_tid(manifest, S["no_spectrum"]["targetid"])["has_spectrum"]
    sample = manifest[manifest["in_sample"]]
    assert sample["has_spectrum"].all() and sample["has_image"].all()
    assert sorted(sample.loc[~sample["has_z"], "targetid"]) == sorted(
        [S["zwarn_nonzero"]["targetid"], S["z_nonpositive"]["targetid"]])
    assert sample["has_wise"].all()
    assert not sample["spectype"].eq("STAR").any()
    p = ledger["extra"]
    assert p["presence_in_sample"]["has_z"] == int(sample["has_z"].sum())
    assert p["presence_fraction_in_sample"]["has_image"] == 1.0


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
    assert filters["has_image"]["dropped"] == 1
    assert ledger["counts"]["sample_rows"] == e["sample_rows"]
    assert sum(ledger["counts"][f"rows_{s}"] for s in manifest_split.SPLITS) == e["sample_rows"]
    assert ledger["extra"]["seed"] == 42 and ledger["extra"]["fractions"] == [0.8, 0.1, 0.1]
    assert list(manifest.columns) == manifest_split.MANIFEST_COLUMNS


def test_split_files_agree_and_cover_the_sample(run, planted):
    _, work, manifest, ledger = run
    split = pd.read_csv(work / manifest_split.SPLIT)
    assert list(split.columns) == ["targetid", "split"]
    sample = manifest[manifest["in_sample"]]
    assert len(split) == len(sample)
    merged = sample.merge(split, on="targetid", suffixes=("", "_file"))
    assert (merged["split"] == merged["split_file"]).all()
    assert set(split["split"]) <= set(manifest_split.SPLITS)
    expected = manifest_split.assign(np.array(planted["expected"]["sample_targetids"]), 42,
                                     [0.8, 0.1, 0.1])
    by_tid = dict(zip(planted["expected"]["sample_targetids"], expected))
    assert all(by_tid[t] == s for t, s in zip(split["targetid"], split["split"]))


def test_rerun_is_deterministic_and_cli_works(run):
    cfg, work, _, _ = run
    first = common.sha256(work / manifest_split.MANIFEST)
    assert manifest_split.main(["--config", cfg["_config_path"]]) == 0
    assert common.sha256(work / manifest_split.MANIFEST) == first


def test_missing_inputs(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    with pytest.raises(manifest_split.ManifestError, match="missing input"):
        manifest_split.run(cfg, **QUIET)
    crossmatch.run(cfg, **QUIET)
    fetch_spectra.run(cfg, **QUIET)
    with pytest.raises(manifest_split.ManifestError, match="cutout directory"):
        manifest_split.run(cfg, **QUIET)
    assert manifest_split.main(["--config", cfg["_config_path"]]) == 1
