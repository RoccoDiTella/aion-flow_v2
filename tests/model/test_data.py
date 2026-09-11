"""M0: the staged split joins its labels, and the batch carries the contract."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from aionflow_model.data import (
    ALL_TOKEN_KEYS,
    MODALITIES,
    RATE_TARGETS,
    SCALAR_TARGETS,
    TOKEN_SIZES,
    DataError,
    Split,
    Standardizer,
    TokenDataset,
    loader,
    log_plug_in_rate,
)

BATCH = {
    "targetid": torch.int64, "present": torch.bool, "y": torch.float32, "y_ok": torch.bool,
    "counts": torch.float64, "bkg": torch.float64, "expo": torch.float64,
    "rate_ok": torch.bool,
    **{key: torch.int32 for key in ALL_TOKEN_KEYS},
}


# ----------------------------------------------------------------------------- the join

def test_a_split_lines_its_labels_up_with_its_staged_rows(splits, staged):
    _, work, _ = staged
    labels = pd.read_csv(work / "labels.csv").set_index("ero_detuid")
    manifest = pd.read_csv(work / "manifest.csv")
    manifest = manifest[manifest["in_sample"]].set_index("targetid")
    total = 0
    for name, split in splits.items():
        total += split.n
        assert split.present.shape == (split.n, len(MODALITIES))
        # S and I are always present: the sample requires a spectrum and a cutout.
        assert split.present[:, 1].all() and split.present[:, 2].all()
        assert (split.present[:, 0] == manifest.loc[split.targetid, "has_z"]).all()
        assert (split.present[:, 3] == manifest.loc[split.targetid, "has_wise"]).all()
        assert (split.spectype == manifest.loc[split.targetid, "spectype"].to_numpy()).all()
        assert np.array_equal(split.detuid,
                              manifest.loc[split.targetid, "ero_detuid"].to_numpy(str))
        rows = labels.loc[split.detuid]
        for i, target in enumerate(SCALAR_TARGETS):
            want = rows[{"flux": "log_flux_1", "lx": "log_lx", "sfr": "log_sfr",
                         "mstar": "logmstar_cigale"}[target]].to_numpy(float)
            assert np.array_equal(split.y_raw[:, i], want, equal_nan=True)
            assert np.array_equal(split.y_ok[:, i], np.isfinite(want))
        for i, target in enumerate(RATE_TARGETS):
            band = target.split("_")[1]
            assert np.array_equal(split.counts[:, i], rows[f"ape_cts_{band}"].to_numpy(float))
            assert np.array_equal(split.expo[:, i], rows[f"ape_exp_{band}"].to_numpy(float))
    assert total == int(manifest["in_sample"].sum())


def test_a_zero_count_band_stays_a_measurement(splits):
    seen = 0
    for split in splits.values():
        zero = split.counts == 0
        seen += int(zero.sum())
        assert split.rate_ok[zero & (split.expo > 0)].all()
    assert seen > 0, "the fixtures should plant at least one zero-count band"


def test_a_missing_or_mismatched_join_is_an_error(splits, staged, tmp_path):
    _, work, staged_dir = staged
    with pytest.raises(DataError, match="unknown split"):
        Split(staged_dir, work, "holdout")
    with pytest.raises(DataError, match="missing staged split"):
        Split(tmp_path, work, "train")
    with pytest.raises(DataError, match="missing manifest"):
        Split(staged_dir, tmp_path, "train")
    short = tmp_path / "short"
    short.mkdir()
    shutil.copy(work / "labels.csv", short / "labels.csv")
    manifest = pd.read_csv(work / "manifest.csv")
    manifest[manifest["targetid"] != splits["train"].targetid[0]].to_csv(
        short / "manifest.csv", index=False)
    with pytest.raises(DataError, match="not in the manifest sample"):
        Split(staged_dir, short, "train")
    shutil.copy(work / "manifest.csv", short / "manifest.csv")
    labels = pd.read_csv(work / "labels.csv")
    labels[labels["ero_detuid"] != splits["train"].detuid[0]].to_csv(
        short / "labels.csv", index=False)
    with pytest.raises(DataError, match="no labels row"):
        Split(staged_dir, short, "train")


# ----------------------------------------------------------------------------- scaling

def test_the_standardizer_is_fitted_on_the_training_split_only(splits):
    train = splits["train"]
    std = Standardizer.fit(train)
    for i, name in enumerate(SCALAR_TARGETS):
        ok = train.y_ok[:, i]
        assert std.mean[name] == pytest.approx(train.y_raw[ok, i].mean())
        assert std.scale[name] == pytest.approx(train.y_raw[ok, i].std())
    for i, name in enumerate(RATE_TARGETS):
        ok = train.rate_ok[:, i]
        values = log_plug_in_rate(train.counts[ok, i], train.bkg[ok, i], train.expo[ok, i])
        assert std.mean[name] == pytest.approx(values.mean())
    with pytest.raises(DataError, match="fitted on the training split"):
        Standardizer.fit(splits["test"])


def test_standardized_targets_have_zero_mean_and_unit_variance_on_train(splits, standardizer):
    train = splits["train"]
    u = train.standardized(standardizer)
    for i, name in enumerate(SCALAR_TARGETS):
        ok = train.y_ok[:, i]
        assert u[ok, i].mean() == pytest.approx(0.0, abs=1e-9)
        assert u[ok, i].std() == pytest.approx(1.0, abs=1e-9)
        assert (u[~ok, i] == 0).all()
        back = standardizer.decode(name, u[ok, i])
        assert np.allclose(back, train.y_raw[ok, i])


def test_the_net_count_floor_keeps_the_log_finite():
    # N below B would give a negative rate; the half-photon floor keeps it defined.
    assert log_plug_in_rate([0.0], [5.0], [100.0]) == pytest.approx(np.log10(0.5 / 100.0))
    assert np.isfinite(log_plug_in_rate([3.0], [3.0], [100.0])).all()


def test_the_standardizer_round_trips_through_a_file(standardizer, tmp_path):
    path = tmp_path / "standardizer.json"
    standardizer.write(path)
    back = Standardizer.read(path)
    assert back.as_dict() == standardizer.as_dict()
    with pytest.raises(DataError, match="missing targets"):
        Standardizer({"flux": 0.0}, {"flux": 1.0})
    bad = standardizer.as_dict()
    bad["scale"]["flux"] = 0.0
    with pytest.raises(DataError, match="non-positive"):
        Standardizer.from_dict(bad)
    # as_dict hands out copies: the standardizer it came from is untouched.
    assert standardizer.scale["flux"] > 0


# ----------------------------------------------------------------------------- batches

def test_a_batch_carries_the_whole_registry_with_the_contract_dtypes(splits, standardizer):
    split = splits["train"]
    dataset = TokenDataset(split, standardizer)
    batch = next(iter(loader(dataset, batch_size=len(dataset), shuffle=False)))
    assert set(batch) == set(BATCH)
    for key, dtype in BATCH.items():
        assert batch[key].dtype == dtype, key
    n = split.n
    for key in ALL_TOKEN_KEYS:
        assert batch[key].shape == (n, TOKEN_SIZES[key]), key
    assert sum(TOKEN_SIZES.values()) == 853        # what AION reads for one source
    assert batch["y"].shape == (n, len(SCALAR_TARGETS))
    assert batch["counts"].shape == (n, len(RATE_TARGETS))
    assert np.array_equal(batch["targetid"].numpy(), split.targetid)
    assert np.array_equal(batch["y"].numpy(), split.standardized(standardizer).astype(np.float32))
    assert np.array_equal(batch["present"].numpy(), split.present)
    with h5py.File(split.tokens_path, "r") as h:
        assert np.array_equal(batch["targetid"].numpy(), h["targetid"][:])
        for key in ALL_TOKEN_KEYS:
            assert np.array_equal(batch[key].numpy(), h[key][:]), key
    # the spectra and images are still reachable, for the tokenizer alone
    with h5py.File(split.path, "r") as h:
        flux, ivar = split.spectra(0, n)
        assert np.array_equal(flux.numpy(), h["spectra"][:])
        assert np.array_equal(ivar.numpy(), h["spectra_ivar"][:])
        assert np.array_equal(split.images(0, n).numpy(), h["image_flux"][:])


def test_a_split_without_tokens_says_which_step_is_missing(splits, standardizer, staged,
                                                           tmp_path):
    _, work, staged_dir = staged
    import shutil
    for name in ("train.h5",):
        shutil.copy(staged_dir / name, tmp_path / name)
    for name in ("labels.csv", "manifest.csv"):
        shutil.copy(work / name, tmp_path / name)
    split = Split(tmp_path, tmp_path, "train")
    with pytest.raises(DataError, match="run aionflow_model.tokenize"):
        TokenDataset(split, standardizer)[0]
    split.close()


def test_shuffling_is_seeded_and_covers_every_row(splits, standardizer):
    dataset = TokenDataset(splits["train"], standardizer)
    order = [b["targetid"] for b in loader(dataset, 4, shuffle=True, seed=42)]
    again = [b["targetid"] for b in loader(dataset, 4, shuffle=True, seed=42)]
    other = [b["targetid"] for b in loader(dataset, 4, shuffle=True, seed=7)]
    assert all(torch.equal(a, b) for a, b in zip(order, again))
    assert not all(torch.equal(a, b) for a, b in zip(order, other))
    assert sorted(torch.cat(order).tolist()) == sorted(splits["train"].targetid.tolist())
