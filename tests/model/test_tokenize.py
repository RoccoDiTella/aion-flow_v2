"""M3: AION's codecs run once per split, and the tokens land in the contract."""

from __future__ import annotations

import os
import pathlib

import h5py
import numpy as np
import pytest
import torch
from aion.modalities import (
    DESISpectrum,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyImage,
    Z,
)

from aionflow_data import common
from aionflow_model.data import ALL_TOKEN_KEYS, TOKEN_SIZES, TOKENS_FILE
from aionflow_model.tokenize import CODEC_REPO, TokenizeError, main, run

QUIET = dict(log=lambda *a, **k: None)
NEEDS_AION = pytest.mark.skipif(not os.environ.get("AIONFLOW_TEST_AION"),
                                reason="set AIONFLOW_TEST_AION=1 to run AION's real codecs")


class StubCodecs:
    """Stands in for AION's CodecManager: records the modalities, returns ids."""

    def __init__(self, sizes=None):
        self.sizes = sizes or TOKEN_SIZES
        self.seen = []

    def encode(self, *modalities):
        self.seen.append(modalities)
        # Ids that depend on the source alone, never on which block it arrived in.
        seed = (modalities[0].flux.double().sum(-1) * 1e3).to(torch.int64)
        out = {}
        for key, width in self.sizes.items():
            ids = (seed[:, None] + torch.arange(width)) % 1024
            out[key] = ids.squeeze(1) if width == 1 else ids     # scalars come back 1-D
        return out


@pytest.fixture(scope="module")
def tokenized(staged):
    """A run of the step over the fixture split, with the stub codecs."""
    cfg, work, staged_dir = staged
    codecs = StubCodecs()
    stats = run(cfg, splits=("train",), codecs=codecs, block=4, **QUIET)
    return cfg, staged_dir, work, codecs, stats


# ----------------------------------------------------------------------------- inputs

def test_the_step_hands_the_codecs_the_staged_inputs(tokenized, splits):
    _, _, _, codecs, _ = tokenized
    split = splits["train"]
    assert len(codecs.seen) == -(-split.n // 4)               # one call per block
    spectrum, image, w1, w2, w3, z = codecs.seen[0]
    assert isinstance(spectrum, DESISpectrum) and isinstance(image, LegacySurveyImage)
    assert [type(s) for s in (w1, w2, w3, z)] == [LegacySurveyFluxW1, LegacySurveyFluxW2,
                                                  LegacySurveyFluxW3, Z]
    with h5py.File(split.path, "r") as h:
        assert np.array_equal(spectrum.flux.numpy(), h["spectra"][:4])
        assert np.array_equal(spectrum.ivar.numpy(), h["spectra_ivar"][:4])
        # a pixel with no inverse variance is a pixel the codec must not normalise on
        assert np.array_equal(spectrum.mask.numpy(), h["spectra_ivar"][:4] <= 0)
        assert np.array_equal(spectrum.wavelength.numpy(),
                              np.tile(h["spectra_lambda"][:], (4, 1)))
        assert np.array_equal(image.flux.numpy(), h["image_flux"][:4])
        assert np.array_equal(z.value.numpy(), h["redshift"][:4])
        for i, scalar in enumerate((w1, w2, w3)):
            assert np.array_equal(scalar.value.numpy(), h[f"flux_w{i + 1}"][:4])
    assert image.bands == list(split.image_bands) == ["DES-G", "DES-R", "DES-I", "DES-Z"]


# ----------------------------------------------------------------------------- output

def test_the_tokens_file_is_row_aligned_with_its_split(tokenized, splits):
    _, staged_dir, _, _, stats = tokenized
    split = splits["train"]
    assert stats["train"]["rows"] == split.n
    with h5py.File(staged_dir / TOKENS_FILE.format(split="train"), "r") as h:
        assert set(h) == {"targetid", *ALL_TOKEN_KEYS}
        assert np.array_equal(h["targetid"][:], split.targetid)
        for key in ALL_TOKEN_KEYS:
            assert h[key].shape == (split.n, TOKEN_SIZES[key]) and h[key].dtype == np.int32
        assert h.attrs["codec_repo"] == CODEC_REPO and h.attrs["split"] == "train"
    assert sum(TOKEN_SIZES.values()) == 853
    assert split.tokens(0)["tok_spectrum_desi"].shape == (273,)


def test_blocking_does_not_change_the_tokens(tokenized):
    cfg, staged_dir, work, _, _ = tokenized
    first = common.sha256(staged_dir / TOKENS_FILE.format(split="train"))
    run(cfg, splits=("train",), codecs=StubCodecs(), block=1000, **QUIET)
    assert common.sha256(staged_dir / TOKENS_FILE.format(split="train")) == first


def test_the_ledger_records_the_codecs_and_the_counts(tokenized, splits):
    cfg, _, _, _, _ = tokenized
    ledger = common.read_ledger("tokenize", cfg)
    assert ledger["counts"]["rows_train"] == splits["train"].n
    assert ledger["extra"]["codec_repo"] == CODEC_REPO
    assert ledger["extra"]["tokens_per_source"] == dict(TOKEN_SIZES)


def test_a_codec_that_returns_the_wrong_shape_is_refused(tokenized):
    cfg, _, _, _, _ = tokenized
    short = dict(TOKEN_SIZES, tok_image=575)
    with pytest.raises(TokenizeError, match="575 tokens, expected 576"):
        run(cfg, splits=("train",), codecs=StubCodecs(short), **QUIET)
    missing = {k: v for k, v in TOKEN_SIZES.items() if k != "tok_z"}
    with pytest.raises(TokenizeError, match="no tok_z"):
        run(cfg, splits=("train",), codecs=StubCodecs(missing), **QUIET)


# ----------------------------------------------------------------------------- real codecs

def test_everything_handed_to_the_codecs_is_on_the_requested_device(splits):
    """The codecs move themselves to the device and not their inputs, so we must.

    On CPU this is vacuous, which is exactly why the omission survived a CPU-only
    smoke and only failed on the box: the first codec to index a buffer of its own
    against our data raised `boundaries is on cuda:0, other tensors on cpu`.
    """
    from aionflow_model.tokenize import modalities

    want = torch.device("cpu")
    for modality in modalities(splits["train"], 0, 2, "cpu"):
        tensors = [v for v in vars(modality).values() if isinstance(v, torch.Tensor)]
        assert tensors, type(modality).__name__
        assert all(t.device == want for t in tensors), type(modality).__name__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_codecs_run_on_a_gpu(splits):
    from aion.codecs import CodecManager

    from aionflow_model.tokenize import encode_block

    tokens = encode_block(CodecManager(device="cuda"), splits["train"], 0, 2, "cuda")
    for key in ALL_TOKEN_KEYS:
        assert tokens[key].shape == (2, TOKEN_SIZES[key])


@NEEDS_AION
def test_the_real_codecs_give_the_token_counts_the_encoder_expects():
    """AION's encoder asserts a fixed token count per modality, so ours must match.

    At the production cutout size, not the fixtures': the image codec centre-crops,
    so the 32-pixel fixture cutouts tokenize to 64 patches and AION would refuse them.
    """
    import yaml
    from aion.codecs import CodecManager

    cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())
    size, nbin = int(cfg["cutouts"]["size"]), int(cfg["spectra"]["nbin"])
    rows = 2
    torch.manual_seed(0)
    ivar = torch.rand(rows, nbin)
    lam0, dlam = float(cfg["spectra"]["lam0_angstrom"]), float(cfg["spectra"]["dlam_angstrom"])
    wavelength = lam0 + dlam * torch.arange(nbin, dtype=torch.float32)
    modalities = [
        DESISpectrum(flux=torch.rand(rows, nbin), ivar=ivar, mask=ivar <= 0,
                     wavelength=wavelength.unsqueeze(0).expand(rows, -1)),
        LegacySurveyImage(flux=torch.rand(rows, 4, size, size),
                          bands=["DES-G", "DES-R", "DES-I", "DES-Z"]),
        LegacySurveyFluxW1(value=torch.rand(rows)),
        LegacySurveyFluxW2(value=torch.rand(rows)),
        LegacySurveyFluxW3(value=torch.rand(rows)),
        Z(value=torch.rand(rows)),
    ]
    codecs = CodecManager(device="cpu")
    tokens = codecs.encode(*modalities)
    assert set(tokens) == set(ALL_TOKEN_KEYS)
    for key, ids in tokens.items():
        ids = ids if ids.dim() > 1 else ids[:, None]
        assert ids.shape == (rows, TOKEN_SIZES[key]), key
        assert int(ids.min()) >= 0
    assert sum(TOKEN_SIZES.values()) == 853
    # the codecs are deterministic, which is what makes caching them exact
    again = codecs.encode(*modalities)
    assert all(torch.equal(tokens[k], again[k]) for k in tokens)


# ----------------------------------------------------------------------------- cli

def test_the_cli_reports_a_missing_split(tmp_path, monkeypatch):
    import yaml
    cfg = {"paths": {"staged": str(tmp_path), "work": str(tmp_path),
                     "provenance": str(tmp_path)}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("aionflow_model.tokenize.load_config", lambda _p: cfg)
    monkeypatch.setattr("aionflow_model.tokenize.Split",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no staged split")))
    assert main(["--config", str(path), "--split", "train"]) == 1
