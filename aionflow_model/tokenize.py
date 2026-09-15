"""Cut M3: run AION's frozen codecs once over the staged inputs.

    python -m aionflow_model.tokenize [--config CONFIG] [--split SPLIT] [--device D]

AION reads discrete tokens, and its codecs are frozen and their argmax is
deterministic, so a source's tokens are a function of the staged inputs alone.
Tokenizing inside the training loop is not an option: a source costs about half a
second of codec forward on CPU, two orders of magnitude more than the encoder
pass it feeds, and every epoch would pay it again. This step pays it once.

Per split, `<staged>/tokens_{split}.h5` holds the rows of `<staged>/{split}.h5`
in the same order:

    targetid            int64   (n,)
    tok_spectrum_desi   int32   (n, 273)
    tok_image           int32   (n, 576)
    tok_flux_w1..w3     int32   (n, 1)
    tok_z               int32   (n, 1)

A row whose redshift or WISE photometry is absent is still tokenized, because
the sampler never offers a modality the source does not have; the presence flags
on the staged file remain the authority. The ledger
`data/provenance/tokenize.json` records the counts and the codec repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from aionflow_data.common import describe_file, load_config, write_ledger
from aionflow_data.manifest_split import SPLITS

from .data import ALL_TOKEN_KEYS, TOKEN_SIZES, Split

STEP = "tokenize"
TOKENS_FILE = "tokens_{split}.h5"
CODEC_REPO = "polymathic-ai/aion-base"
BLOCK = 64


class TokenizeError(RuntimeError):
    pass


def modalities(split: Split, lo: int, hi: int, device: str = "cpu"):
    """AION modality objects for rows [lo, hi) of `split`, on `device`.

    The codecs move themselves to the device but not their inputs, so everything
    handed to them has to be there already. On CPU the omission is invisible; on a
    GPU the first codec that indexes a buffer of its own against our data raises.
    """
    from aion.modalities import (
        DESISpectrum,
        LegacySurveyFluxW1,
        LegacySurveyFluxW2,
        LegacySurveyFluxW3,
        LegacySurveyImage,
        Z,
    )
    flux, ivar = (t.to(device) for t in split.spectra(lo, hi))
    wavelength = torch.from_numpy(split.wavelength).to(device).expand(hi - lo, -1)
    wise = torch.from_numpy(split.wise[lo:hi].astype(np.float32)).to(device)
    return [
        DESISpectrum(flux=flux, ivar=ivar, mask=ivar <= 0, wavelength=wavelength),
        LegacySurveyImage(flux=split.images(lo, hi).to(device),
                          bands=list(split.image_bands)),
        LegacySurveyFluxW1(value=wise[:, 0]),
        LegacySurveyFluxW2(value=wise[:, 1]),
        LegacySurveyFluxW3(value=wise[:, 2]),
        Z(value=torch.from_numpy(split.redshift[lo:hi].astype(np.float32)).to(device)),
    ]


def encode_block(codecs, split: Split, lo: int, hi: int,
                 device: str = "cpu") -> dict[str, np.ndarray]:
    tokens = codecs.encode(*modalities(split, lo, hi, device))
    out = {}
    for key in ALL_TOKEN_KEYS:
        if key not in tokens:
            raise TokenizeError(f"the codecs returned no {key}")
        value = tokens[key]
        value = value if value.dim() > 1 else value[:, None]
        if value.shape[1] != TOKEN_SIZES[key]:
            raise TokenizeError(f"{key}: {value.shape[1]} tokens, expected "
                                f"{TOKEN_SIZES[key]}")
        out[key] = value.to(torch.int32).cpu().numpy()
    return out


def tokenize_split(split: Split, dest: Path, codecs, block: int = BLOCK,
                   device: str = "cpu", log=print) -> dict:
    tmp = dest.with_name(dest.name + ".part")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("targetid", data=split.targetid, track_times=False)
        sets = {key: h.create_dataset(key, shape=(split.n, TOKEN_SIZES[key]),
                                      dtype=np.int32, track_times=False)
                for key in ALL_TOKEN_KEYS}
        for lo in range(0, split.n, block):
            hi = min(lo + block, split.n)
            for key, value in encode_block(codecs, split, lo, hi, device).items():
                sets[key][lo:hi] = value
            log(f"[tokenize] {split.name}: {hi:,}/{split.n:,}", flush=True)
        h.attrs["codec_repo"] = CODEC_REPO
        h.attrs["split"] = split.name
    tmp.replace(dest)
    stats = {"rows": split.n, "bytes": dest.stat().st_size}
    log(f"[tokenize] {split.name}: {split.n:,} rows -> {dest.name} "
        f"({stats['bytes'] / 2 ** 20:.1f} MiB)")
    return stats


def run(cfg: dict, *, splits=SPLITS, device: str = "cpu", block: int = BLOCK,
        codecs=None, log=print) -> dict:
    """`codecs` defaults to AION's CodecManager; the tests pass a stub to keep the
    frozen weights out of the default suite."""
    if codecs is None:
        from aion.codecs import CodecManager
        codecs = CodecManager(device=device)
    staged, work = Path(cfg["paths"]["staged"]), Path(cfg["paths"]["work"])
    out = {}
    for name in splits:
        split = Split(staged, work, name)
        try:
            out[name] = tokenize_split(split, staged / TOKENS_FILE.format(split=name),
                                       codecs, block=block, device=device, log=log)
        finally:
            split.close()
    write_ledger(STEP, cfg,
                 inputs={name: staged / f"{name}.h5" for name in splits},
                 counts={f"rows_{k}": v["rows"] for k, v in out.items()},
                 extra={"codec_repo": CODEC_REPO, "device": device,
                        "tokens_per_source": dict(TOKEN_SIZES),
                        "outputs": {name: describe_file(staged / TOKENS_FILE.format(split=name))
                                    for name in splits}})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--split", action="append", choices=list(SPLITS),
                        help="tokenize one split; repeatable, default all")
    parser.add_argument("--device", default="cpu", help="torch device for the codecs")
    parser.add_argument("--block", type=int, default=BLOCK, help="rows per codec forward")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run(cfg, splits=tuple(args.split or SPLITS), device=args.device, block=args.block)
    except (TokenizeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
