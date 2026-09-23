# aion-flow_v2

Code for *Probabilistic probes for galaxy evolution: signatures of AGN feedback in
the AION foundation model*.

We train probes on the frozen AION-1 encoder to predict X-ray and host-galaxy
properties of eROSITA sources from four inputs: DESI spectra (S), Legacy Survey
images (I), WISE photometry (W) and redshift (Z). A trained CLS token reads every
layer of the encoder, and normalizing-flow heads turn its final state into
posteriors. Photon counts enter through a Poisson likelihood, so a band with zero
counts is a measurement, not a missing value.

The repository has two packages. `aionflow_data` builds the sample from public
catalogues. `aionflow_model` tokenizes the inputs, trains the probes and scores
them. [docs/DATA.md](docs/DATA.md) describes the data and
[docs/MODEL.md](docs/MODEL.md) the model.

## Install

```sh
uv venv && uv pip install -e ".[dev]"     # the data pipeline
uv pip install -e ".[dev,model]"          # add the model
make test
```

The tests run in about a minute on synthetic fixtures committed to the repository.
They need no network and no pretrained weights: the model tests use a small
stand-in encoder built from AION's own transformer block. Two further tests check
the real encoder and codecs when `AIONFLOW_TEST_AION=1` is set.

## Building the data

```sh
make all
```

This runs nine steps in order: `fetch`, `crossmatch`, `labels`, `spectra`,
`cutouts`, `manifest_split`, `stage`, `validate` and `line_features`. Each can also
be run alone with `make <step>`. The fetchers skip files already on disk, so an
interrupted run can be restarted.

| step | wall time | disk |
|---|---|---|
| fetch | 1 to 3 h | 32 GB |
| crossmatch, labels | minutes | 150 MB |
| spectra | about half a day | 9 GB |
| cutouts | about 8 days | 55 GB |
| manifest_split, stage, validate | about an hour | 55 GB |
| line_features | a few hours | 100 MB |

The cutout fetch is rate-limited by the Legacy Survey service and dominates the
total. It can start as soon as `crossmatch` has run. Catalogue URLs, sizes and
checksums are pinned in `config.yaml`. Each step writes a ledger to
`data/provenance/` with the checksums of its inputs and every row it cut; the
ledgers from our run are committed.

## Training and scoring

This part needs a GPU.

```sh
make tokenize DEVICE=cuda
make train RUN=configs/marginals.yaml OUT=runs/marginals DEVICE=cuda
make train RUN=configs/rates.yaml     OUT=runs/rates     DEVICE=cuda
make train RUN=configs/joint4.yaml    OUT=runs/joint4    DEVICE=cuda
make baseline OUT=runs/baseline
make evaluate OUT=runs/marginals DEVICE=cuda
make evaluate OUT=runs/rates  NODES=48 DEVICE=cuda
make evaluate OUT=runs/joint4 NODES=48 DEVICE=cuda
make evaluate OUT=runs/baseline BASELINE=1
make analysis DEVICE=cuda CHUNK=64
make figures
```

`tokenize` runs AION's frozen codecs once and caches the 853 tokens per source.
The codecs take about half a second per source, far longer than the encoder pass,
so we run them once rather than every epoch.

The three runs differ only in their heads. `marginals` has four scalar heads and a
joint over (SFR, M⋆), `rates` a joint over the two band rates, and `joint4` a
joint over the two band rates, SFR and M⋆, from which we read the within-object
correlation. `rates` and `joint4` are scored with 48 quadrature nodes per axis
rather than the training default of 12, which is not converged for heads with
latent rates (see [docs/MODEL.md](docs/MODEL.md#the-count-likelihood)).

`joint4` can be unstable at the default learning rates. In our runs its training
loss rose after epoch 6 and it stopped after 12 epochs. With every learning rate in
`aionflow_model/config.py` halved it trained for 33 epochs, with its best at 27.

## Hardware

We trained on one NVIDIA H200. A batch is 896 sources of up to 853 tokens each,
too large for a single forward pass, so the trainer splits it into chunks and
accumulates the gradient. At the default `CHUNK=448`, training peaks at 95 GiB of
the card's 140 and an epoch takes about 15 minutes. On a smaller card, lower
`CHUNK`: it changes memory and speed but not the gradient. `analysis` draws 32,768
posterior samples per source and needs `CHUNK=64` on the same card. Tokenizing and
the baseline are much lighter.

On a machine without Python development headers, set `TORCH_DISABLE_NATIVE_JIT=1`
before training. Otherwise Triton fails to compile a kernel in the backward pass,
with a `gcc` error about a missing `Python.h`.

## Layout

```
config.yaml        URLs, checksums, constants and paths
configs/           run recipes: the three runs, the baseline, Appendix B
aionflow_data/     one module per data step
aionflow_model/    the probe, flows, objective, training and analysis
tests/             one test file per step; synthetic fixtures in tests/fixtures
docs/              DATA.md and MODEL.md
data/provenance/   one ledger per data step, from our run
```

## License

MIT, see `LICENSE`. To cite, see `CITATION.cff`.
