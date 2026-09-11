# aion-flow_v2

Data pipeline for *Probabilistic probes for galaxy evolution: signatures of AGN
feedback in the AION foundation model*. From four public catalogues and two
public archives it builds the training inputs, the label table, the
train/validation/test split and the emission-line baseline features that the
paper's models consume. Every step is deterministic, resumable, and writes a
provenance ledger with the checksums of what it read, the counts of what it
wrote, and every row it cut.

## What it produces

| output | contents |
|---|---|
| `data/staged/{train,val,test}.h5` | inputs only: DESI spectra and inverse variance on the 7,781-bin grid, redshift, WISE W1-W3 fluxes, Legacy Survey griz cutouts, and the redshift and WISE presence flags |
| `data/work/labels.csv` | one row per source: eROSITA band fluxes with split-normal errors, luminosity, detection likelihoods, aperture photon counts, CIGALE stellar mass and star formation rate |
| `data/work/split.csv` | `targetid, split` |
| `data/work/line_features.csv` | [O III] 5007, [Ne V] 3426, H-alpha and H-beta fluxes fitted on our own spectra, for the classical baseline |
| `data/provenance/<step>.json` | one ledger per step |

Column definitions, selection rules and the counts of the canonical run are in
[docs/DATA.md](docs/DATA.md).

## Inputs

| catalogue or archive | release | size |
|---|---|---|
| SRG/eROSITA-DE NWAY counterparts to Legacy Survey DR10 | DR2 (eRASS:3), `eRASSc3_Main_LS10_Public_27Jul2026` | 1.05 GB |
| SRG/eROSITA-DE Main catalogue | DR2, `eRASS3_Main_v1.3` | 2.14 GB |
| DESI redshift catalogue | DR1 (iron), `zall-pix-iron` | 21.3 GB |
| DESI CIGALE physical properties VAC | DR1, `IronPhysProp_v1.2` | 7.32 GB |
| DESI healpix coadd spectra | DR1 (iron) | ~13 GB read by HTTP range, only our rows |
| Legacy Survey DR10 cutouts | `ls-dr10`, 160 px at 0.262"/px, griz | ~55 GB, one file per target |

URLs, sizes and checksums are pinned in `config.yaml`; the fetch step verifies
them and, where the publisher ships one, the publisher's checksum sidecar.

## Run

```sh
uv venv && uv pip install -e ".[dev]"      # or python -m venv .venv && .venv/bin/pip install -e ".[dev]"
make test                                  # the suite runs on committed synthetic fixtures; no network
make all                                   # every step in order on config.yaml
make <step> CONFIG=other.yaml              # one step on another config
make help
```

The steps, in order: `fetch`, `crossmatch`, `labels`, `spectra`, `cutouts`,
`manifest_split`, `stage`, `validate`, `line_features`. Each is idempotent:
the fetchers resume from what is on disk, the others recompute from their
inputs.

| step | where it runs | wall time | disk |
|---|---|---|---|
| fetch | anywhere with 35 GB free | 1 to 3 h | 32 GB |
| crossmatch | a machine that can hold two 200 MB columns | minutes | 50 MB |
| labels | same | minutes | 100 MB |
| spectra | outbound HTTP; 6 workers | about half a day | ~9 GB |
| cutouts | outbound HTTP; sequential, rate limited by the service | about eight days | ~55 GB |
| manifest_split, stage, validate | same | about an hour | ~55 GB |
| line_features | all cores | hours | 100 MB |

The cutout fetch is the long pole and can start as soon as the crossmatch
exists; it is safe to interrupt and rerun.

## The sample in one paragraph

DESI DR1 primary targets are matched to the LS10 positions of the eROSITA DR2
NWAY counterparts within 1 arcsecond, preferring a main-survey TARGETID where
several fall inside the radius. Only NWAY primary counterparts are used, exact
duplicate rows are collapsed, and a detection keeps its highest-`p_i` row. The
reliability cut is NWAY's own per-tile threshold, `p_any > threshold6`, with a
flat `p_any >= 0.05` where the calibration is absent. A target adopted by two
detections is a split source when the X-ray positions lie within 15
arcseconds (both rows excluded) and a collision otherwise (the higher
`dist_post` wins). The sample is what remains with a spectrum and a cutout.
The split is a seeded random permutation of the sample (seed 42) cut at
80/10/10. Detection likelihood, redshift quality and WISE availability are
carried as label gates and presence flags, never as sample cuts.

## Layout

```
config.yaml            every URL, checksum, constant and path
Makefile               one target per step; `all`, `test`, `lint`, `fixtures`
aionflow_data/         one module per step, plus common.py and linefit.py
tests/                 one test file per step; tests/fixtures holds synthetic
                       catalogues, coadds and cutouts with every edge case planted
docs/DATA.md           the data contract
data/provenance/       committed ledgers of the canonical run
```

## Tests

`make test` runs about 110 tests in half a minute on the committed fixtures.
`tests/fixtures/make_fixtures.py` generates them deterministically and records
in `planted.json` what each step must produce, so the tests assert against
construction rather than against a previous run. The last test runs `make all`
end to end on that fixture config.

## License

MIT. See `LICENSE` and `CITATION.cff`.
