# aion-flow_v2: data pipeline for the AION X-ray paper.
#
#   make test                       run the test suite on the committed fixtures
#   make <step> CONFIG=path.yaml    run one pipeline step on another config
#
# Steps are added one per cut; `all` chains them once they exist.

CONFIG ?= config.yaml
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.DEFAULT_GOAL := help
.PHONY: help test lint fixtures all fetch crossmatch labels spectra cutouts manifest_split stage validate line_features tokenize train

help:  ## list targets
	@awk -F':.*## ' '/^[a-zA-Z_-]+:.*## /{printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test:  ## run the test suite on the committed fixtures
	$(PY) -m pytest -q

lint:  ## static checks
	$(PY) -m ruff check .

fixtures:  ## regenerate tests/fixtures (deterministic; commit the result)
	$(PY) tests/fixtures/make_fixtures.py

# ---- pipeline steps ---------------------------------------------------------
# Pass DRY=1 to a step that supports it to report without acting. Every step is
# idempotent: the fetchers resume, the others recompute from their inputs.

all: fetch crossmatch labels spectra cutouts manifest_split stage validate line_features  ## run every step in order

fetch:  ## step 0: download the four catalogues into paths.raw (resume + checksums)
	$(PY) -m aionflow_data.fetch_catalogs --config $(CONFIG) $(if $(DRY),--dry-run,)

crossmatch:  ## step 1: NWAY x DESI 1" match with the selection rules -> work/crossmatch.parquet
	$(PY) -m aionflow_data.crossmatch --config $(CONFIG)

labels:  ## step 2: X-ray labels from the Main catalogue, host labels from CIGALE -> work/labels.csv
	$(PY) -m aionflow_data.labels --config $(CONFIG)

spectra:  ## step 3: fetch DESI coadd spectra into shards and merge -> work/spectra/source.h5 (LIMIT_GROUPS=N for a smoke)
	$(PY) -m aionflow_data.fetch_spectra --config $(CONFIG) $(if $(LIMIT_GROUPS),--limit-groups $(LIMIT_GROUPS),)

cutouts:  ## step 4: fetch Legacy Survey cutouts, one FITS per target -> work/cutouts/ (LIMIT=N for a smoke; ~8 days in full)
	$(PY) -m aionflow_data.fetch_cutouts --config $(CONFIG) $(if $(LIMIT),--limit $(LIMIT),)

manifest_split:  ## step 5: presence flags, the sample, the seeded permutation split -> work/manifest.csv, work/split.csv
	$(PY) -m aionflow_data.manifest_split --config $(CONFIG)

stage:  ## step 6: inputs-only per-split HDF5, row-chunked -> staged/{train,val,test}.h5
	$(PY) -m aionflow_data.stage --config $(CONFIG)

validate:  ## step 7: check the staged files, split and labels; non-zero exit on any failure
	$(PY) -m aionflow_data.validate --config $(CONFIG)

line_features:  ## step 8: the baseline's four line fluxes on our own spectra -> work/line_features.csv
	$(PY) -m aionflow_data.line_features --config $(CONFIG) $(if $(NPROC),--nproc $(NPROC),)

# ---- model -----------------------------------------------------------------
# Needs the [model] extra. Run `all` first; these read what it staged.

tokenize:  ## run AION's frozen codecs once per split -> staged/tokens_{train,val,test}.h5 (DEVICE=cuda)
	$(PY) -m aionflow_model.tokenize --config $(CONFIG) $(if $(DEVICE),--device $(DEVICE),)

train:  ## train one run: make train RUN=configs/joint4.yaml OUT=runs/joint4 [DEVICE=cuda CHUNK=448]
	$(PY) -m aionflow_model.train --config $(CONFIG) --run $(RUN) --out $(OUT) \
		$(if $(DEVICE),--device $(DEVICE),) $(if $(CHUNK),--chunk $(CHUNK),)
