# aion-flow_v2: data pipeline for the AION X-ray paper.
#
#   make test                       run the test suite on the committed fixtures
#   make <step> CONFIG=path.yaml    run one pipeline step on another config
#
# Steps are added one per cut; `all` chains them once they exist.

CONFIG ?= config.yaml
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.DEFAULT_GOAL := help
.PHONY: help test lint fixtures fetch clean-raw crossmatch labels spectra cutouts manifest_split

help:  ## list targets
	@awk -F':.*## ' '/^[a-zA-Z_-]+:.*## /{printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test:  ## run the test suite on the committed fixtures
	$(PY) -m pytest -q

lint:  ## static checks
	$(PY) -m ruff check .

fixtures:  ## regenerate tests/fixtures (deterministic; commit the result)
	$(PY) tests/fixtures/make_fixtures.py

# ---- pipeline steps ---------------------------------------------------------
# Pass DRY=1 to a step that supports it to report without acting.

fetch:  ## step 0: download the four catalogues into paths.raw (resume + checksums)
	$(PY) -m aionflow_data.fetch_catalogs --config $(CONFIG) $(if $(DRY),--dry-run,)

clean-raw:  ## delete paths.raw once the crossmatch and labels ledgers record its checksums
	$(PY) -m aionflow_data.fetch_catalogs --config $(CONFIG) --clean-raw

crossmatch:  ## step 1: NWAY x DESI 1" match with the selection rules -> work/crossmatch.parquet
	$(PY) -m aionflow_data.crossmatch --config $(CONFIG)

labels:  ## step 2: X-ray labels from the Main catalogue, host labels from CIGALE -> work/labels.csv
	$(PY) -m aionflow_data.labels --config $(CONFIG)

spectra:  ## step 3: fetch DESI coadd spectra into shards and merge -> work/spectra/source.h5 (LIMIT_GROUPS=N for a smoke)
	$(PY) -m aionflow_data.fetch_spectra --config $(CONFIG) $(if $(LIMIT_GROUPS),--limit-groups $(LIMIT_GROUPS),)

cutouts:  ## step 4: fetch Legacy Survey cutouts, one FITS per target -> work/cutouts/ (LIMIT=N for a smoke; ~8 days in full)
	$(PY) -m aionflow_data.fetch_cutouts --config $(CONFIG) $(if $(LIMIT),--limit $(LIMIT),)

manifest_split:  ## step 5: presence flags, the sample, component-grouped keyed split -> work/manifest.csv, work/split.csv
	$(PY) -m aionflow_data.manifest_split --config $(CONFIG)
