"""Shared test fixtures: the committed synthetic data and a config pointed at it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aionflow_data.common import load_config

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def make_fx_cfg(root: Path) -> dict:
    """The fixture config with work, staged and provenance under `root`.

    Raw inputs stay in `tests/fixtures`; the coadd URL template becomes an absolute
    local path so the spectra fetcher reads the fixture coadds.
    """
    cfg = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    cfg["paths"] = {"raw": str(FIXTURES), "work": str(root / "work"),
                    "staged": str(root / "staged"), "provenance": str(root / "provenance")}
    cfg["archives"]["desi_coadd_url"] = str(FIXTURES / "coadd" /
                                            "coadd-{survey}-{program}-{pix}.fits")
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return load_config(path)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def planted() -> dict:
    return json.loads((FIXTURES / "planted.json").read_text())


@pytest.fixture
def fx_cfg(tmp_path) -> dict:
    return make_fx_cfg(tmp_path)
