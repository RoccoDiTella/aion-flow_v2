"""Cut 11: `make all` on the fixture config runs every step end to end and validates."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from aionflow_data import common
from aionflow_data.manifest_split import SPLITS
from tests.conftest import FIXTURES
from tests.serve import serve

REPO = FIXTURES.parents[1]
STEPS = ("raw", "crossmatch", "labels", "spectra", "cutouts", "manifest_split", "stage",
         "validate", "line_features")


@pytest.fixture
def e2e_cfg(tmp_path, planted):
    """A config under tmp_path: copied raw catalogues, local coadds, a served cutout."""
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    for name in common.INPUT_NAMES:
        shutil.copy(FIXTURES / cfg["inputs"][name]["file"], raw)
    served = tmp_path / "served"
    served.mkdir()
    shutil.copy(FIXTURES / "cutouts" / f"{planted['cutout_targetids'][0]}.fits", served / "cutout")
    cfg["paths"] = {"raw": str(raw), "work": str(tmp_path / "work"),
                    "staged": str(tmp_path / "staged"),
                    "provenance": str(tmp_path / "provenance")}
    cfg["archives"]["desi_coadd_url"] = str(FIXTURES / "coadd" /
                                            "coadd-{survey}-{program}-{pix}.fits")
    cfg["cutouts"]["sleep_s"] = 0.0
    cfg["cutouts"]["backoff_s"] = 0.01
    cfg["spectra"]["workers"] = 2
    return tmp_path, served, cfg


def test_make_all_on_the_fixture_config(e2e_cfg, planted):
    tmp_path, served, cfg = e2e_cfg
    with serve(served) as (server, url):
        cfg["archives"]["ls_cutout_url"] = (
            f"{url}/cutout?ra={{ra:.6f}}&dec={{dec:.6f}}&layer={{layer}}"
            f"&pixscale={{pixscale}}&size={{size}}&bands={{bands}}")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result = subprocess.run(
            ["make", "all", f"CONFIG={config_path}", f"PY={sys.executable}", "NPROC=2"],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        n_cutout_requests = len(server.requests)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    loaded = common.load_config(config_path)
    for step in STEPS:
        ledger = common.read_ledger(step, loaded)
        assert ledger is not None, f"no ledger for {step}"
        assert ledger["step"] == step
    raw = common.read_ledger("raw", loaded)
    assert all(e["status"] == "downloaded" or e["status"] == "present"
               for e in raw["inputs"].values())
    verdict = common.read_ledger("validate", loaded)["extra"]
    assert verdict["passed"], verdict["failed"]

    e = planted["expected"]
    assert common.read_ledger("crossmatch", loaded)["counts"]["rows_out"] == e["crossmatch_rows"]
    assert common.read_ledger("manifest_split", loaded)["counts"]["sample_rows"] == e["sample_rows"]
    staged = json.loads((tmp_path / "staged" / "summary.json").read_text())
    assert sum(staged["splits"][s]["rows"] for s in SPLITS) == e["sample_rows"]
    features = pd.read_csv(tmp_path / "work" / "line_features.csv")
    assert len(features) == e["sample_rows"]
    cutouts = common.read_ledger("cutouts", loaded)["counts"]
    assert cutouts["fetched"] == cutouts["targets"] == n_cutout_requests
    assert "ALL PASSED" in result.stdout


def test_make_help_lists_every_step():
    result = subprocess.run(["make", "help"], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0
    for target in ("fetch", "crossmatch", "labels", "spectra", "cutouts", "manifest_split",
                   "stage", "validate", "line_features", "all", "test", "lint", "fixtures"):
        assert f"  {target} " in result.stdout, target
