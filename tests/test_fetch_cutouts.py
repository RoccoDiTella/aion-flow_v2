"""Cut 6: Legacy Survey cutouts, sequential, atomic, resumable, validated before rename."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits

from aionflow_data import common, crossmatch, fetch_cutouts
from tests.conftest import FIXTURES, make_fx_cfg
from tests.serve import serve

QUIET = dict(log=lambda *a: None)
FAST = dict(sleep_s=0.0, backoff_s=0.01)


@pytest.fixture
def served(tmp_path, planted):
    """A server directory whose single file `cutout` answers every cutout URL."""
    d = tmp_path / "served"
    d.mkdir()
    tid = planted["cutout_targetids"][0]
    shutil.copy(FIXTURES / "cutouts" / f"{tid}.fits", d / "cutout")
    return d


def make_cfg(tmp_path, base_url, **cutout_overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = make_fx_cfg(tmp_path)
    text = yaml.safe_load(open(cfg["_config_path"]))
    text["archives"]["ls_cutout_url"] = (
        f"{base_url}/cutout?ra={{ra:.6f}}&dec={{dec:.6f}}&layer={{layer}}"
        f"&pixscale={{pixscale}}&size={{size}}&bands={{bands}}")
    text["cutouts"].update(cutout_overrides)
    with open(cfg["_config_path"], "w") as fh:
        yaml.safe_dump(text, fh)
    cfg = common.load_config(cfg["_config_path"])
    crossmatch.run(cfg, **QUIET)
    return cfg


def cutout_dir(tmp_path):
    return tmp_path / "work" / fetch_cutouts.CUTOUT_DIR


# ----------------------------------------------------------------------------- behaviour

def test_fetches_one_file_per_target_at_the_target_position(tmp_path, served, planted):
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url)
        stats = fetch_cutouts.run(cfg, **FAST, **QUIET)
    n_targets = len(set(planted["expected"]["crossmatch_rows_by_detuid"].values()))
    assert stats["targets"] == n_targets
    assert stats == {**stats, "fetched": n_targets, "failed": 0, "present_before": 0,
                     "present_after": n_targets}
    files = sorted(cutout_dir(tmp_path).glob("*.fits"))
    assert len(files) == n_targets and not list(cutout_dir(tmp_path).glob("*.tmp"))
    size = cfg["cutouts"]["size"]
    for path in files:
        assert fetch_cutouts.read_cutout(path, size).shape == (4, size, size)
    # the request carries the DESI target position and the cutout parameters
    gets = [r[1] for r in server.requests]
    assert len(gets) == n_targets
    frame = pd.read_parquet(tmp_path / "work" / crossmatch.OUTPUT).drop_duplicates("targetid")
    for row in frame.itertuples():
        assert any(f"ra={row.target_ra:.6f}&dec={row.target_dec:.6f}" in g for g in gets)
    assert all(f"layer=ls-dr10&pixscale=0.262&size={size}&bands=griz" in g for g in gets)
    ledger = common.read_ledger("cutouts", cfg)
    assert ledger["counts"]["fetched"] == n_targets and ledger["extra"]["failures"] == []
    assert ledger["extra"]["cutout"]["size"] == size


def test_rerun_skips_present_files(tmp_path, served):
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url)
        fetch_cutouts.run(cfg, **FAST, **QUIET)
        n = len(server.requests)
        stats = fetch_cutouts.run(cfg, **FAST, **QUIET)
    assert len(server.requests) == n
    assert stats["fetched"] == 0 and stats["present_before"] == stats["targets"]
    assert stats["to_fetch"] == 0


def test_limit_fetches_only_the_first_n(tmp_path, served):
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url)
        stats = fetch_cutouts.run(cfg, limit=3, **FAST, **QUIET)
    assert stats["fetched"] == 3 and len(server.requests) == 3
    assert len(list(cutout_dir(tmp_path).glob("*.fits"))) == 3
    assert common.read_ledger("cutouts", cfg)["extra"]["limit"] == 3


def test_non_image_response_is_rejected_and_counted(tmp_path, served):
    (served / "cutout").write_bytes(b"<html>rate limited</html>")
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url)
        stats = fetch_cutouts.run(cfg, limit=1, **FAST, **QUIET)
    assert stats["fetched"] == 0 and stats["failed"] == 1
    assert "not a FITS image" in stats["failures"][0][1]
    assert len(server.requests) == 5                     # every attempt was made
    assert list(cutout_dir(tmp_path).glob("*")) == []


def test_corrupt_response_of_plausible_size_is_rejected(tmp_path, served):
    (served / "cutout").write_bytes(b"\x00" * 30_000)
    with serve(served) as (_, url):
        cfg = make_cfg(tmp_path, url)
        stats = fetch_cutouts.run(cfg, limit=1, **FAST, **QUIET)
    assert stats["failed"] == 1 and list(cutout_dir(tmp_path).glob("*")) == []


def test_wrong_shape_is_rejected(tmp_path, served):
    with serve(served) as (_, url):
        cfg = make_cfg(tmp_path, url, size=64)           # server still returns 32 px
        stats = fetch_cutouts.run(cfg, limit=1, **FAST, **QUIET)
    assert stats["failed"] == 1 and "shape" in stats["failures"][0][1]


def test_rate_limit_backs_off_then_succeeds(tmp_path, served):
    with serve(served, fail_queue=[429, 503]) as (server, url):
        cfg = make_cfg(tmp_path, url)
        stats = fetch_cutouts.run(cfg, limit=1, **FAST, **QUIET)
    assert stats["fetched"] == 1 and stats["failed"] == 0
    assert len(server.requests) == 3


def test_404_is_a_permanent_failure_without_retries(tmp_path, served):
    (served / "cutout").unlink()
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url)
        stats = fetch_cutouts.run(cfg, limit=2, **FAST, **QUIET)
    assert stats["failed"] == 2 and len(server.requests) == 2
    assert all("HTTP 404" in f[1] for f in stats["failures"])
    ledger = common.read_ledger("cutouts", cfg)
    assert ledger["counts"]["failed"] == 2 and len(ledger["extra"]["failures"]) == 2


def test_read_cutout_validates(tmp_path, planted):
    tid = planted["cutout_targetids"][0]
    good = FIXTURES / "cutouts" / f"{tid}.fits"
    assert fetch_cutouts.read_cutout(good, 32).dtype == np.float32
    with pytest.raises(fetch_cutouts.BadCutout, match="shape"):
        fetch_cutouts.read_cutout(good, 160)
    with pytest.raises(fetch_cutouts.BadCutout, match="bands"):
        fetch_cutouts.read_cutout(good, 32, bands="grzy")
    with fits.open(good) as h:
        data, header = h[0].data.copy(), h[0].header.copy()
    data[1, 3, 3] = np.nan
    bad = tmp_path / "nan.fits"
    fits.PrimaryHDU(data, header).writeto(bad)
    with pytest.raises(fetch_cutouts.BadCutout, match="non-finite"):
        fetch_cutouts.read_cutout(bad, 32)
    text = tmp_path / "text.fits"
    text.write_bytes(b"not a fits file")
    with pytest.raises(fetch_cutouts.BadCutout, match="not a FITS image"):
        fetch_cutouts.read_cutout(text, 32)


def test_missing_crossmatch_and_cli(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    with pytest.raises(fetch_cutouts.CutoutError, match="missing input"):
        fetch_cutouts.run(cfg, **FAST, **QUIET)
    assert fetch_cutouts.main(["--config", cfg["_config_path"]]) == 1
