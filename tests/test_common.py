"""Cut 0: config loading, hashing, FITS column reads, ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from aionflow_data import common

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


# ----------------------------------------------------------------------------- config

def test_repo_config_loads_with_every_section():
    cfg = common.load_config(REPO_CONFIG)
    for section in common.REQUIRED_SECTIONS:
        assert section in cfg
    for name in common.INPUT_NAMES:
        entry = cfg["inputs"][name]
        assert entry["url"].startswith("https://")
        assert entry["bytes"] > 0
        assert entry["file"]
    assert cfg["crossmatch"]["radius_arcsec"] == 1.0
    assert cfg["spectra"]["nbin"] == 7781
    assert cfg["cutouts"]["size"] == 160
    assert sum(cfg["split"]["fractions"]) == pytest.approx(1.0)
    assert cfg["_config_path"] == str(REPO_CONFIG)


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    cfg_in = yaml.safe_load(REPO_CONFIG.read_text())
    cfg_in["paths"] = {"raw": "d/raw", "work": "d/work", "staged": "/abs/staged",
                       "provenance": "d/prov"}
    path = tmp_path / "sub" / "config.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(cfg_in))
    cfg = common.load_config(path)
    assert cfg["paths"]["raw"] == str((tmp_path / "sub" / "d" / "raw").resolve())
    assert cfg["paths"]["staged"] == "/abs/staged"


def test_config_env_var_is_honoured(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(REPO_CONFIG.read_text())
    monkeypatch.setenv("AIONFLOW_CONFIG", str(path))
    assert common.load_config()["_config_path"] == str(path)


def test_missing_section_is_an_error(tmp_path):
    cfg_in = yaml.safe_load(REPO_CONFIG.read_text())
    del cfg_in["split"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg_in))
    with pytest.raises(KeyError, match="split"):
        common.load_config(path)


# ----------------------------------------------------------------------------- hashing

def test_sha256_and_md5_match_hashlib(tmp_path):
    blob = bytes(range(256)) * 5000
    path = tmp_path / "blob.bin"
    path.write_bytes(blob)
    assert common.sha256(path) == hashlib.sha256(blob).hexdigest()
    assert common.md5(path) == hashlib.md5(blob).hexdigest()
    assert common.file_digests(path) == {"md5": hashlib.md5(blob).hexdigest(),
                                         "sha256": hashlib.sha256(blob).hexdigest()}


# ----------------------------------------------------------------------------- FITS

def test_native_converts_big_endian_and_leaves_native_alone():
    big = np.arange(4, dtype=">f8")
    out = common.native(big)
    assert out.dtype.byteorder in ("=", "<")
    assert np.array_equal(out, np.arange(4.0))
    nat = np.arange(4, dtype=np.int64)
    assert common.native(nat) is nat


@pytest.fixture
def small_table(tmp_path) -> Path:
    cols = [
        fits.Column(name="ID", format="K", array=np.arange(10, dtype=np.int64) * 7),
        fits.Column(name="RA", format="D", array=np.linspace(0, 9, 10)),
        fits.Column(name="NAME", format="8A", array=[f"src{i:02d}" for i in range(10)]),
        fits.Column(name="FLAG", format="L", array=np.arange(10) % 2 == 0),
    ]
    path = tmp_path / "table.fits"
    fits.BinTableHDU.from_columns(cols).writeto(path)
    return path


def test_read_fits_columns_returns_requested_columns_only(small_table):
    out = common.read_fits_columns(small_table, ["RA", "ID"])
    assert list(out) == ["RA", "ID"]
    assert out["ID"].dtype.byteorder in ("=", "<")
    assert np.array_equal(out["ID"], np.arange(10) * 7)
    assert common.fits_nrows(small_table) == 10
    assert common.fits_column_names(small_table) == ["ID", "RA", "NAME", "FLAG"]


def test_read_fits_columns_subsets_on_read(small_table):
    mask = np.arange(10) >= 7
    out = common.read_fits_columns(small_table, ["ID", "NAME", "FLAG"], rows=mask)
    assert np.array_equal(out["ID"], np.array([49, 56, 63]))
    assert [s.decode() if isinstance(s, bytes) else s for s in out["NAME"]] == \
        ["src07", "src08", "src09"]
    assert np.array_equal(out["FLAG"], np.array([False, True, False]))
    idx = np.array([2, 0])
    assert np.array_equal(common.read_fits_columns(small_table, ["ID"], rows=idx)["ID"],
                          np.array([14, 0]))


def test_read_fits_columns_names_absent_columns(small_table):
    with pytest.raises(KeyError, match="NOPE"):
        common.read_fits_columns(small_table, ["ID", "NOPE"])


# ----------------------------------------------------------------------------- ledgers

def test_filter_ledger_records_each_cut_in_order(capsys):
    led = common.FilterLedger(10)
    keep1 = led.apply("first", np.arange(10) < 8)
    assert keep1.sum() == 8 and led.n == 8
    led.apply("second", np.arange(8) % 2 == 0, note="evens")
    assert led.rows == [
        {"filter": "first", "kept": 8, "dropped": 2},
        {"filter": "second", "kept": 4, "dropped": 4, "note": "evens"},
    ]
    assert "[filter] first" in capsys.readouterr().out
    with pytest.raises(ValueError):
        led.apply("wrong shape", np.ones(3, bool))


def test_ledger_round_trip(tmp_path):
    cfg_in = yaml.safe_load(REPO_CONFIG.read_text())
    cfg_in["paths"] = {k: f"data/{k}" for k in common.PATH_KEYS}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_in))
    cfg = common.load_config(cfg_path)

    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    b = tmp_path / "b.bin"
    b.write_bytes(b"world!")

    assert common.read_ledger("demo", cfg) is None
    out = common.write_ledger(
        "demo", cfg,
        inputs={"a": a, "b": {"path": str(b), "bytes": 6, "sha256": "precomputed"}},
        counts={"rows_in": np.int64(10), "rows_out": 4},
        filters=[{"filter": "f", "kept": 4, "dropped": 6}],
        extra={"census": {"QSO": np.int32(3)}, "ratio": np.float32(0.5)},
    )
    assert out == Path(cfg["paths"]["provenance"]) / "demo.json"
    rec = json.loads(out.read_text())
    assert rec["step"] == "demo"
    assert rec["inputs"]["a"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert rec["inputs"]["a"]["bytes"] == 5
    assert rec["inputs"]["b"]["sha256"] == "precomputed"
    assert rec["counts"] == {"rows_in": 10, "rows_out": 4}
    assert rec["filters"][0]["dropped"] == 6
    assert rec["extra"]["census"]["QSO"] == 3
    assert rec["config"]["sha256"] == common.sha256(cfg_path)
    assert rec["written_utc"].endswith("+00:00")
    assert common.read_ledger("demo", cfg) == rec
