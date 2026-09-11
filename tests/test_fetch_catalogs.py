"""Cut 2: catalogue download with resume, checksum verification and the raw ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from aionflow_data import common, fetch_catalogs
from tests.conftest import FIXTURES
from tests.serve import serve

FAST = dict(retries=3, backoff_s=0.01, timeout_s=5.0, log=lambda *a: None)


@pytest.fixture
def remote(tmp_path):
    """Four small blobs on a local server, and a config that points at them."""
    served = tmp_path / "served"
    served.mkdir()
    blobs = {}
    for i, name in enumerate(common.INPUT_NAMES):
        data = bytes([(i * 37 + k) % 251 for k in range(20_000 + i * 1000)])
        fname = f"{name}.bin"
        (served / fname).write_bytes(data)
        blobs[name] = (fname, data)
    # one unrelated line, one './'-prefixed entry, one plain entry
    (served / "publisher.sha256sum").write_text(
        f"{'0' * 64}  IronPhysProp_AfterBurnerv1.2.fits\n"
        f"{hashlib.sha256(blobs['cigale'][1]).hexdigest()}  ./cigale.bin\n"
        f"{hashlib.sha256(blobs['desi_zcat'][1]).hexdigest()}  desi_zcat.bin\n")
    return served, blobs


def make_cfg(tmp_path: Path, base_url: str, blobs: dict, **overrides) -> dict:
    cfg = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    cfg["paths"] = {k: str(tmp_path / "data" / k) for k in common.PATH_KEYS}
    for name, (fname, data) in blobs.items():
        cfg["inputs"][name] = {"file": fname, "url": f"{base_url}/{fname}", "bytes": len(data),
                               "md5": hashlib.md5(data).hexdigest()}
    cfg["inputs"]["cigale"]["sha256"] = hashlib.sha256(blobs["cigale"][1]).hexdigest()
    cfg["inputs"]["cigale"]["checksum_sidecar_url"] = f"{base_url}/publisher.sha256sum"
    cfg["inputs"]["desi_zcat"]["checksum_sidecar_url"] = f"{base_url}/publisher.sha256sum"
    for name, values in overrides.items():
        cfg["inputs"][name].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return common.load_config(path)


def raw_path(cfg: dict, name: str) -> Path:
    return Path(cfg["paths"]["raw"]) / cfg["inputs"][name]["file"]


# ----------------------------------------------------------------------------- behaviour

def test_downloads_everything_and_writes_the_ledger(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        entries = fetch_catalogs.run(cfg, **FAST)
    for name, (fname, data) in blobs.items():
        assert raw_path(cfg, name).read_bytes() == data
        assert entries[name]["status"] == "downloaded"
        assert entries[name]["sha256"] == hashlib.sha256(data).hexdigest()
        assert entries[name]["md5"] == hashlib.md5(data).hexdigest()
    assert entries["cigale"]["publisher_sha256_verified"] is True
    assert entries["desi_zcat"]["publisher_sha256_verified"] is True
    assert entries["nway"]["publisher_sha256_verified"] is None
    ledger = json.loads(common.ledger_path("raw", cfg).read_text())
    assert ledger["step"] == "raw" and ledger["counts"] == {"files": 4}
    assert set(ledger["inputs"]) == set(common.INPUT_NAMES)
    assert ledger["inputs"]["main"]["retrieved_utc"].endswith("+00:00")
    gets = [r for r in server.requests if r[0] == "GET" and not r[1].endswith(".sha256sum")]
    assert len(gets) == 4 and all(r[2] is None for r in gets)


def test_complete_file_is_skipped_without_a_request(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        fetch_catalogs.run(cfg, **FAST)
        n_first = len(server.requests)
        entries = fetch_catalogs.run(cfg, **FAST)
    assert len(server.requests) == n_first
    assert all(e["status"] == "present" for e in entries.values())


def test_partial_file_resumes_with_a_range_request(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        fname, data = blobs["nway"]
        dest = raw_path(cfg, "nway")
        dest.parent.mkdir(parents=True)
        half = len(data) // 2
        dest.with_name(fname + ".part").write_bytes(data[:half])
        assert fetch_catalogs.status(dest, len(data)) == "partial"
        entries = fetch_catalogs.run(cfg, **FAST)
    ranged = [r for r in server.requests if r[1].endswith(fname)]
    assert ranged == [("GET", f"/{fname}", f"bytes={half}-")]
    assert dest.read_bytes() == data
    assert not dest.with_name(fname + ".part").exists()
    assert entries["nway"]["status"] == "downloaded"


def test_server_without_range_support_restarts_the_file(tmp_path, remote):
    served, blobs = remote
    with serve(served, support_range=False) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        fname, data = blobs["main"]
        dest = raw_path(cfg, "main")
        dest.parent.mkdir(parents=True)
        dest.with_name(fname + ".part").write_bytes(b"garbage" * 100)
        fetch_catalogs.run(cfg, **FAST)
    assert dest.read_bytes() == data


def test_wrong_checksum_raises_and_removes_the_file(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (_, url):
        cfg = make_cfg(tmp_path, url, blobs, nway={"md5": "0" * 32})
        with pytest.raises(fetch_catalogs.FetchError, match="md5"):
            fetch_catalogs.run(cfg, **FAST)
    assert not raw_path(cfg, "nway").exists()
    assert common.read_ledger("raw", cfg) is None


def test_wrong_size_in_config_is_reported(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (_, url):
        cfg = make_cfg(tmp_path, url, blobs, main={"bytes": 10})
        with pytest.raises(fetch_catalogs.FetchError, match="more than the configured 10"):
            fetch_catalogs.run(cfg, **FAST)
    assert not raw_path(cfg, "main").exists()
    assert not raw_path(cfg, "main").with_name("main.bin.part").exists()


def test_publisher_sidecar_mismatch_raises(tmp_path, remote):
    served, blobs = remote
    (served / "publisher.sha256sum").write_text(f"{'a' * 64}  cigale.bin\n")
    with serve(served) as (_, url):
        cfg = make_cfg(tmp_path, url, blobs)
        with pytest.raises(fetch_catalogs.FetchError, match="publisher"):
            fetch_catalogs.run(cfg, **FAST)
    assert not raw_path(cfg, "cigale").exists()


def test_transient_errors_are_retried(tmp_path, remote):
    served, blobs = remote
    with serve(served, fail_queue=[503, 429]) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        entries = fetch_catalogs.run(cfg, **FAST)
    assert entries["nway"]["status"] == "downloaded"          # the first file hit both failures
    assert len([r for r in server.requests if r[1].endswith("nway.bin")]) == 3


def test_permanent_404_fails_without_retrying(tmp_path, remote):
    served, blobs = remote
    (served / "main.bin").unlink()
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        with pytest.raises(fetch_catalogs.FetchError, match="404"):
            fetch_catalogs.run(cfg, **FAST)
    assert len([r for r in server.requests if r[1].endswith("main.bin")]) == 1


def test_dry_run_downloads_nothing(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (server, url):
        cfg = make_cfg(tmp_path, url, blobs)
        entries = fetch_catalogs.run(cfg, dry_run=True, **FAST)
    assert server.requests == []
    assert {e["status"] for e in entries.values()} == {"missing"}
    assert common.read_ledger("raw", cfg) is None
    assert not any(Path(cfg["paths"]["raw"]).glob("*"))


def test_cli_runs(tmp_path, remote):
    served, blobs = remote
    with serve(served) as (_, url):
        cfg = make_cfg(tmp_path, url, blobs)
        rc = fetch_catalogs.main(["--config", cfg["_config_path"], "--backoff", "0.01"])
    assert rc == 0 and raw_path(cfg, "nway").is_file()
