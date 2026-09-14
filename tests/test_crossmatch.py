"""Cut 3: the NWAY x DESI crossmatch reproduces every planted scenario by construction."""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from aionflow_data import common, crossmatch
from tests.conftest import FIXTURES, make_fx_cfg

QUIET = dict(log=lambda *a: None)


@pytest.fixture(scope="module")
def run(tmp_path_factory, planted):
    cfg = make_fx_cfg(tmp_path_factory.mktemp("crossmatch"))
    frame = crossmatch.run(cfg, **QUIET)
    ledger = common.read_ledger("crossmatch", cfg)
    return cfg, frame, ledger


def _row(frame: pd.DataFrame, detuid: str) -> pd.Series:
    rows = frame[frame["ero_detuid"] == detuid]
    assert len(rows) == 1, f"{detuid}: {len(rows)} rows"
    return rows.iloc[0]


# ----------------------------------------------------------------------------- unit

def test_decode_release_reads_the_targetid_bits():
    main = (9010 << 42) | (300010 << 22) | 1234
    assert crossmatch.decode_release(np.array([main]))[0] == 9010
    backup = (1 << 61) | 4_294_000_001
    assert crossmatch.decode_release(np.array([backup]))[0] == 0
    assert crossmatch.decode_release(np.array([(9012 << 42) | 5]))[0] == 9012


# ----------------------------------------------------------------------------- by construction

def test_output_matches_planted_rows_and_census(run, planted):
    _, frame, ledger = run
    e = planted["expected"]
    assert len(frame) == e["crossmatch_rows"]
    assert frame["spectype"].value_counts().to_dict() == e["crossmatch_census"]
    got = dict(zip(frame["ero_detuid"], frame["targetid"].astype(int)))
    assert got == e["crossmatch_rows_by_detuid"]
    assert list(frame.columns) == crossmatch.OUTPUT_COLUMNS
    assert ledger["counts"]["rows_out"] == e["crossmatch_rows"]
    assert ledger["extra"]["census"] == e["crossmatch_census"]


def test_nway_side_rules(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    filters = {f["filter"]: f for f in ledger["filters"]}
    assert filters["nway_exact_duplicates_collapsed"]["dropped"] == 1
    assert filters["nway_match_flag==1"]["dropped"] == 1
    assert filters["nway_one_primary_per_detuid"]["dropped"] == 1
    # the secondary candidate's target was never adopted
    assert S["secondary_flag2"]["rejected_targetid"] not in set(frame["targetid"])
    # the duplicated detection appears once
    assert (frame["ero_detuid"] == S["exact_duplicate"]["detuid"]).sum() == 1
    # the repeated DETUID kept the higher-p_i counterpart
    r = _row(frame, S["repeat_detuid_two_primaries"]["detuid"])
    assert r["targetid"] == S["repeat_detuid_two_primaries"]["kept_targetid"]
    assert S["repeat_detuid_two_primaries"]["dropped_targetid"] not in set(frame["targetid"])


def test_desi_side_rules(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    c = ledger["counts"]
    assert c["desi_rows_raw"] == planted["n_rows"]["zall_pix"]
    assert c["desi_primary_rows"] == c["desi_rows_raw"] - 1          # one non-primary row
    assert c["desi_primary_targetid_le0"] == 1                        # the sky fibre
    assert c["desi_rows_kept"] == c["desi_rows_raw"] - 2
    assert (frame["targetid"] == S["nonprimary_duplicate"]["targetid"]).sum() == 1
    assert S["sky_fibre_near_detection"]["detuid"] not in set(frame["ero_detuid"])
    for d in S["no_desi_within_radius"]["detuids"]:
        assert d not in set(frame["ero_detuid"])


def test_positional_match_and_main_survey_preference(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    r = _row(frame, S["tie_two_main"]["detuid"])
    assert r["targetid"] == S["tie_two_main"]["chosen_targetid"]
    assert S["tie_two_main"]["other_targetid"] not in set(frame["targetid"])
    r = _row(frame, S["tie_main_vs_backup"]["detuid"])
    assert r["targetid"] == S["tie_main_vs_backup"]["chosen_targetid"]
    assert S["tie_main_vs_backup"]["nearest_targetid"] not in set(frame["targetid"])
    r = _row(frame, S["backup_only"]["detuid"])
    assert r["targetid"] == S["backup_only"]["targetid"]
    m = ledger["extra"]["match"]
    assert m["flipped_by_main_survey_preference"] == planted["expected"]["n_flipped_by_preference"]
    assert m["nway_rows_with_2plus_candidates"] == 2
    assert m["matched_to_main_survey"] == m["nway_rows_matched"] - 1    # the backup-only target
    assert (frame["sep_arcsec"] <= 1.0).all()
    clean = frame[frame["ero_detuid"].isin(S["clean"]["detuids"])]
    assert (clean["sep_arcsec"] < 0.1).all()


def test_reliability_cut(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    r = _row(frame, S["uncalibrated_keep"]["detuid"])
    assert np.isnan(r["nway_threshold6"]) and r["nway_p_any"] >= 0.05
    assert S["uncalibrated_drop"]["detuid"] not in set(frame["ero_detuid"])
    assert S["below_threshold6"]["detuid"] not in set(frame["ero_detuid"])
    rel = ledger["extra"]["reliability"]
    assert rel["uncalibrated_kept"] == 1 and rel["uncalibrated_dropped"] == 1
    assert rel["calibrated_dropped"] == 1
    calibrated = frame[np.isfinite(frame["nway_threshold6"])]
    assert (calibrated["nway_p_any"] > calibrated["nway_threshold6"]).all()


def test_shared_targets(run, planted):
    _, frame, ledger = run
    S = planted["scenarios"]
    r = _row(frame, S["collision"]["kept_detuid"])
    assert r["targetid"] == S["collision"]["targetid"] and not r["split_source"]
    assert S["collision"]["dropped_detuid"] not in set(frame["ero_detuid"])
    pair = frame[frame["ero_detuid"].isin(S["split_source"]["detuids"])]
    assert len(pair) == 2 and pair["split_source"].all()
    assert (pair["targetid"] == S["split_source"]["targetid"]).all()
    assert sorted(pair["ero_detuid"]) == sorted(planted["expected"]["split_source_detuids"])
    sh = ledger["extra"]["shared_targets"]
    assert sh == {"shared_target_groups": 2, "split_source_groups": 1, "split_source_rows": 2,
                  "collision_groups": 1, "collision_rows_dropped": 1, "largest_group": 2,
                  "split_source_max_sep_arcsec": 15.0}
    # every other target is adopted by exactly one detection
    rest = frame[~frame["split_source"]]
    assert not rest["targetid"].duplicated().any()


def test_stars_leave_here(run, planted):
    """We neither train nor predict on stars, so they go at the crossmatch rather than
    being carried through labels, spectra and cutouts to be dropped at the end.

    The planted case is the one no other gate catches: a Galactic star whose redshift
    is perfectly good, ZWARN == 0, and about 1e-4 -- real, and not a distance. Pushed
    through a luminosity distance it would sit many sigma below the sample and drag the
    training standardizer with it, and no redshift-quality flag would object.
    """
    _, frame, ledger = run
    star = planted["scenarios"]["foreground_star"]
    assert star["spectype"] == "STAR" and star["zwarn"] == 0
    assert 0 < star["z"] < 0.001                              # a good redshift, not a distance
    assert star["targetid"] not in set(frame["targetid"])
    assert not frame["spectype"].eq("STAR").any()
    assert "STAR" not in ledger["extra"]["census"]
    filters = {f["filter"]: f for f in ledger["filters"]}
    assert filters["spectype_not_stellar"]["dropped"] == 2


def test_carried_columns_and_types(run, planted):
    _, frame, _ = run
    S = planted["scenarios"]
    r = frame[frame["targetid"] == S["w3_nonpositive"]["targetid"]].iloc[0]
    assert r["ls10_flux_w3"] == -5.0
    r = frame[frame["targetid"] == S["zwarn_nonzero"]["targetid"]].iloc[0]
    assert r["zwarn"] == 4
    r = frame[frame["targetid"] == S["z_nonpositive"]["targetid"]].iloc[0]
    assert r["z"] < 0
    assert frame["targetid"].dtype == np.int64
    assert frame["survey"].isin(["main"]).all()
    assert set(frame["program"]) <= {"dark", "bright", "backup"}
    assert frame["split_source"].dtype == bool


def test_ledger_inputs_and_determinism(run):
    cfg, frame, ledger = run
    assert set(ledger["inputs"]) == {"nway", "desi_zcat"}
    assert ledger["inputs"]["nway"]["sha256"] == common.sha256(FIXTURES / "nway.fits")
    out = common.ledger_path("crossmatch", cfg).parent.parent / "work" / crossmatch.OUTPUT
    first = common.sha256(out)
    again = crossmatch.run(cfg, **QUIET)
    assert common.sha256(out) == first
    pd.testing.assert_frame_equal(again, pd.read_parquet(out))


def test_gzipped_nway_is_inflated_once(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("zall_pix.fits", "main.fits", "cigale.fits"):
        shutil.copy(FIXTURES / name, raw / name)
    with open(FIXTURES / "nway.fits", "rb") as src, gzip.open(raw / "nway.fits.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    text = yaml.safe_load(open(cfg["_config_path"]))
    text["paths"]["raw"] = str(raw)
    text["inputs"]["nway"]["file"] = "nway.fits.gz"
    with open(cfg["_config_path"], "w") as fh:
        yaml.safe_dump(text, fh)
    cfg = common.load_config(cfg["_config_path"])
    frame = crossmatch.run(cfg, **QUIET)
    inflated = tmp_path / "work" / "nway.fits"
    assert inflated.is_file() and len(frame) > 0
    ledger = common.read_ledger("crossmatch", cfg)
    assert ledger["extra"]["nway_inflated"]["sha256"] == common.sha256(FIXTURES / "nway.fits")
    assert ledger["inputs"]["nway"]["path"].endswith("nway.fits.gz")
    mtime = inflated.stat().st_mtime_ns
    crossmatch.run(cfg, **QUIET)
    assert inflated.stat().st_mtime_ns == mtime


def test_missing_input_and_cli(tmp_path):
    cfg = make_fx_cfg(tmp_path)
    text = yaml.safe_load(open(cfg["_config_path"]))
    text["paths"]["raw"] = str(tmp_path / "nowhere")
    with open(cfg["_config_path"], "w") as fh:
        yaml.safe_dump(text, fh)
    cfg = common.load_config(cfg["_config_path"])
    with pytest.raises(crossmatch.CrossmatchError, match="missing input"):
        crossmatch.run(cfg, **QUIET)
    assert crossmatch.main(["--config", cfg["_config_path"]]) == 1
