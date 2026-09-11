"""M9: the figures and Table 1 read what was computed, and recompute nothing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aionflow_model.figures import (
    RHO_BIN,
    TABLE,
    FigureError,
    figure1,
    figure2,
    figure3,
    gains,
    read,
)
from aionflow_model.figures import run as figures_run
from aionflow_model.objective import SUBSET_NAMES

QUIET = dict(log=lambda *a, **k: None)


def an_analysis() -> dict:
    table = []
    for head in ("flux", "lx", "sfr", "mstar"):
        for i, name in enumerate(SUBSET_NAMES):
            table.append({"head": head, "inputs": name, "n": 100,
                          "information_gain": 0.1 * i, "se": 0.01})
    z = np.linspace(0.1, 2.0, 50)
    modality = {m: {"pooled": 0.3 + 0.1 * j, "se": 0.02,
                    "vs_redshift": {"z": z.tolist(),
                                    "mean": (1.0 - 0.3 * z).tolist(),
                                    "lo": (0.9 - 0.3 * z).tolist(),
                                    "hi": (1.1 - 0.3 * z).tolist()}}
                for j, m in enumerate(("S", "I", "W"))}
    return {"table": table,
            "modality_gains": {"redshift_alone": 1.02, "n": 100,
                               "per_modality": modality,
                               "attributed_fraction": {"S": 0.4, "I": 0.35, "W": 0.25}},
            "rho": {"galaxies": {"n": 500, "fraction_negative": 0.74,
                                 "mean": -0.015, "median": -0.014}}}


def a_results() -> dict:
    rows = [{"head": head, "inputs": name, "n": 100, "information_gain": 0.2,
             f"r2_{head}": 0.512}
            for head in ("flux", "lx", "sfr", "mstar") for name in SUBSET_NAMES]
    return {"rows": rows, "common_subsample": {"flux": 12654, "lx": 12531,
                                               "sfr": 10259, "mstar": 11372}}


# ----------------------------------------------------------------------------- reading

def test_a_head_missing_a_combination_is_refused():
    analysis = an_analysis()
    analysis["table"] = [r for r in analysis["table"] if r["inputs"] != "ZSIW"]
    with pytest.raises(FigureError, match=r"no row for \['ZSIW'\]"):
        gains(analysis["table"], "flux")
    assert len(gains(an_analysis()["table"], "lx")) == 15


def test_a_null_standard_error_becomes_zero():
    rows = [{"head": "flux", "inputs": name, "information_gain": 1.0, "se": None}
            for name in SUBSET_NAMES]
    assert all(se == 0.0 for _, se in gains(rows, "flux").values())


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FigureError, match="missing"):
        read(tmp_path / "analysis.json")


# ----------------------------------------------------------------------------- drawing

def test_each_figure_is_written_as_pdf_and_png(tmp_path):
    analysis, results = an_analysis(), a_results()
    rng = np.random.default_rng(0)
    for path in (figure1(analysis, results, tmp_path),
                 figure2(analysis, tmp_path),
                 figure3(rng.normal(-0.015, 0.08, 500), analysis["rho"]["galaxies"],
                         tmp_path)):
        assert path.is_file() and path.stat().st_size > 1000
        assert path.with_suffix(".png").is_file()
    assert RHO_BIN == 0.005


def test_figure3_survives_a_single_source(tmp_path):
    path = figure3(np.array([0.1, np.nan]), {"n": 1}, tmp_path)
    assert path.is_file()


# ----------------------------------------------------------------------------- table 1

def test_table1_is_the_fifteen_by_four_grid(tmp_path):
    from aionflow_model.figures import table1
    text = table1(an_analysis(), a_results())
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    assert len(lines) == 1 + 15                       # the header plus a row per combination
    assert lines[1].startswith("| Z |") and lines[-1].startswith("| ZSIW |")
    assert "X-ray flux IG" in lines[0] and "log M* R2" in lines[0]
    assert "12,654" in text and "10,259" in text
    assert text.count("0.512") == 15 * 4              # every R2 cell


def test_the_run_writes_everything_it_was_given(tmp_path):
    analysis_dir, out = tmp_path / "a", tmp_path / "out"
    analysis_dir.mkdir()
    (analysis_dir / "analysis.json").write_text(json.dumps(an_analysis()))
    rng = np.random.default_rng(0)
    pd.DataFrame({"targetid": np.arange(200), "spectype": ["GALAXY"] * 150 + ["QSO"] * 50,
                  "redshift": rng.uniform(0.1, 2, 200),
                  "rho": rng.normal(-0.015, 0.08, 200)}).to_csv(analysis_dir / "rho.csv",
                                                                index=False)
    for name in ("marginals", "baseline"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps(a_results()))
    written = figures_run(analysis_dir, out, marginals=tmp_path / "marginals",
                          baseline=tmp_path / "baseline", **QUIET)
    assert [p.name for p in written] == ["figure1.pdf", "figure2.pdf", "figure3.pdf", TABLE]
    assert all(p.is_file() for p in written)
    # nothing is recomputed: the table's gains are the analysis file's, cell for cell
    text = (out / TABLE).read_text()
    assert "1.400" in text and "0.000" in text
