"""M8: composite targets, the within-object correlation, and the bootstrap."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from aionflow_model.analysis import (
    ANALYSIS,
    MASS_SPAN,
    RHO,
    WINDOW,
    bootstrap_se,
    combination_table,
    decode_draws,
    hardness_ratio,
    log_ssfr_independent,
    log_ssfr_joint,
    modality_gains,
    rho_from_draws,
    rolling_mean,
)
from aionflow_model.analysis import run as analysis_run
from aionflow_model.data import Standardizer
from aionflow_model.objective import SUBSET_NAMES
from tests.model.fake import FakeBackbone

QUIET = dict(log=lambda *a, **k: None)
TARGETS = ("flux", "lx", "sfr", "mstar", "rate_p2", "rate_p3")


def a_backbone():
    torch.manual_seed(0)
    return FakeBackbone(width=96, heads=4, depth=2)


def a_standardizer(**scale) -> Standardizer:
    return Standardizer({k: 0.0 for k in TARGETS},
                        {k: float(scale.get(k, 1.0)) for k in TARGETS})


class Normal:
    """A standard normal whatever the context, of whatever width it is handed."""

    def log_prob(self, u, context):
        return (-0.5 * u.square() - 0.5 * math.log(2 * math.pi)).sum(-1)


# ----------------------------------------------------------------------------- statistics

def test_the_bootstrap_recovers_the_standard_error_of_the_mean():
    values = np.random.default_rng(0).normal(size=4000)
    assert bootstrap_se(values, replicates=400, seed=1) == pytest.approx(
        values.std(ddof=1) / math.sqrt(values.size), rel=0.08)
    assert bootstrap_se(values, replicates=200, seed=3) == bootstrap_se(values, 200, 3)
    assert bootstrap_se(values, 200, 3) != bootstrap_se(values, 200, 4)
    assert math.isnan(bootstrap_se(np.array([1.0])))


def test_the_rolling_mean_follows_a_trend_with_a_band_around_it():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2, 3000)
    y = 1.0 - 0.4 * x + rng.normal(scale=0.05, size=x.size)
    got = rolling_mean(x, y, window=WINDOW, replicates=100, seed=0)
    assert np.all(np.diff(got["z"]) >= 0) and got["z"].size == x.size
    inside = (got["z"] > 0.2) & (got["z"] < 1.8)
    assert np.abs(got["mean"][inside] - (1.0 - 0.4 * got["z"][inside])).max() < 0.03
    assert (got["lo"] <= got["mean"]).all() and (got["mean"] <= got["hi"]).all()
    assert WINDOW == 0.08


def test_the_tables_read_the_dump(tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"redshift": rng.uniform(0.1, 2.0, n),
                          "prior_lx": np.zeros(n), "common_lx": np.ones(n, bool)})
    for name in SUBSET_NAMES:
        frame[f"ll_lx_{name}"] = float(len(name))
    rows = combination_table(frame, ["lx"])
    assert len(rows) == 15
    assert {r["inputs"]: round(r["information_gain"], 9) for r in rows} == {
        name: float(len(name)) for name in SUBSET_NAMES}
    assert all(r["se"] == pytest.approx(0.0, abs=1e-12) for r in rows)
    gains = modality_gains(frame, "lx")
    assert gains["redshift_alone"] == pytest.approx(1.0)
    assert {k: v["pooled"] for k, v in gains["per_modality"].items()} == {
        "S": 1.0, "I": 1.0, "W": 1.0}
    assert sum(gains["attributed_fraction"].values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- composites

def test_the_hardness_ratio_is_the_ratio_of_the_two_rates():
    rates = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1e-9, 1.0]])
    got = hardness_ratio(rates)
    assert got[0] == 0.0 and got[1] == pytest.approx(0.5) and got[2] == pytest.approx(-0.5)
    assert got[3] == pytest.approx(1.0, abs=1e-8)
    assert ((got >= -1) & (got <= 1)).all()


def test_draws_decode_to_rates_and_natural_units():
    standardizer = a_standardizer(sfr=0.4, rate_p2=0.5)
    draws = np.zeros((2, 3, 4))
    draws[..., 0] = 2.0        # rate_p2, standardized log10
    draws[..., 2] = 1.0        # sfr
    natural = decode_draws(draws, ("rate_p2", "rate_p3", "sfr", "mstar"), standardizer)
    assert np.allclose(natural["rate_p2"], 10.0**1.0)
    assert np.allclose(natural["sfr"], 0.4)
    assert np.allclose(natural["rate_p3"], 1.0)


def test_rho_is_the_within_object_pearson_correlation():
    targets = ("rate_p2", "rate_p3", "sfr", "mstar")
    standardizer = a_standardizer()
    n = 500
    rng = np.random.default_rng(0)
    draws = np.zeros((3, n, 4))
    hard = rng.normal(size=n)
    draws[0, :, 1] = hard                       # source 0: harder goes with lower sSFR
    draws[0, :, 2] = -hard
    draws[1, :, 1] = hard                       # source 1: harder goes with higher sSFR
    draws[1, :, 2] = hard
    draws[2, :, 1] = hard                       # source 2: sSFR is independent of it
    draws[2, :, 2] = rng.normal(size=n)
    rho = rho_from_draws(draws, targets, standardizer)
    # not quite -1 and +1: the hardness ratio is a nonlinear function of the rates,
    # so a perfectly anti-correlated pair of logs falls a little short
    assert rho[0] < -0.94 and rho[1] > 0.94 and abs(rho[2]) < 0.15
    flat = np.zeros((1, n, 4))
    assert np.isnan(rho_from_draws(flat, targets, standardizer)).all()


def test_the_ssfr_density_is_the_change_of_variables(caplog):
    """s = log SFR - log M* under a standard normal pair is normal with the quadrature
    sum of the two scales, which the integral has to reproduce."""
    standardizer = a_standardizer(sfr=0.4, mstar=0.3)
    s = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    context = torch.zeros(s.numel(), 1, dtype=torch.float64)
    got = log_ssfr_joint(Normal(), context, s, standardizer)
    sigma = math.hypot(0.4, 0.3)
    want = -0.5 * (s / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    assert torch.allclose(got, want, atol=1e-10)
    grid = torch.linspace(-4, 4, 4001, dtype=torch.float64)
    density = log_ssfr_joint(Normal(), torch.zeros(grid.numel(), 1, dtype=torch.float64),
                             grid, standardizer).exp()
    assert float(torch.trapezoid(density, grid)) == pytest.approx(1.0, abs=1e-9)
    assert MASS_SPAN > 5.0                 # five would cost the tails of log sSFR


def test_a_joint_that_is_the_product_of_its_marginals_gains_nothing():
    """The 0.24 nats is the joint against the two heads taken as independent, so a
    joint that already factorises has to score exactly the same."""
    standardizer = a_standardizer(sfr=0.4, mstar=0.3)
    s = torch.tensor([-1.0, 0.0, 0.7], dtype=torch.float64)
    context = torch.zeros(s.numel(), 1, dtype=torch.float64)
    together = log_ssfr_joint(Normal(), context, s, standardizer)
    apart = log_ssfr_independent((Normal(), Normal()), (context, context), s, standardizer)
    assert torch.allclose(together, apart, atol=1e-12)


# ----------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def analysed(staged, tmp_path_factory):
    from aionflow_model.evaluate import run as evaluate_run
    from aionflow_model.train import run as train_run
    cfg, _, _ = staged
    root = tmp_path_factory.mktemp("analysis")
    for name in ("marginals", "rates", "joint4"):
        train_run(cfg, f"configs/{name}.yaml", root / name, chunk=8, max_epochs=1,
                  backbone=a_backbone(), **QUIET)
    evaluate_run(cfg, root / "marginals", chunk=8, draws=8, backbone=a_backbone(), **QUIET)
    results = analysis_run(cfg, root / "out", marginals=root / "marginals",
                           rates=root / "rates", joint4=root / "joint4",
                           chunk=8, draws=64, backbone=a_backbone(), **QUIET)
    return root, results


def test_the_analysis_writes_what_the_figures_read(analysed, splits):
    root, results = analysed
    assert set(results) == {"table", "modality_gains", "ssfr", "hardness", "rho"}
    assert len(results["table"]) == 15 * 4
    assert set(results["modality_gains"]["per_modality"]) == {"S", "I", "W"}
    assert results["ssfr"]["n"] >= 0 and "gain_nats" in results["ssfr"]
    written = json.loads((root / "out" / ANALYSIS).read_text())
    assert [r["inputs"] for r in written["table"]] == [r["inputs"] for r in results["table"]]
    assert [r["information_gain"] for r in written["table"]] == pytest.approx(
        [r["information_gain"] for r in results["table"]])
    # a standard error over a single source is NaN, and the file must stay valid JSON
    assert "NaN" not in (root / "out" / ANALYSIS).read_text()

    rho = pd.read_csv(root / "out" / RHO)
    assert list(rho.columns) == ["targetid", "spectype", "redshift", "rho", "rho_zs_only"]
    assert np.array_equal(rho["targetid"].to_numpy(), splits["test"].targetid)
    inside = rho[["rho", "rho_zs_only"]].to_numpy()
    assert np.all((inside[np.isfinite(inside)] >= -1) & (inside[np.isfinite(inside)] <= 1))
    assert results["rho"]["draws"] == 64 and results["rho"]["nearby_below_z"] == 0.7
    assert "withheld_photometry" in results["rho"]

    hardness = pd.read_csv(root / "out" / HARDNESS_COLUMNS[0])
    assert list(hardness.columns) == HARDNESS_COLUMNS[1]
    assert (hardness["hr_lo"] <= hardness["hr_median"]).all()
    assert (hardness["hr_median"] <= hardness["hr_hi"]).all()


HARDNESS_COLUMNS = ("hardness.csv",
                    ["targetid", "spectype", "redshift", "hr_median", "hr_lo", "hr_hi"])


def test_at_least_one_run_directory_is_required():
    from aionflow_model.analysis import main
    assert main(["--out", "x"]) == 1
