"""The star-formation cut: which galaxies' X-rays star formation could explain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aionflow_model.cleaning import (
    LEHMER16,
    MINEO14,
    MINEO14_BAND,
    SALPETER_PER_CHABRIER,
    K,
    mineo_log_lx,
    predicted_log_lx,
    sf_dominated,
    to_erosita,
)

GAL = dict(logmstar=10.5, log_sfr=1.0, z=0.3)


def test_the_coefficients_are_lehmer_2016_table_3():
    """The fit to the L10 sample plus CDF-S, Kroupa IMF. A typo here would move the
    cut without any other test noticing, so the numbers are pinned."""
    assert LEHMER16 == {(0.5, 2.0): (29.04, 3.78, 39.38, 0.99),
                        (2.0, 10.0): (29.37, 2.03, 39.28, 1.31)}
    assert K == 10.0


def test_the_band_conversion_is_a_gamma_2_power_law():
    assert to_erosita((0.5, 2.0)) == pytest.approx(math.log(11.5) / math.log(4))
    assert to_erosita((2.0, 10.0)) == pytest.approx(math.log(11.5) / math.log(5))


def test_the_prediction_is_the_mass_term_plus_the_sfr_term():
    want = math.log10(to_erosita((2.0, 10.0)) * (10 ** (29.37 + 10.0) + 10 ** 39.28))
    assert float(predicted_log_lx((2.0, 10.0), 10.0, 0.0, 0.0)) == pytest.approx(want)
    # and both terms grow with redshift
    assert predicted_log_lx((2.0, 10.0), 10.0, 0.0, 1.0) > predicted_log_lx((2.0, 10.0),
                                                                           10.0, 0.0, 0.0)


def test_a_galaxy_is_flagged_up_to_k_times_the_prediction_and_not_beyond():
    top = max([float(predicted_log_lx(b, **GAL)) for b in LEHMER16]
              + [float(mineo_log_lx(GAL["log_sfr"]))])
    edge = top + math.log10(K)
    got = sf_dominated([edge - 0.01, edge + 0.01], [GAL["logmstar"]] * 2, [GAL["log_sfr"]] * 2,
                       [GAL["z"]] * 2, ["GALAXY"] * 2)
    assert got.tolist() == [True, False]


def test_either_band_is_enough_to_flag():
    lo, hi = sorted(float(predicted_log_lx(b, **GAL)) for b in LEHMER16)
    assert hi > lo, "the two bands must disagree for this test to mean anything"
    # above the lower band's threshold, inside the higher band's
    between = (lo + hi) / 2 + math.log10(K)
    assert sf_dominated([between], [GAL["logmstar"]], [GAL["log_sfr"]], [GAL["z"]],
                        ["GALAXY"]).tolist() == [True]


def test_quasars_and_unclassifiable_sources_are_never_flagged():
    """A quasar's CIGALE SFR is inflated by the quasar; a source without M*, SFR or z
    cannot be classified. Neither is withheld, however faint."""
    m, s, z, faint = GAL["logmstar"], GAL["log_sfr"], GAL["z"], 38.0
    got = sf_dominated([faint] * 5, [m, m, np.nan, m, m], [s, s, s, np.nan, s],
                       [z, z, z, z, np.nan], ["QSO", "GALAXY", "GALAXY", "GALAXY", "GALAXY"])
    assert got.tolist() == [False, True, False, False, False]


def test_mineo_2014_is_pinned_and_converted_from_salpeter():
    """4.0e39 erg/s per Msun/yr in 0.5-8 keV for a Salpeter IMF; CIGALE's Chabrier
    SFR is raised by 1/0.63 before it is applied."""
    assert (MINEO14, MINEO14_BAND) == (4.0e39, (0.5, 8.0))
    assert SALPETER_PER_CHABRIER == pytest.approx(1 / 0.63)
    want = math.log10(math.log(11.5) / math.log(16) * 4.0e39 * 10.0 / 0.63)
    assert float(mineo_log_lx(1.0)) == pytest.approx(want)


def test_a_galaxy_can_be_flagged_by_mineo_alone():
    """Strongly star-forming, low mass, nearby: where Mineo's SFR-only relation
    predicts more than Lehmer's, it alone decides."""
    m, s, z = 8.0, 1.0, 0.0
    lehmer = max(float(predicted_log_lx(b, m, s, z)) for b in LEHMER16)
    mineo = float(mineo_log_lx(s))
    assert mineo > lehmer
    between = (lehmer + mineo) / 2 + math.log10(K)
    assert sf_dominated([between], [m], [s], [z], ["GALAXY"]).tolist() == [True]
