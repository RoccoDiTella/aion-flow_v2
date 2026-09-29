"""Galaxies whose X-rays star formation could explain.

X-ray binaries and hot gas emit X-rays in proportion to a galaxy's stellar mass and
star formation rate. Where that emission could account for the observed luminosity,
the X-ray labels do not measure the AGN, and a run can withhold them from training.

Two calibrations predict that emission. Lehmer et al. (2016, ApJ 825, 7), Table 3,
fit the local sample of Lehmer et al. (2010) plus the Chandra Deep Field-South, for
a Kroupa IMF, in 0.5-2 and 2-10 keV:

    L_X = alpha_0 (1 + z)^gamma M* + beta_0 (1 + z)^delta SFR

Mineo et al. (2014, MNRAS 437, 1698) fit high-mass X-ray binaries and hot gas
together out to z of about 1.3, for a Salpeter IMF, and find no evolution:

    L_X(0.5-8 keV) = 4.0e39 SFR

CIGALE's SFRs are Chabrier, close to Kroupa, so for Mineo they are raised to
Salpeter by 1/0.63 (Madau & Dickinson 2014). Each band is converted to the eROSITA
0.2-2.3 keV band assuming a Gamma = 2 power law, for which a band's luminosity
scales with the log of its energy ratio. A galaxy is flagged when its observed
luminosity is at most K times any of the three predictions.

Only DESI GALAXY sources are flagged. A quasar dominates its own photometry, which
inflates the CIGALE star formation rate and would flag the AGN we mean to keep. A
source missing L_X, M* or SFR cannot be classified and is never flagged.
"""

from __future__ import annotations

import numpy as np

K = 10.0
EROSITA_BAND = (0.2, 2.3)                                   # keV
#: band (keV) -> (log alpha_0, gamma, log beta_0, delta), Lehmer et al. 2016, Table 3
LEHMER16 = {
    (0.5, 2.0): (29.04, 3.78, 39.38, 0.99),
    (2.0, 10.0): (29.37, 2.03, 39.28, 1.31),
}
#: erg/s per (Msun/yr, Salpeter) in 0.5-8 keV, Mineo et al. 2014
MINEO14 = 4.0e39
MINEO14_BAND = (0.5, 8.0)
SALPETER_PER_CHABRIER = 1 / 0.63
#: recorded in each run's choices.json, so a run says which rule withheld its rows
DESCRIPTION = ("flagged if L_X(0.2-2.3 keV) <= 10x the star-formation prediction of "
               "Lehmer+16 (0.5-2 or 2-10 keV) or Mineo+14 (0.5-8 keV), Gamma = 2 "
               "band conversion, DESI GALAXY only, unclassifiable sources kept")


def to_erosita(band: tuple[float, float]) -> float:
    """L(0.2-2.3 keV) / L(band) for a Gamma = 2 power law."""
    return np.log(EROSITA_BAND[1] / EROSITA_BAND[0]) / np.log(band[1] / band[0])


def predicted_log_lx(band: tuple[float, float], logmstar, log_sfr, z) -> np.ndarray:
    """log10 of the X-ray luminosity star formation predicts, in erg/s, in 0.2-2.3 keV."""
    log_alpha, gamma, log_beta, delta = LEHMER16[band]
    logmstar, log_sfr, z = (np.asarray(a, float) for a in (logmstar, log_sfr, z))
    lx = (10.0 ** (log_alpha + logmstar) * (1 + z) ** gamma
          + 10.0 ** (log_beta + log_sfr) * (1 + z) ** delta)
    return np.log10(to_erosita(band) * lx)


def mineo_log_lx(log_sfr) -> np.ndarray:
    """log10 of Mineo et al. (2014)'s X-ray luminosity for a CIGALE (Chabrier) SFR,
    in erg/s, in 0.2-2.3 keV."""
    sfr = SALPETER_PER_CHABRIER * 10.0 ** np.asarray(log_sfr, float)
    return np.log10(to_erosita(MINEO14_BAND) * MINEO14 * sfr)


def sf_dominated(log_lx, logmstar, log_sfr, z, spectype, k: float = K) -> np.ndarray:
    """True where star formation could supply at least 1/k of a galaxy's X-rays."""
    log_lx = np.asarray(log_lx, float)
    known = (np.isfinite(log_lx) & np.isfinite(np.asarray(logmstar, float))
             & np.isfinite(np.asarray(log_sfr, float)) & np.isfinite(np.asarray(z, float))
             & (np.asarray(spectype) == "GALAXY"))
    flagged = np.zeros(log_lx.shape, bool)
    with np.errstate(invalid="ignore"):
        predictions = [predicted_log_lx(band, logmstar, log_sfr, z) for band in LEHMER16]
        predictions.append(mineo_log_lx(log_sfr))
        for prediction in predictions:
            flagged |= known & (log_lx <= prediction + np.log10(k))
    return flagged
