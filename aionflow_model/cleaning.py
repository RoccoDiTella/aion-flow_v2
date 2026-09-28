"""Galaxies whose X-rays star formation could explain.

X-ray binaries and hot gas emit X-rays in proportion to a galaxy's stellar mass and
star formation rate. Where that emission could account for the observed luminosity,
the X-ray labels do not measure the AGN, and a run can withhold them from training.

The prediction is Lehmer et al. (2016, ApJ 825, 7), Table 3, their fit to the local
sample of Lehmer et al. (2010) plus the Chandra Deep Field-South, for a Kroupa IMF:

    L_X(XRB) = alpha_0 (1 + z)^gamma M* + beta_0 (1 + z)^delta SFR

Each band is converted to the eROSITA 0.2-2.3 keV band assuming a Gamma = 2 power
law, for which a band's luminosity scales with the log of its energy ratio. A galaxy
is flagged when its observed luminosity is at most K times the prediction in either
band.

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


def sf_dominated(log_lx, logmstar, log_sfr, z, spectype, k: float = K) -> np.ndarray:
    """True where star formation could supply at least 1/k of a galaxy's X-rays."""
    log_lx = np.asarray(log_lx, float)
    known = (np.isfinite(log_lx) & np.isfinite(np.asarray(logmstar, float))
             & np.isfinite(np.asarray(log_sfr, float)) & np.isfinite(np.asarray(z, float))
             & (np.asarray(spectype) == "GALAXY"))
    flagged = np.zeros(log_lx.shape, bool)
    with np.errstate(invalid="ignore"):
        for band in LEHMER16:
            flagged |= known & (log_lx <= predicted_log_lx(band, logmstar, log_sfr, z)
                                + np.log10(k))
    return flagged
