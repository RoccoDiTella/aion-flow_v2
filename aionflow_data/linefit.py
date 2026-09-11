"""Single-component emission-line fitter for the baseline's four line fluxes.

Each line complex is fitted in the rest frame over a fixed window with a local
linear continuum plus one kinematic component: every line in the complex shares
one velocity and one velocity width, and lines of one species have fixed
relative amplitudes (the [O III] doublet at 0.335, [N II] at 1:2.96 next to
H-alpha, with a free [N II] amplitude). The integrated flux of the primary line
follows from the fitted amplitude and width, F = A * sigma_lambda * sqrt(2 pi),
which convolution preserves, so no instrumental deconvolution is needed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

C_KMS = 299792.458
MIN_COVER = 0.80
MIN_PIXELS = 60
SQRT2PI = float(np.sqrt(2.0 * np.pi))

COMPLEXES = {
    "nev": dict(
        species=[("nev", [(3426.85, 1.0)])],
        window=(3360.0, 3500.0), primary=("nev", 3426.85), vmax=1500.0, smax=2000.0),
    "hbeta": dict(
        species=[("hb", [(4862.683, 1.0)])],
        window=(4770.0, 4940.0), primary=("hb", 4862.683), vmax=1500.0, smax=3000.0),
    "oiii": dict(
        species=[("oiii", [(4960.295, 0.335), (5008.240, 1.0)])],
        window=(4920.0, 5080.0), primary=("oiii", 5008.240), vmax=1500.0, smax=1500.0),
    "halpha": dict(
        species=[("ha", [(6564.614, 1.0)]), ("nii", [(6549.86, 1.0), (6585.27, 2.96)])],
        window=(6480.0, 6650.0), primary=("ha", 6564.614), vmax=1500.0, smax=3000.0),
}


def in_window(name: str, z: np.ndarray, lam_lo: float, lam_hi: float) -> np.ndarray:
    """True where the complex's rest window lies inside [lam_lo, lam_hi] observed."""
    lo, hi = COMPLEXES[name]["window"]
    z = np.asarray(z, np.float64)
    return np.isfinite(z) & (z > 0) & (z >= lam_lo / lo - 1.0) & (z <= lam_hi / hi - 1.0)


def _species_flux(x, lines, amp, v, s):
    out = np.zeros_like(x)
    for lam0, ratio in lines:
        centre = lam0 * (1.0 + v / C_KMS)
        sigma = centre * s / C_KMS
        out += (amp * ratio) * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    return out


def make_model(cx: dict):
    species = cx["species"]
    ref = cx["primary"][1]

    def model(p, x):
        y = p[0] + p[1] * (x - ref)
        for i, (_, lines) in enumerate(species):
            y = y + _species_flux(x, lines, p[4 + i], p[2], p[3])
        return y

    return model, len(species)


def _resid(p, x, y, w, model):
    return (model(p, x) - y) * w


def fit_one(name: str, lam_rest: np.ndarray, flux: np.ndarray, ivar: np.ndarray) -> dict:
    """Fit one complex on a rest-frame spectrum. Returns a dict with `status`.

    On status "ok": continuum (c0, slope), v_kms, sigma_kms, amp (primary species
    amplitude), an (amp / median noise), flux_rest and flux_rest_err (the primary
    line integrated over rest wavelength, in flux-density units times Angstrom),
    chi2, rchi2, n.
    """
    cx = COMPLEXES[name]
    model, n_species = make_model(cx)
    out: dict = {"line": name, "status": ""}
    lo_w, hi_w = cx["window"]
    sel = (lam_rest >= lo_w) & (lam_rest <= hi_w)
    n_win = int(sel.sum())
    if n_win < MIN_PIXELS:
        out["status"] = "off_grid"
        return out
    x, y, iv = (lam_rest[sel].astype(float), flux[sel].astype(float), ivar[sel].astype(float))
    good = (iv > 0) & np.isfinite(y)
    if good.sum() < MIN_COVER * n_win:
        out["status"] = "window_ivar"
        return out
    x, y, iv = x[good], y[good], iv[good]
    w = np.sqrt(iv)
    noise = float(np.median(1.0 / np.sqrt(iv)))
    if not np.isfinite(noise) or noise <= 0:
        out["status"] = "bad_ivar"
        return out
    n = x.size
    span = hi_w - lo_w
    edge = (x < lo_w + 0.22 * span) | (x > hi_w - 0.22 * span)
    c0 = float(np.median(y[edge])) if edge.sum() > 10 else float(np.median(y))
    peak = float(np.nanmax(y) - c0)
    if not np.isfinite(peak):
        out["status"] = "bad_flux"
        return out
    amp_max = max(10.0 * abs(peak), 50.0 * noise, 1e-3)
    vmax, smax = cx["vmax"], cx["smax"]
    lo = np.array([-np.inf, -np.inf, -vmax, 25.0] + [0.0] * n_species)
    hi = np.array([np.inf, np.inf, vmax, smax] + [amp_max] * n_species)
    best = None
    for s0 in (120.0, 300.0, 700.0):
        p0 = np.clip(np.array([c0, 0.0, 0.0, s0] + [max(peak, 3 * noise)] * n_species),
                     lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(_resid, p0, bounds=(lo, hi), args=(x, y, w, model),
                              method="trf", max_nfev=3000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        out["status"] = "fit_fail"
        return out
    p = best.x
    primary_index = [s[0] for s in cx["species"]].index(cx["primary"][0])
    i_amp = 4 + primary_index
    amp, v, s = float(p[i_amp]), float(p[2]), float(p[3])
    centre = cx["primary"][1] * (1.0 + v / C_KMS)
    sigma_lambda = centre * s / C_KMS
    flux_rest = amp * sigma_lambda * SQRT2PI
    err = np.nan
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
        var_a, var_s = float(cov[i_amp, i_amp]), float(cov[3, 3])
        if amp > 0 and s > 0 and var_a >= 0 and var_s >= 0:
            err = flux_rest * float(np.sqrt(var_a / amp ** 2 + var_s / s ** 2))
    except np.linalg.LinAlgError:
        pass
    n_par = 4 + n_species
    chi2 = 2.0 * float(best.cost)
    out.update(status="ok", n=n, noise=noise, c0=float(p[0]), slope=float(p[1]),
               v_kms=v, sigma_kms=s, amp=amp, an=amp / noise, flux_rest=flux_rest,
               flux_rest_err=err, chi2=chi2, rchi2=chi2 / max(n - n_par, 1),
               at_bound=int(abs(v) > vmax - 10 or s > smax - 10))
    return out
