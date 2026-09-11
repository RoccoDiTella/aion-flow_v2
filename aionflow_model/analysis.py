"""Cut M8: what the posteriors say once they are trained.

    python -m aionflow_model.analysis --out results [--config CONFIG]
        [--marginals runs/marginals] [--rates runs/rates] [--joint4 runs/joint4]

Three things, each from a different run.

Composite targets. "Specific star formation rate (sSFR) is defined as star
formation rate per unit stellar mass. We can treat it as a composite target,
since the joint posterior over the two underlying targets (SFR and M*) recovers
it directly with a well-propagated uncertainty. Across the sample, the catalogue
sSFR is 0.24 nats more likely under the joint posterior than under the same
model's one-dimensional SFR and M* heads treated as independent." The density of
s = log SFR - log M* is the integral of p(s + m, m) over m, taken here on a
Riemann sum over the stellar-mass axis; the two densities differ only in which
joint goes inside, so the change of variables is the same for both and the
difference is what the paper quotes.

Hardness ratio. "We write the ratio of the two latent rates as HR = (lP3 - lP2) /
(lP3 + lP2)." Draws from the (lP2, lP3) joint give a posterior over it per source.
The count gain that goes with it is already an output of `evaluate`: it is the
rate head's information gain against the KDE prior over the same pair of rates.

Within-object correlation. "Draws from the four-dimensional joint over (lP2, lP3,
SFR, M*) provide a correlation rho between HR and sSFR for each source." rho is
the Pearson correlation of HR and log sSFR over a source's draws, and the
fractions the paper quotes are over test-split galaxies, over the nearby ones,
and over quasars. Appendix D's check recomputes it with the image and WISE
withheld, which is a conditioning mask and nothing more.

Bootstrap standard errors and Figure 2's redshift trend read `per_source.csv` and
need no model at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aionflow_data.common import load_config

from .data import MODALITIES, SCALAR_TARGETS, Split, Standardizer, TokenDataset, loader
from .evaluate import common_subsample
from .objective import SUBSETS, Model
from .train import CHECKPOINT, chunks, to_device

ANALYSIS = "analysis.json"
RHO = "rho.csv"
HARDNESS = "hardness.csv"
DRAWS = 32_768          # the paper's draws per source for rho
MASS_NODES = 256        # nodes on the stellar-mass axis of the sSFR integral
MASS_SPAN = 8.0         # standardized units of stellar mass either side, not the paper's
                        # Poisson span: this integral is a change of variables, and
                        # truncating it at five costs the tails of log sSFR
REPLICATES = 1_000      # bootstrap resamples of test sources
WINDOW = 0.08           # rank-ordered fraction of the test set in Figure 2's rolling mean
NEARBY = 0.7            # "nearby galaxies with z < 0.7"


class AnalysisError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- statistics

def bootstrap_se(values: np.ndarray, replicates: int = REPLICATES, seed: int = 0) -> float:
    """Standard error of the mean, by resampling test sources."""
    values = np.asarray(values, float)
    if values.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(replicates, values.size))
    return float(values[picks].mean(axis=1).std(ddof=1))


def rolling_mean(x: np.ndarray, y: np.ndarray, window: float = WINDOW,
                 replicates: int = REPLICATES, seed: int = 0) -> dict[str, np.ndarray]:
    """Mean of `y` over a centred window of `window` of the sources, ordered by `x`,
    with a 95% band from resampling the sources inside each window."""
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    half = max(1, int(round(window * x.size / 2)))
    rng = np.random.default_rng(seed)
    centres, means, lo, hi = [], [], [], []
    for i in range(x.size):
        a, b = max(0, i - half), min(x.size, i + half + 1)
        block = y[a:b]
        centres.append(x[i])
        means.append(block.mean())
        picks = rng.integers(0, block.size, size=(replicates, block.size))
        spread = block[picks].mean(axis=1)
        lo.append(np.quantile(spread, 0.025))
        hi.append(np.quantile(spread, 0.975))
    return {"z": np.asarray(centres), "mean": np.asarray(means),
            "lo": np.asarray(lo), "hi": np.asarray(hi)}


def modality_gains(frame: pd.DataFrame, head: str = "lx") -> dict:
    """What each modality adds to redshift, per source and pooled: Figure 2."""
    keep = frame[f"common_{head}"].to_numpy(bool)
    base = frame.loc[keep, f"ll_{head}_Z"].to_numpy()
    out = {"redshift_alone": float((base - frame.loc[keep, f"prior_{head}"]).mean()),
           "n": int(keep.sum()), "per_modality": {}, "attributed_fraction": {}}
    pooled = {}
    for modality in ("S", "I", "W"):
        column = f"ll_{head}_" + "".join(sorted("Z" + modality, key="ZSIW".index))
        gain = frame.loc[keep, column].to_numpy() - base
        pooled[modality] = float(gain.mean())
        out["per_modality"][modality] = {
            "pooled": pooled[modality],
            "se": bootstrap_se(gain),
            "vs_redshift": rolling_mean(frame.loc[keep, "redshift"].to_numpy(), gain),
        }
    total = sum(pooled.values())
    out["attributed_fraction"] = {k: v / total for k, v in pooled.items()} if total else {}
    return out


def combination_table(frame: pd.DataFrame, heads) -> list[dict]:
    """Information gain and its bootstrap standard error for every row of Table 1."""
    from .objective import SUBSET_NAMES
    rows = []
    for head in heads:
        keep = frame[f"common_{head}"].to_numpy(bool)
        prior = frame.loc[keep, f"prior_{head}"].to_numpy()
        for name in SUBSET_NAMES:
            gain = frame.loc[keep, f"ll_{head}_{name}"].to_numpy() - prior
            rows.append({"head": head, "inputs": name, "n": int(keep.sum()),
                         "information_gain": float(gain.mean()),
                         "se": bootstrap_se(gain)})
    return rows


# ----------------------------------------------------------------------------- composites

def hardness_ratio(rates: np.ndarray) -> np.ndarray:
    """(lP3 - lP2) / (lP3 + lP2) from the two latent rates, last axis (..., 2)."""
    p2, p3 = rates[..., 0], rates[..., 1]
    return (p3 - p2) / (p3 + p2)


def decode_draws(draws: np.ndarray, targets, standardizer: Standardizer) -> dict[str, np.ndarray]:
    """Each dimension of a joint's draws in natural units; rates as rates, not logs."""
    out = {}
    for d, target in enumerate(targets):
        natural = standardizer.decode(target, draws[..., d])
        out[target] = 10.0 ** natural if target.startswith("rate_") else natural
    return out


def rho_from_draws(draws: np.ndarray, targets, standardizer: Standardizer) -> np.ndarray:
    """Pearson correlation of HR and log sSFR over each source's draws."""
    natural = decode_draws(draws, targets, standardizer)
    rates = np.stack([natural["rate_p2"], natural["rate_p3"]], axis=-1)
    hr = hardness_ratio(rates)
    ssfr = natural["sfr"] - natural["mstar"]
    a = hr - hr.mean(axis=1, keepdims=True)
    b = ssfr - ssfr.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(1) / denominator, np.nan)


def mass_grid(s: torch.Tensor, standardizer: Standardizer, nodes: int = MASS_NODES,
              device=None) -> tuple[torch.Tensor, float]:
    """The (u_sfr, u_mstar) nodes that all give log sSFR = s, and the node spacing in
    natural stellar-mass units.

    The density of s = log SFR - log M* is the integral of p(s + m, m) over m, and the
    change of variables from the standardized pair is the same whichever joint goes
    inside, so it cancels in the comparison Section 3.3 makes.
    """
    mean_m, scale_m = standardizer.mean["mstar"], standardizer.scale["mstar"]
    mean_s, scale_s = standardizer.mean["sfr"], standardizer.scale["sfr"]
    u2 = torch.linspace(-MASS_SPAN, MASS_SPAN, nodes, dtype=torch.float64,
                        device=device)
    m = mean_m + scale_m * u2                                # log M* in natural units
    u1 = (s[:, None] + m[None, :] - mean_s) / scale_s        # the log SFR that gives this s
    grid = torch.stack([u1, u2[None, :].expand_as(u1)], dim=-1)
    return grid, float(m[1] - m[0])


def _finish(log_p: torch.Tensor, spacing: float, standardizer: Standardizer) -> torch.Tensor:
    jacobian = -np.log(standardizer.scale["sfr"]) - np.log(standardizer.scale["mstar"])
    return torch.logsumexp(log_p, dim=-1) + np.log(spacing) + jacobian


def log_ssfr_joint(flow, context: torch.Tensor, s: torch.Tensor,
                   standardizer: Standardizer, nodes: int = MASS_NODES) -> torch.Tensor:
    """log p(log sSFR) under the (SFR, M*) joint."""
    grid, spacing = mass_grid(s, standardizer, nodes, context.device)
    return _finish(flow.log_prob(grid.to(context.dtype), context).to(torch.float64),
                   spacing, standardizer)


def log_ssfr_independent(flows, contexts, s: torch.Tensor, standardizer: Standardizer,
                         nodes: int = MASS_NODES) -> torch.Tensor:
    """The same, under the two one-dimensional heads treated as independent."""
    sfr_flow, mstar_flow = flows
    sfr_context, mstar_context = contexts
    grid, spacing = mass_grid(s, standardizer, nodes, sfr_context.device)
    log_p = (sfr_flow.log_prob(grid[..., :1].to(sfr_context.dtype), sfr_context)
             + mstar_flow.log_prob(grid[..., 1:].to(mstar_context.dtype), mstar_context))
    return _finish(log_p.to(torch.float64), spacing, standardizer)


# ----------------------------------------------------------------------------- the run

def load_model(run_dir: Path, backbone, device: str) -> Model:
    from .config import load_run, recipe_path
    checkpoint = torch.load(run_dir / CHECKPOINT, map_location=device, weights_only=False)
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    model = Model(backbone, load_run(recipe_path(checkpoint["run"])),
                  standardizer).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def ssfr_comparison(model: Model, split: Split, device: str, chunk: int) -> dict:
    """The catalogue sSFR under the joint against the product of the two scalar heads."""
    joint = model.run.head("sfr_mstar")
    standardizer = model.standardizer
    keep = common_subsample(joint, split)
    truth = torch.from_numpy(split.y_raw[:, SCALAR_TARGETS.index("sfr")]
                             - split.y_raw[:, SCALAR_TARGETS.index("mstar")])
    together, apart = [], []
    batches = loader(TokenDataset(split, standardizer), 512, shuffle=False)
    at = 0
    for batch in batches:
        batch = to_device(batch, device)
        for part in chunks(batch, chunk):
            rows = part["y"].shape[0]
            mask = SUBSETS[-1].to(device).expand(rows, -1)
            contexts = model.contexts(part, mask)
            s = truth[at:at + rows].to(device)
            together.append(log_ssfr_joint(model.flows["sfr_mstar"],
                                           contexts["sfr_mstar"], s,
                                           standardizer).cpu().numpy())
            apart.append(log_ssfr_independent(
                (model.flows["sfr"], model.flows["mstar"]),
                (contexts["sfr"], contexts["mstar"]), s, standardizer).cpu().numpy())
            at += rows
    together, apart = np.concatenate(together)[keep], np.concatenate(apart)[keep]
    difference = together - apart
    return {"n": int(keep.sum()),
            "log_p_joint": float(together.mean()),
            "log_p_independent": float(apart.mean()),
            "gain_nats": float(difference.mean()),
            "se": bootstrap_se(difference)}


@torch.no_grad()
def draw_joint(model: Model, split: Split, head_name: str, mask_row: torch.Tensor,
               device: str, chunk: int, draws: int) -> np.ndarray:
    """Posterior draws of one head for every row, (n, draws, D), in standardized units."""
    out = []
    for batch in loader(TokenDataset(split, model.standardizer), 256, shuffle=False):
        batch = to_device(batch, device)
        for part in chunks(batch, chunk):
            rows = part["y"].shape[0]
            contexts = model.contexts(part, mask_row.to(device).expand(rows, -1))
            out.append(model.flows[head_name].sample(contexts[head_name],
                                                     draws).double().cpu().numpy())
    return np.concatenate(out)


def within_object(model: Model, split: Split, device: str, chunk: int, draws: int,
                  mask_row: torch.Tensor, log=print) -> tuple[np.ndarray, dict]:
    """rho per source, and the fractions Section 3.3 quotes."""
    head = model.run.heads[0]
    sample = draw_joint(model, split, head.name, mask_row, device, chunk, draws)
    rho = rho_from_draws(sample, head.targets, model.standardizer)
    galaxy = split.spectype == "GALAXY"
    quasar = split.spectype == "QSO"
    nearby = galaxy & (split.redshift < NEARBY)
    finite = np.isfinite(rho)

    def share(mask):
        use = mask & finite
        if not use.any():
            return {"n": 0}
        values = rho[use]
        return {"n": int(use.sum()), "fraction_negative": float((values < 0).mean()),
                "mean": float(values.mean()), "median": float(np.median(values)),
                "se": bootstrap_se(values)}

    log(f"[analysis] rho over {int(finite.sum())} sources, {int(galaxy.sum())} galaxies")
    return rho, {"galaxies": share(galaxy), "nearby_galaxies": share(nearby),
                 "quasars": share(quasar), "all": share(np.ones_like(finite)),
                 "draws": draws, "nearby_below_z": NEARBY}


def run(cfg: dict, out: str | Path, *, marginals=None, rates=None, joint4=None,
        device: str = "cpu", chunk: int = 256, draws: int = DRAWS, backbone=None,
        log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    staged, work = Path(cfg["paths"]["staged"]), Path(cfg["paths"]["work"])
    test = Split(staged, work, "test")
    if backbone is None and (marginals or rates or joint4):
        from .encoder import load_backbone
        backbone = load_backbone()
    results: dict = {}
    try:
        if marginals:
            frame = pd.read_csv(Path(marginals) / "per_source.csv")
            results["table"] = combination_table(frame, ["flux", "lx", "sfr", "mstar"])
            results["modality_gains"] = modality_gains(frame)
            results["ssfr"] = ssfr_comparison(load_model(Path(marginals), backbone, device),
                                              test, device, chunk)
        if rates:
            model = load_model(Path(rates), backbone, device)
            sample = draw_joint(model, test, model.run.heads[0].name, SUBSETS[-1], device,
                                chunk, min(draws, 4096))
            natural = decode_draws(sample, model.run.heads[0].targets, model.standardizer)
            hr = hardness_ratio(np.stack([natural["rate_p2"], natural["rate_p3"]], -1))
            pd.DataFrame({"targetid": test.targetid, "spectype": test.spectype,
                          "redshift": test.redshift, "hr_median": np.median(hr, 1),
                          "hr_lo": np.quantile(hr, 0.16, axis=1),
                          "hr_hi": np.quantile(hr, 0.84, axis=1)}).to_csv(out / HARDNESS,
                                                                          index=False)
            results["hardness"] = {"n": int(test.n), "draws": int(sample.shape[1])}
        if joint4:
            model = load_model(Path(joint4), backbone, device)
            rho, summary = within_object(model, test, device, chunk, draws, SUBSETS[-1],
                                         log=log)
            withheld = torch.zeros(len(MODALITIES), dtype=torch.bool)
            withheld[[MODALITIES.index("Z"), MODALITIES.index("S")]] = True
            rho_zs, summary_zs = within_object(model, test, device, chunk,
                                               min(draws, 4096), withheld, log=log)
            summary["withheld_photometry"] = summary_zs
            results["rho"] = summary
            pd.DataFrame({"targetid": test.targetid, "spectype": test.spectype,
                          "redshift": test.redshift, "rho": rho,
                          "rho_zs_only": rho_zs}).to_csv(out / RHO, index=False)
    finally:
        test.close()
    (out / ANALYSIS).write_text(json.dumps(_listify(results), indent=1) + "\n")
    log(f"[analysis] -> {out / ANALYSIS}")
    return results


def _listify(obj):
    """Arrays to lists and non-finite numbers to null, so the output is valid JSON.

    A bootstrap standard error is NaN wherever a subsample holds a single source,
    and json.dumps would otherwise write a bare NaN that no other reader accepts.
    """
    if isinstance(obj, dict):
        return {k: _listify(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _listify(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_listify(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config.yaml)")
    parser.add_argument("--out", required=True, help="where the analysis lands")
    parser.add_argument("--marginals", default=None, help="the four-scalar run directory")
    parser.add_argument("--rates", default=None, help="the two-rate joint run directory")
    parser.add_argument("--joint4", default=None, help="the four-dimensional joint run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args(argv)
    if not (args.marginals or args.rates or args.joint4):
        print("FAIL: give at least one run directory", file=sys.stderr)
        return 1
    try:
        run(load_config(args.config), args.out, marginals=args.marginals, rates=args.rates,
            joint4=args.joint4, device=args.device, chunk=args.chunk, draws=args.draws)
    except (AnalysisError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
