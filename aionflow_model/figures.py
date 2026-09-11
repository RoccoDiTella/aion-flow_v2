"""Cut M9: the figures and Table 1.

    python -m aionflow_model.figures --analysis results --marginals runs/marginals
        --baseline runs/baseline --out figures

Figure 1 ranks the 15 input combinations by information gain on broad-band X-ray
flux against the emission-line baseline, with the four-target table beside it.
Figure 2 shows what each modality adds to redshift about log LX, against redshift
and pooled. Figure 3 is the histogram of the within-object correlation over test
galaxies. Figure 4 is a schematic of the probe and is drawn by hand, not here.

Table 1 is the whole 15 by 4 grid of information gain and R2.

Everything here reads `analysis.json` and the run directories' `results.json`;
nothing is recomputed, so a figure and the number quoted beside it cannot drift
apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import TARGETS  # noqa: E402
from .objective import SUBSET_NAMES  # noqa: E402

SCALARS = ("flux", "lx", "sfr", "mstar")
RHO_BIN = 0.005            # the paper's bin width for Figure 3
FIGURES = ("figure1", "figure2", "figure3")
TABLE = "table1.md"


class FigureError(RuntimeError):
    pass


def read(path: Path) -> dict:
    if not path.is_file():
        raise FigureError(f"missing {path}")
    return json.loads(path.read_text())


def gains(rows, head: str) -> dict[str, tuple[float, float]]:
    """Information gain and its standard error per combination, for one head."""
    out = {}
    for row in rows:
        if row["head"] == head:
            se = row.get("se")
            out[row["inputs"]] = (row["information_gain"],
                                  0.0 if se is None else float(se))
    missing = set(SUBSET_NAMES) - set(out)
    if missing:
        raise FigureError(f"head {head!r} has no row for {sorted(missing)}")
    return out


# ----------------------------------------------------------------------------- figures

def figure1(analysis: dict, baseline: dict, out: Path) -> Path:
    """The 15 combinations on X-ray flux, against the baseline."""
    model = gains(analysis["table"], "flux")
    line = gains(baseline["rows"], "flux")["ZSIW"][0]
    order = sorted(SUBSET_NAMES, key=lambda name: model[name][0])
    values = [model[name][0] for name in order]
    errors = [model[name][1] for name in order]
    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    ax.bar(range(len(order)), values, yerr=errors, color="#3b6ea5", capsize=2)
    ax.axhline(line, linestyle="--", color="#b4472a",
               label=f"emission-line baseline ({line:.3f} nats)")
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_ylabel("information gain on X-ray flux (nats)")
    ax.set_xlabel("input combination")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out / "figure1.pdf")


def figure2(analysis: dict, out: Path) -> Path:
    """What each modality adds to redshift about log LX, pooled and against redshift."""
    gains_by_modality = analysis["modality_gains"]["per_modality"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.4), width_ratios=(1, 2),
                                      constrained_layout=True)
    names = {"S": "spectrum", "I": "image", "W": "WISE"}
    colours = {"S": "#3b6ea5", "I": "#4a8b5c", "W": "#b4472a"}
    pooled = [gains_by_modality[m]["pooled"] for m in names]
    left.bar(range(3), pooled, yerr=[gains_by_modality[m]["se"] or 0.0 for m in names],
             color=[colours[m] for m in names], capsize=3)
    left.set_xticks(range(3), [names[m] for m in names])
    left.set_ylabel("nats beyond redshift alone")
    left.set_title(f"pooled (redshift alone: "
                   f"{analysis['modality_gains']['redshift_alone']:.2f} nats)", fontsize=9)
    for modality, label in names.items():
        trend = gains_by_modality[modality]["vs_redshift"]
        z = np.asarray(trend["z"], float)
        right.plot(z, trend["mean"], color=colours[modality], label=label)
        right.fill_between(z, trend["lo"], trend["hi"], color=colours[modality], alpha=0.2)
    right.axhline(0.0, color="0.6", linewidth=0.8)
    right.set_xlabel("redshift")
    right.set_ylabel("nats beyond redshift alone")
    right.legend(frameon=False)
    return _save(fig, out / "figure2.pdf")


def figure3(rho: np.ndarray, summary: dict, out: Path) -> Path:
    """The within-object correlation across test galaxies."""
    rho = np.asarray(rho, float)
    rho = rho[np.isfinite(rho)]
    edges = np.arange(np.floor(rho.min() / RHO_BIN) * RHO_BIN,
                      np.ceil(rho.max() / RHO_BIN) * RHO_BIN + RHO_BIN, RHO_BIN)
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    ax.hist(rho, bins=edges if edges.size > 1 else 1, color="#3b6ea5")
    ax.axvline(0.0, linestyle="--", color="0.3")
    ax.set_xlabel(r"within-object correlation $\rho$(HR, sSFR)")
    ax.set_ylabel(f"test galaxies (n = {summary.get('n', rho.size)})")
    negative = summary.get("fraction_negative")
    if negative is not None:
        ax.set_title(f"{negative:.0%} negative, median {summary['median']:+.3f}", fontsize=9)
    return _save(fig, out / "figure3.pdf")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- table 1

def table1(analysis: dict, results: dict) -> str:
    """The 15 by 4 grid of information gain and R2, as markdown."""
    r2 = {(row["head"], row["inputs"]): row.get(f"r2_{row['head']}")
          for row in results["rows"]}
    sizes = results["common_subsample"]
    header = ("| inputs | " + " | ".join(
        f"{TARGETS[h].label} IG | {TARGETS[h].label} R2" for h in SCALARS) + " |")
    lines = [header, "|" + "---|" * (1 + 2 * len(SCALARS))]
    for name in SUBSET_NAMES:
        cells = []
        for head in SCALARS:
            gain = gains(analysis["table"], head)[name][0]
            value = r2.get((head, name))
            cells.append(f"{gain:.3f} | {'' if value is None else f'{value:.3f}'}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    counts = ", ".join(f"{TARGETS[h].label} {sizes[h]:,}" for h in SCALARS if h in sizes)
    lines.append("")
    lines.append(f"Common subsample, fixed across rows: {counts}.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- the run

def run(analysis_dir: str | Path, out: str | Path, *, marginals=None, baseline=None,
        log=print) -> list[Path]:
    analysis_dir, out = Path(analysis_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    analysis = read(analysis_dir / "analysis.json")
    written = []
    if baseline:
        written.append(figure1(analysis, read(Path(baseline) / "results.json"), out))
    if "modality_gains" in analysis:
        written.append(figure2(analysis, out))
    rho_path = analysis_dir / "rho.csv"
    if rho_path.is_file():
        import pandas as pd
        frame = pd.read_csv(rho_path)
        galaxies = frame[frame["spectype"] == "GALAXY"]
        use = galaxies if len(galaxies) else frame
        written.append(figure3(use["rho"].to_numpy(),
                               analysis.get("rho", {}).get("galaxies", {}), out))
    if marginals:
        (out / TABLE).write_text(table1(analysis, read(Path(marginals) / "results.json")))
        written.append(out / TABLE)
    for path in written:
        log(f"[figures] {path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--analysis", required=True, help="where analysis.py wrote")
    parser.add_argument("--out", required=True, help="where the figures land")
    parser.add_argument("--marginals", default=None, help="the four-scalar run directory")
    parser.add_argument("--baseline", default=None, help="the baseline run directory")
    args = parser.parse_args(argv)
    try:
        run(args.analysis, args.out, marginals=args.marginals, baseline=args.baseline)
    except (FigureError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
