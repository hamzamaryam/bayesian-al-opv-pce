#!/usr/bin/env python
"""
make_figures.py
===============
Regenerate every manuscript / SI figure from the saved benchmark outputs
(results/al_results.json and data/processed/Combined_df.csv) WITHOUT
re-running the active-learning loop.

Figures written to results/figures/ (PNG at 700 dpi + vector PDF):
  pce_histogram              – Figure 1 (PCE distribution, high-efficiency threshold)
  mean_discovery_curves      – Figure 2 (fraction of the PCE > 12 % pool recovered)
  mean_best_pce_discovered   – Figure 3 (best PCE found vs. oracle calls)
  calibration_curves         – Figure 4 (NLPD, log scale: (a) mean, (b) median over seeds)
  nlpd_per_seed              – Figure S2 (per-seed NLPD for MACCS, Morgan, Extended, Avalon)

Usage
-----
    python src/make_figures.py                       # defaults to repo paths
    python src/make_figures.py --json results/al_results.json \
        --data data/processed/Combined_df.csv --outdir results/figures

Note on terminology: the "high-efficiency pool" is every molecule with
PCE > 12 %, i.e. 370 of 1,481 molecules (25.0 % of the dataset). It was
mislabelled "top-10 %" in earlier versions of the code and manuscript.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_DPI = 700
N_ITER = 100
THRESHOLD = 12.0
COLORS = plt.cm.tab10.colors


def _rep_colors(results: dict) -> dict:
    # Same colour assignment as the benchmark script: index in results order.
    return {rep: COLORS[i % len(COLORS)] for i, rep in enumerate(results)}


def _save(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outdir / stem}.png / .pdf")


def _n_iter(results: dict) -> int:
    """Run length, taken from the data (100 in the paper)."""
    for seeds_data in results.values():
        for v in seeds_data.values():
            if v.get("discovery_curve"):
                return len(v["discovery_curve"])
    return N_ITER


def _stack(seeds_data: dict, key: str, n: int) -> np.ndarray:
    rows = [v[key] for v in seeds_data.values() if len(v.get(key, [])) == n]
    return np.array(rows, dtype=float)


def plot_histogram(y: np.ndarray, outdir: Path):
    n_hi = int((y > THRESHOLD).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(THRESHOLD, color="crimson", ls="--", lw=1.8,
               label=f"High-efficiency threshold (PCE > {THRESHOLD:g}%; "
                     f"{n_hi} molecules, {100 * n_hi / len(y):.1f}%)")
    ax.set_xlabel("Power Conversion Efficiency, PCE (%)", fontsize=11)
    ax.set_ylabel("Number of Molecules", fontsize=11)
    ax.set_title(f"Distribution of PCE Values in the {len(y):,}-Molecule Dataset",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    _save(fig, outdir, "pce_histogram")


def plot_discovery(results: dict, outdir: Path):
    n = _n_iter(results)
    iters = np.arange(1, n + 1)
    cols = _rep_colors(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    random_plotted = False
    for rep, sd_ in results.items():
        dc = _stack(sd_, "discovery_curve", n)
        if dc.size:
            mu, sd = dc.mean(0), dc.std(0)
            ax.plot(iters, mu, label=rep, color=cols[rep], lw=1.8)
            ax.fill_between(iters, mu - sd, mu + sd, alpha=0.12, color=cols[rep])
        if not random_plotted:
            rd = _stack(sd_, "random_discovery", n)
            if rd.size:
                mu, sd = rd.mean(0), rd.std(0)
                ax.plot(iters, mu, "k--", label="Random", lw=1.5)
                ax.fill_between(iters, mu - sd, mu + sd, alpha=0.08, color="k")
                random_plotted = True
    ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=9)
    ax.set_ylabel(f"Fraction of High-Efficiency Pool (PCE > {THRESHOLD:g}%) Recovered",
                  fontsize=9)
    ax.set_title("Discovery Curves — Molecular Representation Benchmark", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=1.0, color="gray", ls=":", lw=1)
    plt.tight_layout()
    _save(fig, outdir, "mean_discovery_curves")


def plot_best_pce(results: dict, outdir: Path):
    n = _n_iter(results)
    iters = np.arange(1, n + 1)
    cols = _rep_colors(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for rep, sd_ in results.items():
        bp = _stack(sd_, "best_pce_curve", n)
        if bp.size:
            mu, sd = bp.mean(0), bp.std(0)
            ax.plot(iters, mu, label=rep, color=cols[rep], lw=1.8)
            ax.fill_between(iters, mu - sd, mu + sd, alpha=0.12, color=cols[rep])
    ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=9)
    ax.set_ylabel("Best PCE Found (%)", fontsize=9)
    ax.set_title("Best PCE Discovered vs. Oracle Calls", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, n)
    plt.tight_layout()
    _save(fig, outdir, "mean_best_pce_discovered")


def plot_calibration(results: dict, outdir: Path):
    """Figure 4: (a) mean and (b) median NLPD over the five seeds, log y-axis.

    The arithmetic mean is dominated by individual seeds whose NLPD reaches
    10^2-10^3; the median shows the typical-seed behaviour. A log axis is used
    so that both regimes are legible on one plot.
    """
    n = _n_iter(results)
    iters = np.arange(1, n + 1)
    cols = _rep_colors(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for rep, sd_ in results.items():
        cc = _stack(sd_, "calibration_curve", n)
        if not cc.size:
            continue  # GCN: no NLPD recorded
        axes[0].plot(iters, np.nanmean(cc, 0), label=rep, color=cols[rep], lw=1.6)
        axes[1].plot(iters, np.nanmedian(cc, 0), label=rep, color=cols[rep], lw=1.6)
    for ax, ttl in zip(axes, ["(a) Mean NLPD over five seeds",
                              "(b) Median NLPD over five seeds"]):
        ax.set_yscale("log")
        ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=13)
        ax.set_title(ttl, fontsize=14)
        ax.set_xlim(0, n)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel("NLPD on unlabeled pool (log scale)", fontsize=13)
    axes[1].legend(fontsize=11, loc="upper right", ncol=2)
    plt.tight_layout()
    _save(fig, outdir, "calibration_curves")


def plot_nlpd_per_seed(results: dict, outdir: Path,
                       reps=("MACCS", "Morgan", "Extended", "Avalon")):
    n = _n_iter(results)
    iters = np.arange(1, n + 1)
    reps = [r for r in reps if r in results]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    seed_cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for ax, rep in zip(axes.ravel(), reps):
        for j, (seed, v) in enumerate(results[rep].items()):
            ax.plot(iters, v["calibration_curve"], color=seed_cols[j % 5], lw=1.3,
                    label=f"seed {seed}")
        ax.set_yscale("log")
        ax.set_title(rep, fontsize=11)
        ax.axhline(10, color="gray", ls=":", lw=1)
    for ax in axes[1]:
        ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("NLPD (log scale)", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    _save(fig, outdir, "nlpd_per_seed")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default=REPO_ROOT / "results" / "al_results.json", type=Path)
    p.add_argument("--data", default=REPO_ROOT / "data" / "processed" / "Combined_df.csv",
                   type=Path)
    p.add_argument("--outdir", default=REPO_ROOT / "results" / "figures", type=Path)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    results = json.loads(a.json.read_text())
    y = pd.read_csv(a.data)["PCE"].to_numpy(dtype=float)

    plot_histogram(y, a.outdir)
    plot_discovery(results, a.outdir)
    plot_best_pce(results, a.outdir)
    plot_calibration(results, a.outdir)
    plot_nlpd_per_seed(results, a.outdir)


if __name__ == "__main__":
    main()
