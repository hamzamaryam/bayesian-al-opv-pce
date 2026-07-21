#!/usr/bin/env python
"""
make_supplementary_movies.py
============================
Generate three supplementary-information movies (720p) from the results
stored in ``al_results.json``.  Each movie is an animated version of one
of the paper's result figures, and -- unlike the static figures, which
overlay every representation at once -- each movie shows **one molecular
representation (fingerprint) at a time**, cycling through all of them:

    Movie 1  ->  figure1_pce_histogram_movie.mp4
                 Animated version of Figure 1 (dataset PCE histogram).
                 The static distribution of experimentally measured PCE
                 is drawn in the background; a marker sweeps rightward to
                 show the best PCE discovered so far by the current
                 fingerprint at each active-learning iteration.

    Movie 2  ->  figure2_discovery_curve_movie.mp4
                 Animated version of Figure 2 (discovery curves).  The
                 cumulative fraction of the top-10% PCE pool recovered is
                 drawn progressively over the 100 iterations, mean +/- SD
                 across the five seeds, with the random baseline shown for
                 reference.

    Movie 3  ->  figure3_best_pce_movie.mp4
                 Animated version of Figure 3 (best PCE found).  The
                 highest PCE observed in the labeled pool is drawn
                 progressively, mean +/- SD across the five seeds, with a
                 dashed line at the dataset maximum.

Inputs
------
* ``al_results.json`` (REQUIRED for all three movies) -- the benchmark
  results.  Expected structure::

      { "<Representation>": { "<seed>": { "discovery_curve":  [...100...],
                                          "best_pce_curve":   [...100...],
                                          "calibration_curve":[...100...],
                                          "random_discovery": [...100...] },
                              ... seeds ... },
        ... representations ... }

* ``supplementary_dataset.csv`` (only needed for Movie 1) -- the curated
  dataset with a per-molecule ``PCE`` column (1,481 molecules).  This is
  the source of the Figure 1 histogram; the per-molecule PCE values are
  NOT stored in al_results.json, so this file supplies them.  Movies 2
  and 3 do not use it.

All three movies are rendered at 1280 x 720 (720p).

Dependencies
------------
    pip install numpy pandas matplotlib imageio imageio-ffmpeg

(``imageio-ffmpeg`` bundles an ffmpeg binary, so no separate ffmpeg
installation is required.)

Usage
-----
    python make_supplementary_movies.py                 # all three movies
    python make_supplementary_movies.py --which 2 3     # only movies 2 & 3
    python make_supplementary_movies.py --fps 25 --frame-step 2   # faster
    python make_supplementary_movies.py \
        --json al_results.json --dataset supplementary_dataset.csv \
        --outdir movies
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless / file-only rendering
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Order in which representations are shown (best-performing first).  Any
# representation present in the JSON but missing here is appended at the end.
REP_ORDER = ["Avalon", "Extended", "Morgan", "APF", "MACCS",
             "PF", "Mordred", "GCN", "RDKit"]

# One fixed colour per representation, kept consistent across all movies.
REP_COLORS = {
    "Avalon":   "#e6194B",
    "Extended": "#3cb44b",
    "Morgan":   "#4363d8",
    "APF":      "#f58231",
    "MACCS":    "#911eb4",
    "PF":       "#42d4f4",
    "Mordred":  "#a9a9a9",
    "GCN":      "#bfef45",
    "RDKit":    "#f032e6",
}
FALLBACK_COLOR = "#000000"

# 720p output:  12.8 in x 7.2 in @ 100 dpi  ->  1280 x 720 px
FIG_W_IN, FIG_H_IN, FIG_DPI = 12.8, 7.2, 100

TOP10_THRESHOLD = 12.0                # PCE (%) defining the top-10% pool

# Defaults (overridable on the command line)
DEFAULT_FPS        = 20               # frames per second in the output mp4
DEFAULT_FRAME_STEP = 1                # draw every k-th iteration (>=1)
DEFAULT_HOLD       = 15               # extra frames held on each rep's final state


# ==============================================================================
# DATA LOADING / AGGREGATION
# ==============================================================================

def load_results(json_path):
    """Load al_results.json and return the raw dict."""
    with open(json_path) as f:
        return json.load(f)


def ordered_reps(results):
    """Return representations in REP_ORDER, then any extras, filtered to
    those actually present in the results file."""
    present = list(results.keys())
    reps = [r for r in REP_ORDER if r in present]
    reps += [r for r in present if r not in reps]
    return reps


def _stack(results, rep, key):
    """Stack a per-seed curve into an array of shape (n_seeds, n_iter).
    Returns None if the key is absent for this representation."""
    seeds = sorted(results[rep].keys(), key=lambda s: int(s))
    rows = [results[rep][s][key] for s in seeds if key in results[rep][s]]
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def mean_sd(results, rep, key):
    """Mean and (population) standard deviation across seeds for one curve."""
    arr = _stack(results, rep, key)
    if arr is None:
        return None, None
    return arr.mean(axis=0), arr.std(axis=0)


def random_baseline(results):
    """Mean random-selection discovery curve, averaged over every
    representation/seed that provides it."""
    rows = []
    for rep in results:
        for s in results[rep]:
            if "random_discovery" in results[rep][s]:
                rows.append(results[rep][s]["random_discovery"])
    if not rows:
        return None
    return np.asarray(rows, dtype=float).mean(axis=0)


def color_for(rep):
    return REP_COLORS.get(rep, FALLBACK_COLOR)


# ==============================================================================
# FIGURE / AXIS HELPERS
# ==============================================================================

def new_canvas():
    """Create a fixed-size figure/axis that renders to exactly 1280x720."""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
    ax = fig.add_axes([0.09, 0.12, 0.88, 0.78])   # fixed margins (no tight_layout)
    return fig, ax


def grab_frame(fig):
    """Render the current figure to an RGB numpy array (H x W x 3)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def open_writer(out_path, fps):
    """Open an mp4 writer that yields a browser-friendly H.264 file."""
    return imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        format="FFMPEG", pixelformat="yuv420p",
        macro_block_size=1, quality=8,
    )


# ==============================================================================
# MOVIE 1 -- FIGURE 1 : PCE HISTOGRAM + BEST-PCE-FOUND SWEEP
# ==============================================================================

def render_movie1(results, dataset_csv, out_path, reps, fps, frame_step, hold):
    """Animated Figure 1: dataset PCE histogram (background) with a marker
    tracking the best PCE discovered so far, one representation at a time."""
    df = pd.read_csv(dataset_csv)
    if "PCE" not in df.columns:
        raise ValueError(
            f"{dataset_csv} has no 'PCE' column (found: {list(df.columns)})")
    pce = df["PCE"].dropna().to_numpy(dtype=float)

    pce_max = float(pce.max())
    x_hi = np.ceil(pce_max) + 1.0
    bins = np.linspace(0.0, x_hi, 41)
    counts, _ = np.histogram(pce, bins=bins)
    y_hi = counts.max() * 1.12

    writer = open_writer(out_path, fps)
    n_iter = 100
    for rep in reps:
        best_mean, _ = mean_sd(results, rep, "best_pce_curve")
        if best_mean is None:
            continue
        n_iter = len(best_mean)
        color = color_for(rep)

        steps = list(range(1, n_iter + 1, frame_step))
        if steps[-1] != n_iter:
            steps.append(n_iter)
        steps += [n_iter] * hold                      # hold on final state

        fig, ax = new_canvas()
        for t in steps:
            ax.clear()
            # --- static histogram background ---
            ax.hist(pce, bins=bins, color="0.80", edgecolor="white", linewidth=0.4)
            ax.axvspan(TOP10_THRESHOLD, x_hi, color="gold", alpha=0.12)
            ax.axvline(TOP10_THRESHOLD, color="0.45", ls=":", lw=1.4)
            ax.text(TOP10_THRESHOLD + 0.1, y_hi * 0.96,
                    "top-10% pool (PCE > 12%)", fontsize=10,
                    color="0.35", va="top")

            # --- animated best-PCE-found marker ---
            best = best_mean[t - 1]
            ax.axvline(best, color=color, lw=2.6)
            ax.axvspan(0.0, best, color=color, alpha=0.06)
            ax.plot([best], [y_hi * 0.5], marker="v", color=color,
                    markersize=13, clip_on=False)
            # Keep the label inside the axes: place it to the left of the
            # marker once the marker gets close to the right-hand edge.
            if best > x_hi * 0.70:
                ax.text(best - 0.2, y_hi * 0.55,
                        f"best PCE so far: {best:.2f}%  ",
                        color=color, fontsize=12, fontweight="bold",
                        ha="right", va="bottom")
            else:
                ax.text(best + 0.2, y_hi * 0.55,
                        f"  best PCE so far: {best:.2f}%",
                        color=color, fontsize=12, fontweight="bold",
                        ha="left", va="bottom")

            ax.axvline(pce_max, color="k", ls="--", lw=1.2)
            ax.text(pce_max, y_hi * 0.02, f"dataset max {pce_max:.2f}%  ",
                    ha="right", va="bottom", fontsize=10)

            ax.set_xlim(0.0, x_hi)
            ax.set_ylim(0.0, y_hi)
            ax.set_xlabel("Power Conversion Efficiency (%)", fontsize=13)
            ax.set_ylabel("Number of molecules", fontsize=13)
            ax.set_title(
                f"Figure 1  |  PCE distribution & best PCE discovered  "
                f"-- {rep}   (iteration {t}/{n_iter})",
                fontsize=14, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25, ls="--")
            writer.append_data(grab_frame(fig))
        plt.close(fig)
        print(f"  [Movie 1] {rep} done")
    writer.close()
    print(f"Saved -> {out_path}")


# ==============================================================================
# MOVIE 2 -- FIGURE 2 : DISCOVERY CURVE (progressively drawn)
# ==============================================================================

def render_movie2(results, out_path, reps, fps, frame_step, hold):
    """Animated Figure 2: per-representation discovery curve drawn
    progressively (mean +/- SD across seeds) vs. the random baseline."""
    rand = random_baseline(results)

    # Common y-limit so every representation is on the same scale.
    y_hi = 0.0
    for rep in reps:
        m, sd = mean_sd(results, rep, "discovery_curve")
        if m is not None:
            y_hi = max(y_hi, float((m + sd).max()))
    if rand is not None:
        y_hi = max(y_hi, float(rand.max()))
    y_hi *= 1.15

    writer = open_writer(out_path, fps)
    for rep in reps:
        mean, sd = mean_sd(results, rep, "discovery_curve")
        if mean is None:
            continue
        n_iter = len(mean)
        x = np.arange(1, n_iter + 1)
        color = color_for(rep)

        steps = list(range(1, n_iter + 1, frame_step))
        if steps[-1] != n_iter:
            steps.append(n_iter)
        steps += [n_iter] * hold

        fig, ax = new_canvas()
        for t in steps:
            ax.clear()
            if rand is not None:
                ax.plot(x, rand, color="k", ls="--", lw=1.5,
                        label="Random baseline")
            ax.fill_between(x[:t], (mean - sd)[:t], (mean + sd)[:t],
                            color=color, alpha=0.20)
            ax.plot(x[:t], mean[:t], color=color, lw=2.6, label=rep)
            ax.plot([x[t - 1]], [mean[t - 1]], marker="o", color=color,
                    markersize=8)
            ax.text(0.015, 0.97,
                    f"discovered: {mean[t-1]*100:.1f}% of top-10% pool",
                    transform=ax.transAxes, va="top", fontsize=12,
                    color=color, fontweight="bold")

            ax.set_xlim(0, n_iter)
            ax.set_ylim(0.0, y_hi)
            ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=13)
            ax.set_ylabel("Fraction of Top-10% Molecules Discovered", fontsize=13)
            ax.set_title(
                f"Figure 2  |  Discovery curve -- {rep}   "
                f"(iteration {t}/{n_iter})",
                fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.25, ls="--")
            ax.legend(loc="lower right", fontsize=11)
            writer.append_data(grab_frame(fig))
        plt.close(fig)
        print(f"  [Movie 2] {rep} done")
    writer.close()
    print(f"Saved -> {out_path}")


# ==============================================================================
# MOVIE 3 -- FIGURE 3 : BEST PCE FOUND (progressively drawn)
# ==============================================================================

def render_movie3(results, out_path, reps, fps, frame_step, hold):
    """Animated Figure 3: per-representation best-PCE-found curve drawn
    progressively (mean +/- SD across seeds) vs. the dataset maximum."""
    # Dataset maximum inferred from the highest PCE any run ever found.
    global_max = 0.0
    for rep in results:
        for s in results[rep]:
            if "best_pce_curve" in results[rep][s]:
                global_max = max(global_max, max(results[rep][s]["best_pce_curve"]))

    y_lo, y_hi = None, 0.0
    for rep in reps:
        m, sd = mean_sd(results, rep, "best_pce_curve")
        if m is None:
            continue
        lo = float((m - sd).min())
        y_lo = lo if y_lo is None else min(y_lo, lo)
        y_hi = max(y_hi, float((m + sd).max()))
    y_lo = (y_lo if y_lo is not None else 0.0) - 0.5
    y_hi = max(y_hi, global_max) + 0.5

    writer = open_writer(out_path, fps)
    for rep in reps:
        mean, sd = mean_sd(results, rep, "best_pce_curve")
        if mean is None:
            continue
        n_iter = len(mean)
        x = np.arange(1, n_iter + 1)
        color = color_for(rep)

        steps = list(range(1, n_iter + 1, frame_step))
        if steps[-1] != n_iter:
            steps.append(n_iter)
        steps += [n_iter] * hold

        fig, ax = new_canvas()
        for t in steps:
            ax.clear()
            ax.axhline(global_max, color="k", ls="--", lw=1.5,
                       label=f"Dataset max ({global_max:.2f}%)")
            ax.fill_between(x[:t], (mean - sd)[:t], (mean + sd)[:t],
                            color=color, alpha=0.20)
            ax.plot(x[:t], mean[:t], color=color, lw=2.6, label=rep)
            ax.plot([x[t - 1]], [mean[t - 1]], marker="o", color=color,
                    markersize=8)
            ax.text(0.015, 0.05,
                    f"best PCE found: {mean[t-1]:.2f}%",
                    transform=ax.transAxes, va="bottom", fontsize=12,
                    color=color, fontweight="bold")

            ax.set_xlim(0, n_iter)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=13)
            ax.set_ylabel("Best PCE Found (%)", fontsize=13)
            ax.set_title(
                f"Figure 3  |  Best PCE found -- {rep}   "
                f"(iteration {t}/{n_iter})",
                fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.25, ls="--")
            ax.legend(loc="lower right", fontsize=11)
            writer.append_data(grab_frame(fig))
        plt.close(fig)
        print(f"  [Movie 3] {rep} done")
    writer.close()
    print(f"Saved -> {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default="al_results.json",
                        help="path to al_results.json (default: al_results.json)")
    parser.add_argument("--dataset", default="supplementary_dataset.csv",
                        help="dataset CSV with a 'PCE' column, used only for "
                             "Movie 1 (default: supplementary_dataset.csv)")
    parser.add_argument("--outdir", default=".",
                        help="output directory (default: current directory)")
    parser.add_argument("--which", nargs="+", type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="which movies to render (default: 1 2 3)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"frames per second (default: {DEFAULT_FPS})")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP,
                        help="draw every k-th iteration; >1 = shorter/faster "
                             f"movie (default: {DEFAULT_FRAME_STEP})")
    parser.add_argument("--hold", type=int, default=DEFAULT_HOLD,
                        help="extra frames held on each representation's final "
                             f"state (default: {DEFAULT_HOLD})")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise SystemExit(f"ERROR: results file not found: {json_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_results(json_path)
    reps = ordered_reps(results)
    print(f"Loaded {json_path} | {len(reps)} representations: {', '.join(reps)}")
    print(f"Rendering movies {args.which} at {FIG_W_IN*FIG_DPI:.0f}x"
          f"{FIG_H_IN*FIG_DPI:.0f} (720p), {args.fps} fps\n")

    if 1 in args.which:
        ds = Path(args.dataset)
        if not ds.exists():
            print(f"WARNING: dataset '{ds}' not found -- skipping Movie 1 "
                  f"(the Figure 1 histogram needs per-molecule PCE values, "
                  f"which are not stored in al_results.json).")
        else:
            print("Rendering Movie 1 (Figure 1 -- PCE histogram) ...")
            render_movie1(results, ds,
                          outdir / "figure1_pce_histogram_movie.mp4",
                          reps, args.fps, args.frame_step, args.hold)

    if 2 in args.which:
        print("Rendering Movie 2 (Figure 2 -- discovery curves) ...")
        render_movie2(results,
                      outdir / "figure2_discovery_curve_movie.mp4",
                      reps, args.fps, args.frame_step, args.hold)

    if 3 in args.which:
        print("Rendering Movie 3 (Figure 3 -- best PCE found) ...")
        render_movie3(results,
                      outdir / "figure3_best_pce_movie.mp4",
                      reps, args.fps, args.frame_step, args.hold)

    print("\nDone. Movies written to", outdir.resolve())


if __name__ == "__main__":
    main()
