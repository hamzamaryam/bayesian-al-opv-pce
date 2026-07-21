#!/usr/bin/env python
"""
make_supplementary_movies.py
============================
Generate Supplementary_Movie_1.mp4 / Supplementary_Movie_2.mp4 style
animations from the GPR + EI active learning benchmark defined in
active_learning_benchmark_700dpi.py.

Layout (mirrors the reference supplementary movies)
---------------------------------------------------
2 x 4 panel grid, one frame per active-learning iteration:
  - Top-left panel : "Number of Experiments" — step curves of the best
                     (max) PCE found so far, one curve per molecular
                     representation, plus a dashed "Max Value" line at
                     the dataset optimum.
  - 7 other panels : one per representation — scatter of PCE vs. test
                     candidate index. Gray = full unlabeled pool
                     ("Values"), black = initial seed molecules
                     ("Initial Test"), colored = molecules acquired by
                     EI so far ("Tests").

Reference movies play at 1.67 fps, 1440 x 936 px:
  - Supplementary_Movie_1.mp4 :  60 iterations (~36 s)
  - Supplementary_Movie_2.mp4 : 200 iterations (~120 s)

Outputs are written to al_results/ so the original supplementary movies
in the project root are never overwritten.

Usage
-----
  python make_supplementary_movies.py                 # both movies, full runs
  python make_supplementary_movies.py --iters 5       # quick smoke test
  python make_supplementary_movies.py --reps RDKit MACCS
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import imageio

# Reuse dataset paths, CONFIG, fingerprint registry, GPR and EI from the
# benchmark script (importing does not trigger its main()).
import active_learning_benchmark_700dpi as bench

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR    = Path(__file__).parent
RESULTS_DIR = bench.CONFIG["results_dir"]

# Seven representations → 2x4 layout
# (1 curve panel + 7 scatter panels).
DEFAULT_REPS = ["RDKit", "MACCS", "Morgan", "Avalon", "Mordred", "PF", "APF"]

# Colors matched to the reference movies' palette (green, blue, red,
# purple, orange, brown, teal for the seven strategy panels).
REP_COLORS = ["green", "blue", "red", "purple", "orange",
              "brown", "teal", "magenta"]

MOVIES = {
    "Supplementary_Movie_1.mp4": {"n_iterations": 60},
    "Supplementary_Movie_2.mp4": {"n_iterations": 200},
}

FPS        = 1.67
FIG_SIZE   = (19.2, 9.36)   # x 200 dpi → 3840 x 1872 px (widened for 4 columns)
FIG_DPI    = 200
MOVIE_SEED = 0


# ==============================================================================
# ACTIVE LEARNING WITH SELECTION HISTORY
# ==============================================================================

def run_gpr_al_with_history(X_full, y_full, n_iterations, seed, rep_name):
    """
    Same GPR + EI loop as bench.run_gpr_al_loop, but records which
    molecule is acquired at every iteration so frames can be drawn later.

    Returns
    -------
    dict with:
        seed_idx  – list[int], initial labeled molecule indices
        acquired  – list[int], molecule index chosen at each iteration
        best_pce  – list[float], best PCE found so far per iteration
    """
    rng = np.random.default_rng(seed)
    N = len(y_full)

    seed_idx  = [int(i) for i in
                 rng.choice(N, size=bench.CONFIG["n_seed"], replace=False)]
    labeled   = list(seed_idx)
    unlabeled = [i for i in range(N) if i not in set(labeled)]

    acquired, best_pce = [], []

    for itr in range(n_iterations):
        X_train, y_train = X_full[labeled], y_full[labeled]
        X_pool = X_full[unlabeled]

        gpr = bench.build_gpr()
        try:
            gpr.fit(X_train, y_train)
            mu, sigma = gpr.predict(X_pool, return_std=True)
            ei = bench.expected_improvement(
                mu, sigma, float(y_train.max()), xi=bench.CONFIG["ei_jitter"]
            )
            chosen_local = int(np.argmax(ei))
        except Exception as exc:
            print(f"    [WARN] {rep_name} GPR failed at iter {itr}: {exc}. "
                  f"Random fallback.")
            chosen_local = int(rng.integers(len(unlabeled)))

        chosen_global = unlabeled.pop(chosen_local)
        labeled.append(chosen_global)
        acquired.append(chosen_global)
        best_pce.append(float(y_full[labeled].max()))

        if (itr + 1) % 25 == 0:
            print(f"    [{rep_name}] iter={itr+1:3d} | "
                  f"best PCE: {best_pce[-1]:.2f}%")

    return {"seed_idx": seed_idx, "acquired": acquired, "best_pce": best_pce}


def get_histories(rep_names, fingerprint_arrays, y_full, n_iterations, seed):
    """
    Run (or load cached) AL selection histories for every representation.
    Cached to al_results/movie_histories_seed{seed}.json; reused as long
    as the cache holds at least n_iterations iterations per rep.
    """
    cache_path = RESULTS_DIR / f"movie_histories_seed{seed}.json"
    histories = {}
    if cache_path.exists():
        with open(cache_path) as f:
            histories = json.load(f)

    changed = False
    for rep in rep_names:
        if rep in histories and len(histories[rep]["acquired"]) >= n_iterations:
            print(f"[{rep}] using cached history "
                  f"({len(histories[rep]['acquired'])} iterations)")
            continue
        print(f"[{rep}] running GPR + EI active learning "
              f"({n_iterations} iterations) …")
        histories[rep] = run_gpr_al_with_history(
            fingerprint_arrays[rep], y_full, n_iterations, seed, rep
        )
        changed = True

    if changed:
        with open(cache_path, "w") as f:
            json.dump(histories, f)
        print(f"Histories cached → {cache_path}")

    return histories


# ==============================================================================
# FRAME RENDERING
# ==============================================================================

def draw_frame(fig, axes, rep_names, histories, y_full, y_max, t, n_iterations):
    """Draw frame t (t acquisitions done) onto the given figure."""
    x_all = np.arange(len(y_full))

    for ax in axes.flat:
        ax.clear()

    # ---- Panel 1: best-PCE step curves ----
    ax0 = axes[0, 0]
    for rep, color in zip(rep_names, REP_COLORS):
        best = histories[rep]["best_pce"][:t]
        if best:
            seed_best = float(y_full[histories[rep]["seed_idx"]].max())
            ax0.step(np.arange(t + 1), [seed_best] + best,
                     where="post", color=color, lw=1.2, label=rep)
        else:
            ax0.plot([], [], color=color, lw=1.2, label=rep)
    ax0.axhline(y_max, color="k", ls="--", lw=0.8, label="Max Value")
    ax0.set_title("Number of Experiments", fontsize=10)
    ax0.set_ylabel("Max PCE", fontsize=9)
    ax0.set_xlim(0, n_iterations)
    ax0.set_ylim(0, y_max * 1.1)
    ax0.legend(fontsize=6, loc="lower right")
    ax0.grid(True, alpha=0.3, ls="--")

    # ---- Panels 2-8: per-representation scatter ----
    for (rep, color, ax) in zip(rep_names, REP_COLORS, axes.flat[1:]):
        h = histories[rep]
        tested = h["acquired"][:t]
        ax.scatter(x_all, y_full, s=8, color="0.8", alpha=0.6,
                   label="Values", edgecolors="none")
        ax.scatter(h["seed_idx"], y_full[h["seed_idx"]], s=18,
                   color="black", label="Initial Test")
        ax.scatter(tested, y_full[tested], s=18, color=color, label="Tests")
        ax.set_title(rep, fontsize=10)
        ax.set_xlabel("Test Candidates", fontsize=9)
        ax.set_ylabel("PCE", fontsize=9)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_xticks([])
        ax.legend(fontsize=6, loc="lower right", markerscale=0.8)

    # Hide any unused panels (when fewer than 7 representations are given)
    for ax in axes.flat[1 + len(rep_names):]:
        ax.axis("off")

    fig.tight_layout()


def render_movie(out_path, rep_names, histories, y_full, n_iterations):
    """Render one .mp4: frame 0 = seed only, frame t = t acquisitions."""
    y_max = float(y_full.max())
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZE, dpi=FIG_DPI)

    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264",
        pixelformat="yuv420p", macro_block_size=1,
    )
    for t in range(n_iterations + 1):
        draw_frame(fig, axes, rep_names, histories, y_full, y_max,
                   t, n_iterations)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        if t % 25 == 0:
            print(f"  frame {t}/{n_iterations}")
    writer.close()
    plt.close(fig)
    print(f"Movie saved → {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", nargs="+", default=DEFAULT_REPS,
                        choices=list(bench.FINGERPRINT_REGISTRY.keys()),
                        help="representations to include (7 fits the 2x4 grid)")
    parser.add_argument("--iters", type=int, default=None,
                        help="override iteration counts (smoke testing)")
    parser.add_argument("--seed", type=int, default=MOVIE_SEED)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    df = pd.read_csv(bench.CONFIG["data_file"])
    y_full = df[bench.CONFIG["target_col"]].values.astype(np.float32)
    smiles_list = df[bench.CONFIG["smiles_col"]].astype(str).tolist()
    print(f"Dataset: {len(df)} molecules | "
          f"PCE {y_full.min():.2f}–{y_full.max():.2f} %")

    # ---- Load fingerprint arrays (from benchmark cache, else compute) ----
    fp_cache_dir = RESULTS_DIR / "fp_cache"
    fp_cache_dir.mkdir(exist_ok=True)
    fingerprint_arrays = {}
    for rep in args.reps:
        cache_path = fp_cache_dir / f"{rep}.npy"
        if cache_path.exists():
            fingerprint_arrays[rep] = np.load(cache_path)
            print(f"[{rep}] loaded cached features {fingerprint_arrays[rep].shape}")
        else:
            print(f"[{rep}] computing fingerprints …")
            cfg = bench.FINGERPRINT_REGISTRY[rep]
            raw_df = cfg["fn"](smiles_list)
            X = bench.preprocess_features(
                raw_df, apply_corr_filter=cfg.get("corr_filter", False)
            )
            np.save(cache_path, X)
            fingerprint_arrays[rep] = X

    # ---- Run AL once at the longest horizon; movies slice this history ----
    movie_iters = {name: (args.iters if args.iters else cfg["n_iterations"])
                   for name, cfg in MOVIES.items()}
    max_iters = max(movie_iters.values())
    histories = get_histories(args.reps, fingerprint_arrays, y_full,
                              max_iters, args.seed)

    # ---- Render movies ----
    for name, n_it in movie_iters.items():
        out_path = RESULTS_DIR / name
        print(f"\nRendering {name} ({n_it} iterations) …")
        render_movie(out_path, args.reps, histories, y_full, n_it)

    print("\nDone. Movies are in", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
