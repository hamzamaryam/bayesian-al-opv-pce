# Benchmarking Molecular Representations in Bayesian Active Learning for Accelerated Discovery of High-Efficiency Organic Solar Cells

## Overview

This repository contains the code, data, and results accompanying the paper 'Benchmarking Molecular Representations in Bayesian Active Learning for Accelerated Discovery of High-Efficiency Organic Solar Cells'. It benchmarks eight handcrafted molecular representations (RDKit descriptors, Mordred descriptors, MACCS keys, Avalon fingerprints, Morgan circular fingerprints, atom-pair fingerprints, path-based fingerprints, and extended fingerprints) plus a graph convolutional network (GCN) baseline within a standardized Bayesian active learning (AL) loop — a Gaussian process regressor (GPR) surrogate with an Expected Improvement (EI) acquisition function — for discovering high-power-conversion-efficiency (PCE) non-fullerene-acceptor (NFA) organic solar cell (OSC) molecules. The core problem addressed is: given a fixed pool of 1,481 candidate molecules with known PCE, which molecular representation lets an AL agent recover the highest-efficiency candidates in the fewest oracle queries?

## Repository structure

```
.
├── active_learning_benchmark_700dpi.py   # Main pipeline script — produced the committed results in al_results/
├── active_learning_benchmark.py          # Earlier variant of the same pipeline (combined 2-panel figure layout)
├── make_supplementary_movies.py          # Earlier/unused movie-generation script
├── Active_Learning_Data_Generation.ipynb # Builds Combined_df.csv from First_Database.csv + train_test_combine.csv
├── Active_Learning_Rdkit.ipynb           # Exploratory: RDKit descriptors + GPR (single representation)
├── Active_Learning_Rdkit_GPR.ipynb       # Exploratory: RDKit descriptors + GPR, alternate kernel setup
├── Active_Learning_Rdkit_Gpr_RF.ipynb    # Exploratory: RDKit descriptors, GPR vs. random-forest comparison
├── Active_Learning_Mordred.ipynb         # Exploratory: Mordred descriptors + GPR
├── Active_Learning_maccs_fingerprint.ipynb    # Exploratory: MACCS keys + GPR
├── Active_Learning_avalon_fingerprint.ipynb   # Exploratory: Avalon fingerprints + GPR
├── Active_Learning_apf_fingerprint.ipynb      # Exploratory: atom-pair fingerprints + GPR
├── Active_Learning_pf_fingerprint.ipynb       # Exploratory: path-based fingerprints + GPR
├── Active_Learning_extended_fingerprint.ipynb # Exploratory: extended fingerprints + GPR
├── Active_Learning_rdk_fingerprint.ipynb      # Exploratory: RDKit topological fingerprint + GPR
├── Graph_Active_Learning.ipynb           # Exploratory: GCN baseline (precursor to the GCN code in the .py script)
├── HBE_BayesOpt.ipynb                    # Exploratory: Bayesian-optimization scratch notebook
│
├── Combined_df.csv                       # Main dataset: 1,481 molecules, columns [SMILES, PCE] — input to the pipeline
├── combined_df.smi                       # Same molecules as a plain SMILES list
├── First_Database.csv                    # Raw literature-assembled donor/acceptor PCE records (input to Data_Generation.ipynb)
├── train_test_combine.csv                # Second raw literature-assembled SMILES/PCE source (input to Data_Generation.ipynb)
│
├── rdkit_combined_df.csv                 # Precomputed RDKit descriptor matrix for Combined_df.csv (pipeline cache input)
├── mordred_combined_df.csv               # Precomputed Mordred descriptor matrix (pipeline cache input)
├── maccs_combined_df.csv                 # Precomputed MACCS key matrix (pipeline cache input)
├── morgan_combined_df.csv                # Precomputed Morgan fingerprint matrix (pipeline cache input)
├── avalon_combined.csv                   # Precomputed Avalon fingerprint matrix (pipeline cache input)
├── apf_combined.csv                      # Precomputed atom-pair fingerprint matrix (pipeline cache input)
├── pf_combined_df.csv                    # Precomputed path-based fingerprint matrix (pipeline cache input)
├── extended_combined_df.csv              # Precomputed extended fingerprint matrix (pipeline cache input)
├── rdk_fingerprint_combined.csv          # Precomputed RDKit topological fingerprint matrix (used by notebook, not the main script)
├── ttf_combined.csv                      # Legacy combined fingerprint export (not read by the main script)
├── AtomPairs2DFingerprintCount.csv       # Legacy PaDEL-generated descriptor export (not read by the main script)
├── AtomPairs2DFingerprinter.csv          # Legacy PaDEL-generated descriptor export (not read by the main script)
├── EStateFingerprinter.csv               # Legacy PaDEL-generated descriptor export (not read by the main script)
│
├── *.xml, descriptors_xml.zip            # PaDEL-Descriptor fingerprint/descriptor definition files (legacy, unused by the main pipeline script; 12 files)
│
├── active_learning_PCE_Prediction.csv          # Scopus literature-search export used for the paper's literature review
├── larger_active_learning_literature_review.csv # Extended Scopus literature-search export
└── al_results/                           # All pipeline outputs (committed so results can be inspected without re-running)
    ├── al_results.json                   # Raw numeric results: discovery/best-PCE/calibration curves per representation per seed
    ├── pce_histogram.png                 # Figure 1 — PCE distribution of the 1,481-molecule dataset
    ├── mean_discovery_curves.png         # Figure 2 — top-10% discovery fraction vs. iteration, all representations
    ├── mean_best_pce_discovered.png      # Figure 3 — best PCE found vs. oracle calls, all representations
    ├── calibration_curves.png            # GPR uncertainty-calibration diagnostic across representations
    ├── supplementary_dataset.csv         # Per-molecule PCE values used to render Movie 1
    ├── figure1_pce_histogram_movie.mp4   # Animated version of Figure 1 (per-representation sweep)
    ├── figure2_discovery_curve_movie.mp4 # Animated version of Figure 2
    ├── figure3_best_pce_movie.mp4        # Animated version of Figure 3
    ├── make_supplementary_movies_revised.py # Script that generated the three .mp4 files above
    └── fp_cache/                         # Cached preprocessed feature matrices (.npy), one per representation
        ├── APF.npy
        ├── Avalon.npy
        ├── Extended.npy
        ├── MACCS.npy
        ├── Mordred.npy
        ├── Morgan.npy
        ├── PF.npy
        └── RDKit.npy
```

## Installation

Tested on Ubuntu with Python 3.10+ (developed with Python 3.12). No `requirements.txt` / `environment.yml` is currently checked into the repository — the commands below install the packages actually imported by the code; consider freezing them into a `requirements.txt` with `pip freeze > requirements.txt` for exact reproducibility.

```bash
# 1. Clone the repository
git clone <REPO_URL> pce-al-benchmark
cd pce-al-benchmark

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install core dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm networkx

# 4. Install cheminformatics dependencies
pip install rdkit mordred

# 5. Install PyTorch + PyTorch Geometric (CPU build shown; swap the index URL for a CUDA build if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric

# 6. Install dependencies for the supplementary movies (optional)
pip install imageio imageio-ffmpeg

# 7. Only needed if you plan to re-run the exploratory PaDEL-based notebooks
pip install padelpy jupyter
```

## Data

- **`Combined_df.csv`** (repo root) — the dataset used directly by the pipeline. 1,481 rows, columns `SMILES` (molecular structure) and `PCE` (experimentally measured power conversion efficiency, %), range 0.01%–18.38%. This is the file `active_learning_benchmark_700dpi.py` loads at startup (`CONFIG["data_file"]`).
- **`combined_df.smi`** — the same 1,481 molecules as a plain-text SMILES list.
- **`First_Database.csv`** and **`train_test_combine.csv`** — the two raw, literature-assembled source tables (donor/acceptor SMILES with measured PCE) that `Active_Learning_Data_Generation.ipynb` deduplicates, canonicalizes, and concatenates to produce `Combined_df.csv`. Both are included in the repo, so no external download is required to reproduce `Combined_df.csv` from scratch.
- **Precomputed descriptor/fingerprint CSVs** (`rdkit_combined_df.csv`, `mordred_combined_df.csv`, `maccs_combined_df.csv`, `morgan_combined_df.csv`, `avalon_combined.csv`, `apf_combined.csv`, `pf_combined_df.csv`, `extended_combined_df.csv`) — one row per molecule in `Combined_df.csv`, in the same order, with per-representation feature columns. The main script loads these on first run and caches the preprocessed arrays to `al_results/fp_cache/*.npy`; if a cache/CSV is missing it recomputes the representation directly from SMILES with RDKit/Mordred.
- The exact original literature sources (papers/databases) from which `First_Database.csv` and `train_test_combine.csv` were assembled are not recorded in the code: `<DATASET SOURCE / DOI>`.

## Running the pipeline end-to-end

1. **(Optional — only if regenerating the dataset from raw sources)** Run `Active_Learning_Data_Generation.ipynb` top to bottom. Reads `First_Database.csv` and `train_test_combine.csv`, deduplicates by canonical SMILES (keeping the highest PCE per molecule), and writes `Combined_df.csv`. Skip this step if you only want to reproduce the paper's results, since `Combined_df.csv` is already committed.
2. **Run the main benchmark:**
   ```bash
   python active_learning_benchmark_700dpi.py
   ```
   Loads `Combined_df.csv`; computes (or loads from the precomputed CSVs / `al_results/fp_cache/*.npy`) all eight fingerprint/descriptor representations; runs the GPR + EI active learning loop for each representation across 5 random seeds × 100 iterations from a 10-molecule seed set; trains the GCN + MC-Dropout + EI baseline across the same 5 seeds × 100 iterations; saves all numeric results and figures to `al_results/`. This is the script that produced the results currently committed in `al_results/` (`active_learning_benchmark.py` is an earlier variant that writes a combined two-panel `discovery_curves.png` instead).
3. **(Optional) Generate the supplementary movies:**
   ```bash
   python al_results/make_supplementary_movies_revised.py --outdir al_results
   ```
   Requires `al_results/al_results.json` (produced in step 2) and `al_results/supplementary_dataset.csv` (already committed). Produces the three `.mp4` animations under `al_results/`.

The exploratory notebooks (`Active_Learning_Rdkit*.ipynb`, `Active_Learning_*_fingerprint.ipynb`, `Active_Learning_Mordred.ipynb`, `Graph_Active_Learning.ipynb`, `HBE_BayesOpt.ipynb`) are single-representation prototypes that were consolidated and fixed up into `active_learning_benchmark_700dpi.py` (see the "Issues addressed vs. individual notebooks" section in that script's docstring). They are not required to reproduce the paper's results and are kept for audit purposes only.

## Expected outputs

A successful run of `active_learning_benchmark_700dpi.py` writes the following to `al_results/`:

- `pce_histogram.png` — Figure 1: PCE distribution across the 1,481-molecule dataset.
- `mean_discovery_curves.png` — Figure 2: mean ± SD fraction of the top-10% PCE pool recovered vs. AL iteration, per representation, with a random-selection baseline.
- `mean_best_pce_discovered.png` — Figure 3: mean ± SD best PCE found vs. oracle calls, per representation.
- `calibration_curves.png` — GPR predictive-uncertainty calibration diagnostic, per representation.
- `al_results.json` — raw per-representation, per-seed arrays (`discovery_curve`, `best_pce_curve`, `calibration_curve`, `random_discovery`), each of length 100 (one value per AL iteration).
- `fp_cache/*.npy` — cached preprocessed feature matrices, one file per representation (`APF`, `Avalon`, `Extended`, `MACCS`, `Mordred`, `Morgan`, `PF`, `RDKit`).
- A console summary table (printed at the end of the run) ranking representations by final discovery performance.

Running `al_results/make_supplementary_movies_revised.py` additionally produces `figure1_pce_histogram_movie.mp4`, `figure2_discovery_curve_movie.mp4`, and `figure3_best_pce_movie.mp4` in the target `--outdir`.

## Requirements

- **Hardware:** CPU-only is sufficient — the GPR surrogate (scikit-learn) and the GCN baseline (2 graph-convolution layers, hidden dim 64, 60 epochs, batch size 32) are both small enough to train on CPU; a GPU will speed up the GCN portion but is not required.
- **RAM:** Mordred computes ~1,800 descriptors per molecule before PCA reduction (capped at 50 components / 95% explained variance in `CONFIG`), which is the most memory-intensive step. `<RAM REQUIREMENT — fill in based on your test run, e.g. "peak ~X GB observed on my machine">`.
- **Runtime:** The full run repeats the AL loop for 8 representations × 5 seeds × 100 iterations, plus the GCN baseline × 5 seeds × 100 iterations, with GPR retrained from scratch at every iteration. First run is slower because uncached representations (especially Mordred) are computed from SMILES and written to `al_results/fp_cache/`; subsequent runs reuse the cache. `<APPROXIMATE TOTAL RUNTIME — fill in from your hardware, e.g. "~X hours on an N-core CPU">`.

## Citation

If you use this code or dataset, please cite:

<AUTHORS>, "<TITLE>," *<JOURNAL>*, <YEAR>. DOI: <DOI>

```bibtex
@article{<CITEKEY>,
  author  = {<AUTHORS>},
  title   = {<TITLE>},
  journal = {<JOURNAL>},
  year    = {<YEAR>},
  doi     = {<DOI>}
}
```

## License

This code is released under the MIT License (see `LICENSE`).

**TODO:** No `LICENSE` file currently exists in this repository — add one before publishing/sharing the repo.

## Contact

<YOUR NAME>, <AFFILIATION> — <EMAIL>
