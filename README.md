# Benchmarking Molecular Representations in Bayesian Active Learning for Accelerated Discovery of High-Efficiency Organic Solar Cells

## Overview

This repository contains the code, data, and results accompanying the paper 'Benchmarking Molecular Representations in Bayesian Active Learning for Accelerated Discovery of High-Efficiency Organic Solar Cells'. It benchmarks eight handcrafted molecular representations (RDKit descriptors, Mordred descriptors, MACCS keys, Avalon fingerprints, Morgan circular fingerprints, atom-pair fingerprints, path-based fingerprints, and extended fingerprints) plus a graph convolutional network (GCN) baseline within a standardized Bayesian active learning (AL) loop — a Gaussian process regressor (GPR) surrogate with an Expected Improvement (EI) acquisition function — for discovering high-power-conversion-efficiency (PCE) non-fullerene-acceptor (NFA) organic solar cell (OSC) molecules. The core problem addressed is: given a fixed pool of 1,481 candidate molecules with known PCE, which molecular representation lets an AL agent recover the highest-efficiency candidates in the fewest oracle queries?

## Repository structure

```
.
├── data/
│   ├── raw/                               # Original literature-assembled source tables
│   │   ├── First_Database.csv             # Raw donor/acceptor PCE records
│   │   └── train_test_combine.csv         # Second raw SMILES/PCE source
│   ├── processed/                         # Curated dataset used by the pipeline
│   │   ├── Combined_df.csv                # Main dataset: 1,481 molecules, columns [SMILES, PCE]
│   │   └── combined_df.smi                # Same molecules as a plain SMILES list
│   ├── features/                          # Precomputed descriptor / fingerprint matrices (row order = Combined_df.csv)
│   │   ├── rdkit_combined_df.csv          # RDKit descriptors
│   │   ├── mordred_combined_df.csv        # Mordred descriptors
│   │   ├── maccs_combined_df.csv          # MACCS keys
│   │   ├── morgan_combined_df.csv         # Morgan circular fingerprints
│   │   ├── avalon_combined.csv            # Avalon fingerprints
│   │   ├── apf_combined.csv               # Atom-pair fingerprints
│   │   ├── pf_combined_df.csv             # Path-based fingerprints
│   │   ├── extended_combined_df.csv       # Extended fingerprints
│   │   ├── rdk_fingerprint_combined.csv   # RDKit topological fingerprint (notebook only)
│   │   ├── ttf_combined.csv               # Legacy combined export (not read by the main script)
│   │   └── padel_legacy/                  # Legacy PaDEL-generated exports (not read by the main script)
│   └── padel_descriptor_xml/              # PaDEL-Descriptor definition files (*.xml, descriptors_xml.zip)
│
├── src/                                   # Pipeline scripts (run from anywhere; paths resolve to the repo root)
│   ├── active_learning_benchmark_700dpi.py   # Main pipeline — produced the committed results/
│   ├── active_learning_benchmark.py          # Earlier variant (combined 2-panel figure layout)
│   ├── make_supplementary_movies_revised.py  # Generates the three supplementary .mp4 movies
│   └── make_supplementary_movies.py          # Earlier/unused movie-generation script
│
├── notebooks/
│   ├── Active_Learning_Data_Generation.ipynb # Builds Combined_df.csv and all feature matrices
│   └── exploratory/                          # Single-representation prototypes (kept for audit)
│       ├── Active_Learning_Rdkit*.ipynb, Active_Learning_Mordred.ipynb
│       ├── Active_Learning_{maccs,avalon,apf,pf,extended,rdk}_fingerprint.ipynb
│       ├── Graph_Active_Learning.ipynb       # GCN baseline prototype
│       └── HBE_BayesOpt.ipynb                # Bayesian-optimization scratch notebook
│
├── results/                               # All pipeline outputs (committed for inspection)
│   ├── al_results.json                    # Raw per-representation, per-seed curves
│   ├── supplementary_dataset.csv          # Per-molecule PCE values used for Movie 1
│   ├── figures/                           # pce_histogram, mean_discovery_curves, mean_best_pce_discovered, calibration_curves (.png)
│   ├── movies/                            # figure1/2/3 *_movie.mp4 supplementary animations
│   └── fp_cache/                          # Cached preprocessed feature matrices (.npy), one per representation
│
├── docs/
│   ├── paper_draft.md                     # Manuscript draft
│   ├── literature/                        # Scopus literature-search exports used for the review
│   └── references/                        # Key reference PDF (Abroshan et al., Front. Chem. 2021)
│
├── environment.yml                        # Conda environment
├── LICENSE                                # MIT License
└── README.md
```

## Installation

Tested on Ubuntu with Python 3.10+ (developed with Python 3.12). A Conda environment is provided (`conda env create -f environment.yml`); alternatively, the pip commands below install the packages actually imported by the code.

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

- **`data/processed/Combined_df.csv`** — the dataset used directly by the pipeline. 1,481 rows, columns `SMILES` (molecular structure) and `PCE` (experimentally measured power conversion efficiency, %), range 0.01%–18.38%. This is the file `active_learning_benchmark_700dpi.py` loads at startup (`CONFIG["data_file"]`).
- **`data/processed/combined_df.smi`** — the same 1,481 molecules as a plain-text SMILES list.
- **`data/raw/First_Database.csv`** and **`data/raw/train_test_combine.csv`** — the two raw, literature-assembled source tables (donor/acceptor SMILES with measured PCE) that `notebooks/Active_Learning_Data_Generation.ipynb` deduplicates, canonicalizes, and concatenates to produce `Combined_df.csv`. Both are included in the repo, so no external download is required to reproduce `Combined_df.csv` from scratch.
- **Precomputed descriptor/fingerprint CSVs** in `data/features/` (`rdkit_combined_df.csv`, `mordred_combined_df.csv`, `maccs_combined_df.csv`, `morgan_combined_df.csv`, `avalon_combined.csv`, `apf_combined.csv`, `pf_combined_df.csv`, `extended_combined_df.csv`) — one row per molecule in `Combined_df.csv`, in the same order, with per-representation feature columns. The main script loads these on first run and caches the preprocessed arrays to `results/fp_cache/*.npy`; if a cache/CSV is missing it recomputes the representation directly from SMILES with RDKit/Mordred.
- The exact original literature sources (papers/databases) from which `First_Database.csv` and `train_test_combine.csv` were assembled are not recorded in the code: `<DATASET SOURCE / DOI>`.

## Running the pipeline end-to-end

1. **(Optional — only if regenerating the dataset from raw sources)** Run `notebooks/Active_Learning_Data_Generation.ipynb` top to bottom (from inside `notebooks/`). Reads `First_Database.csv` and `train_test_combine.csv`, deduplicates by canonical SMILES (keeping the highest PCE per molecule), and writes `Combined_df.csv`. Skip this step if you only want to reproduce the paper's results, since `Combined_df.csv` is already committed.
2. **Run the main benchmark:**
   ```bash
   python src/active_learning_benchmark_700dpi.py
   ```
   Loads `Combined_df.csv`; computes (or loads from `data/features/` / `results/fp_cache/*.npy`) all eight fingerprint/descriptor representations; runs the GPR + EI active learning loop for each representation across 5 random seeds × 100 iterations from a 10-molecule seed set; trains the GCN + MC-Dropout + EI baseline across the same 5 seeds × 100 iterations; saves numeric results to `results/` and figures to `results/figures/`. This is the script that produced the results currently committed in `results/` (`active_learning_benchmark.py` is an earlier variant that writes a combined two-panel `discovery_curves.png` instead).
3. **(Optional) Generate the supplementary movies:**
   ```bash
   python src/make_supplementary_movies_revised.py \
       --json results/al_results.json \
       --dataset results/supplementary_dataset.csv \
       --outdir results/movies
   ```
   Requires `results/al_results.json` (produced in step 2) and `results/supplementary_dataset.csv` (already committed). Produces the three `.mp4` animations under `results/movies/`.

The exploratory notebooks in `notebooks/exploratory/` (`Active_Learning_Rdkit*.ipynb`, `Active_Learning_*_fingerprint.ipynb`, `Active_Learning_Mordred.ipynb`, `Graph_Active_Learning.ipynb`, `HBE_BayesOpt.ipynb`) are single-representation prototypes that were consolidated and fixed up into `src/active_learning_benchmark_700dpi.py` (see the "Issues addressed vs. individual notebooks" section in that script's docstring). They are not required to reproduce the paper's results and are kept for audit purposes only.

## Expected outputs

A successful run of `src/active_learning_benchmark_700dpi.py` writes the following to `results/`:

- `figures/pce_histogram.png` — Figure 1: PCE distribution across the 1,481-molecule dataset.
- `figures/mean_discovery_curves.png` — Figure 2: mean ± SD fraction of the top-10% PCE pool recovered vs. AL iteration, per representation, with a random-selection baseline.
- `figures/mean_best_pce_discovered.png` — Figure 3: mean ± SD best PCE found vs. oracle calls, per representation.
- `figures/calibration_curves.png` — GPR predictive-uncertainty calibration diagnostic, per representation.
- `al_results.json` — raw per-representation, per-seed arrays (`discovery_curve`, `best_pce_curve`, `calibration_curve`, `random_discovery`), each of length 100 (one value per AL iteration).
- `fp_cache/*.npy` — cached preprocessed feature matrices, one file per representation (`APF`, `Avalon`, `Extended`, `MACCS`, `Mordred`, `Morgan`, `PF`, `RDKit`).
- A console summary table (printed at the end of the run) ranking representations by final discovery performance.

Running `src/make_supplementary_movies_revised.py` additionally produces `figure1_pce_histogram_movie.mp4`, `figure2_discovery_curve_movie.mp4`, and `figure3_best_pce_movie.mp4` in the target `--outdir` (committed copies are in `results/movies/`).

## Requirements

- **Hardware:** CPU-only is sufficient — the GPR surrogate (scikit-learn) and the GCN baseline (2 graph-convolution layers, hidden dim 64, 60 epochs, batch size 32) are both small enough to train on CPU; a GPU will speed up the GCN portion but is not required.
- **RAM:** Mordred computes ~1,800 descriptors per molecule before PCA reduction (capped at 50 components / 95% explained variance in `CONFIG`), which is the most memory-intensive step. `<RAM REQUIREMENT — fill in based on your test run, e.g. "peak ~X GB observed on my machine">`.
- **Runtime:** The full run repeats the AL loop for 8 representations × 5 seeds × 100 iterations, plus the GCN baseline × 5 seeds × 100 iterations, with GPR retrained from scratch at every iteration. First run is slower because uncached representations (especially Mordred) are computed from SMILES and written to `results/fp_cache/`; subsequent runs reuse the cache. `<APPROXIMATE TOTAL RUNTIME — fill in from your hardware, e.g. "~X hours on an N-core CPU">`.

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

## Contact

<YOUR NAME>, <AFFILIATION> — <EMAIL>
