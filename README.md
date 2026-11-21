# Circular fingerprints outpace learned graph representations in active-learning discovery of organic semiconductor materials

Code, data and results for the article of the same title (H. Siddiqui, *ACS Omega*, manuscript ao-2026-11833c, under revision).

## Overview

This repository benchmarks how the **molecular representation** affects pool-based Bayesian active learning (AL) for discovering high power-conversion-efficiency (PCE) organic solar cell (OSC) molecules. Given a fixed pool of 1,481 molecules with known PCE, which representation lets an AL agent recover high-efficiency candidates in the fewest oracle queries?

* **Eight handcrafted representations** (table below) are each coupled to the **same** Gaussian process regressor (GPR; Constant × RBF + White kernel) and the **same** Expected Improvement (EI) acquisition function.
* A **graph convolutional network (GCN) baseline** is evaluated with the same EI formula, exploration parameter, initial seed sets and query budget, but it does **not** use a GPR and its embeddings are **not** passed to a GPR. The GCN is trained end-to-end on the labeled molecules, and its predictive mean and uncertainty come from **MC-Dropout** (15 stochastic forward passes).
* The **high-efficiency pool** is every molecule with PCE > 12 %: **370 of 1,481 molecules (25.0 %)**. Earlier versions of the code and manuscript called this pool "top-10 %"; that label was wrong, because the 90th percentile of PCE is 14.3 %.

## Representations actually used

These settings are those of the precomputed matrices in `data/features/`, which produced every reported result. `src/verify_features.py` regenerates them from SMILES and checks them against the cache. The output is in `results/feature_provenance.json`.

| Name in paper/code | What it is | Generator (RDKit 2024.03.5) | Raw dim. | GPR input dim. (PCA variance retained) |
|---|---|---|---|---|
| RDKit | 2-D descriptors, explicit H | `MolecularDescriptorCalculator(Descriptors._descList)` on `Chem.AddHs(mol)` | 210 | 32 (95.5 %) |
| Mordred | descriptors | `mordred.Calculator(descriptors, ignore_3D=False)`, no conformers (3-D → missing → removed) | 1,826 | 41 (95.2 %) |
| MACCS | substructure keys | `MACCSkeys.GenMACCSKeys` | 167 | 18 (no PCA) |
| Avalon | **hashed substructure** FP (not circular) | `pyAvalonTools.GetAvalonFP(mol, nBits=512)` | 512 | 50 (**81.8 %**) |
| Morgan | circular, **radius 1** (ECFP2-type) | `GetMorganFingerprintAsBitVect(mol, radius=1, nBits=2048)` | 2,048 | 26 (95.0 %) |
| Extended | circular, **radius 2** (ECFP4-type) | `GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)` | 2,048 | 50 (95.2 %) |
| APF | atom pairs | `rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096)` | **4,096** | 50 (**69.1 %**) |
| PF | **pattern** (SMARTS substructure-screening) FP, not path-based | `Chem.rdmolops.PatternFingerprint(mol)` | 2,048 | 50 (**87.9 %**) |

**Preprocessing**, identical for all eight representations:

1. Coerce to numeric; NaN and inf become 0.
2. Remove features with variance < 0.10.
3. Remove one of each pair with |r| > 0.80, only when ≤ 300 features remain (RDKit, MACCS).
4. Standardise.
5. Apply PCA to 95 % variance, **capped at 50 components**.

Because of the cap, APF, Avalon and PF retain less than 95 % of their variance. All of these steps are unsupervised and are fitted once on the whole pool.

## Repository structure

```
.
├── config/
│   └── benchmark_config.json          # Machine-readable parameters (loaded by the main script)
├── data/
│   ├── raw/
│   │   ├── First_Database.csv         # 1,558 donor:acceptor device records (320 donors, 763 acceptors) with literature reference ("Ref.")
│   │   └── train_test_combine.csv     # 944 molecule–PCE records (912 unique SMILES)
│   ├── processed/
│   │   ├── Combined_df.csv            # 1,481 molecules: [index, SMILES, PCE]
│   │   └── combined_df.smi
│   ├── features/                      # Precomputed feature matrices used for the paper (row order = Combined_df.csv)
│   │   └── padel_legacy/, ttf_combined.csv, rdk_fingerprint_combined.csv   # not used by the main script
│   └── padel_descriptor_xml/
├── src/
│   ├── active_learning_benchmark_700dpi.py   # MAIN PIPELINE – produced results/
│   ├── make_figures.py                       # Re-draws all figures from results/al_results.json (no AL re-run)
│   ├── make_supplementary_movies_revised.py  # Supplementary movies S1–S3
│   ├── verify_features.py                    # Feature provenance / cache audit
│   └── legacy/                               # Superseded scripts (kept for audit, not used)
├── notebooks/
│   ├── Active_Learning_Data_Generation.ipynb # Builds Combined_df.csv and all feature matrices
│   └── exploratory/                          # Single-representation prototypes (audit only; settings may differ)
├── results/
│   ├── al_results.json                # Per-representation, per-seed curves (the paper's results)
│   ├── feature_provenance.json        # Output of src/verify_features.py
│   ├── figures/, movies/, fp_cache/
│   └── supplementary_dataset.csv
├── docs/literature/, docs/references/
├── environment.yml
└── LICENSE
```

## Dataset construction (`notebooks/Active_Learning_Data_Generation.ipynb`)

1. **`train_test_combine.csv`** (944 records). Rows with an identical SMILES string are collapsed into the row with the **highest PCE**, leaving 912.
2. **`First_Database.csv`** (1,558 donor:acceptor device records). The **acceptor** SMILES and the device PCE are kept. Rows with an identical acceptor SMILES are collapsed into the **highest PCE** over all donors and devices, leaving 763.
3. The two tables are concatenated (1,675 rows), and every SMILES is converted to **RDKit canonical** form. This reveals 194 duplicate structures, which are again collapsed into the **highest PCE**, leaving **1,481** unique molecules.

Each molecule's label is therefore the **best PCE reported for any device containing it**. It is not an average, and it does not encode the donor partner or the processing conditions.

> **TODO (author):** cite the original publication or database from which `train_test_combine.csv` was compiled. `First_Database.csv` already records a reference for each device.

## Installation

Developed with Python 3.12.5. Key versions: RDKit 2024.03.5, Mordred 1.2.0, scikit-learn 1.5.1, NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.2, PyTorch 2.4.1, PyTorch Geometric 2.6.0, Matplotlib 3.9.2. See `environment.yml`.

```bash
git clone https://github.com/hamzamaryam/bayesian-al-opv-pce.git
cd bayesian-al-opv-pce
conda env create -f environment.yml
conda activate active_learning
```

A minimal pip alternative:

```bash
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.14.1 scikit-learn==1.5.1 matplotlib==3.9.2 rdkit==2024.3.5 mordred==1.2.0
pip install torch==2.4.1 torch_geometric==2.6.0
pip install imageio imageio-ffmpeg   # optional: supplementary movies
```

## Reproducing the paper

1. **Benchmark** (8 representations × 5 seeds × 100 iterations, plus the GCN):

   ```bash
   python src/active_learning_benchmark_700dpi.py
   ```

   Parameters are read from `config/benchmark_config.json`. The seeds are {0, 1, 2, 3, 4}, and the same initial sets are used for every representation, the GCN and the random baseline. The script writes `results/al_results.json` (it now also stores the fitted GPR hyperparameters per iteration) and the figures. `gpr_random_state` is `null` by default, which reproduces the paper's unseeded optimizer restarts. Set it to an integer for bit-reproducible reruns.

2. **Figures only**, from the stored results with no AL re-run:

   ```bash
   python src/make_figures.py
   ```

   This produces Figure 1 (`pce_histogram`), Figure 2 (`mean_discovery_curves`), Figure 3 (`mean_best_pce_discovered`), Figure 4 (`calibration_curves`: mean and median NLPD over seeds, log axis) and Figure S2 (`nlpd_per_seed`).

3. **Supplementary movies**:

   ```bash
   python src/make_supplementary_movies_revised.py --json results/al_results.json --dataset results/supplementary_dataset.csv --outdir results/movies
   ```

4. **Feature audit**:

   ```bash
   python src/verify_features.py
   ```

## Notes for interpreting the results

* **NLPD** is computed on the current unlabeled pool, and only for the GPR runs; it was not recorded for the GCN. The large mean NLPD of MACCS (until iteration 86) and Morgan (until iteration 51) comes from **one and two individual seeds**, not from a systematic early-iteration effect (see `results/figures/nlpd_per_seed.png`). These two representations have the fewest distinct feature vectors after preprocessing: 352 for MACCS and 960 for Morgan, out of 1,481.
* **Discovery** is the fraction of the 370-molecule high-efficiency pool (PCE > 12 %) contained in the labeled set. **AUC** is the trapezoidal area under that curve divided by the number of iterations.

## Hardware

CPU-only is sufficient. The GPR is refitted on at most 110 points with at most 50 input dimensions. The GCN is small: 2 GCNConv layers of 64 units, 60 epochs, retrained at each iteration. The first run spends the most time computing uncached features, especially Mordred; later runs reuse `results/fp_cache/`.

## Citation

```bibtex
@article{Siddiqui_AL_OSC,
  author  = {Siddiqui, Hamza},
  title   = {Circular fingerprints outpace learned graph representations in active-learning discovery of organic semiconductor materials},
  journal = {ACS Omega},
  note    = {Manuscript ao-2026-11833c, under revision},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).

## Contact

Hamza Siddiqui, Solar Development Cell, Integral University, Lucknow, India (corresponding author of the article).
