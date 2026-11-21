#!/usr/bin/env python
"""
active_learning_benchmark_700dpi.py   (canonical pipeline script)
====================================
Bayesian Active Learning benchmark for OSC PCE discovery.

* 8 handcrafted representations (RDKit, Mordred, MACCS, Avalon, Morgan,
  Extended, APF, PF) are each coupled to the SAME GPR surrogate
  (Constant x RBF + White kernel) and Expected Improvement (EI) acquisition.
* The GCN baseline does NOT use a GPR and its embeddings are NOT passed to a
  GPR: the GCN is trained end-to-end and its predictive mean / uncertainty
  come from MC-Dropout (15 stochastic passes). These are plugged into the
  same EI formula, with the same xi, seed sets and query budget.

Representations actually used (= data/features/*.csv, see notebooks/
Active_Learning_Data_Generation.ipynb and src/verify_features.py):
    Morgan   : Morgan radius 1 (ECFP2-type), 2048 bits
    Extended : Morgan radius 2 (ECFP4-type), 2048 bits
    Avalon   : 512-bit hashed SUBSTRUCTURE fingerprint (not circular)
    APF      : RDKit atom-pair generator, 4096 bits
    PF       : RDKit PatternFingerprint (SMARTS substructure patterns), 2048 bits
    MACCS    : 167 keys;  RDKit: 210 descriptors (explicit H);  Mordred: 1826
The precomputed CSVs are loaded when present; the compute_* functions below
are a fallback and have been aligned with these settings.

"High-efficiency pool" = PCE > 12 % = 370 of 1,481 molecules (top 25 %).
(Earlier versions mislabelled this pool "top-10 %".)

Dataset  : data/processed/Combined_df.csv  (1,481 molecules, SMILES + PCE)
Paper    : "Circular fingerprints outpace learned graph representations in
            active-learning discovery of organic semiconductor materials"
Parameters: config/benchmark_config.json (overrides the CONFIG defaults below)

Issues addressed vs. individual notebooks
------------------------------------------
  - DtypeWarning in avalon_combined.csv           → read with low_memory=False
  - mordred Missing/error descriptor values        → coerce to numeric + fillna(0)
  - float32 overflow in descriptor matrices        → explicit clipping before GPR
  - GPR kernel crash on high-dimensional inputs   → PCA (95 % var, CAPPED at 50 components;
                                                    the cap means APF/Avalon/PF retain < 95 %)
  - correlation filter too slow for large FPs      → only applied to FPs with ≤ 300 features
                                                    after the variance filter (RDKit, MACCS)
  - GCN notebook was incomplete / not trained      → full training loop + MC-Dropout EI
  - no systematic multi-rep comparison             → unified evaluation with fixed seeds
  - repeated computation on re-runs                → cache preprocessed arrays to disk
"""

# ==============================================================================
# SECTION 1 – IMPORTS
# ==============================================================================
import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # non-interactive backend – works without display
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- Global figure-export settings ---------------------------------------
# All plots are saved as PNG at 700 dpi for publication-quality output.
PLOT_DPI    = 700
PLOT_FORMAT = "png"
plt.rcParams["savefig.dpi"]    = PLOT_DPI
plt.rcParams["savefig.format"] = PLOT_FORMAT
plt.rcParams["figure.dpi"]     = 100  # on-screen / figure rendering dpi
# --------------------------------------------------------------------------

# --- RDKit ---
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdmolops import PatternFingerprint
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Avalon import pyAvalonTools            # confirmed available

# --- Mordred ---
from mordred import Calculator as MordredCalc
from mordred import descriptors as mordred_desc   # confirmed available

# --- Scikit-learn ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as CK, WhiteKernel
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error

# --- PyTorch + PyTorch Geometric ---
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader  # PyG 2.6
from torch_geometric.nn import GCNConv, global_mean_pool

# Silence noisy RDKit / PyG logs
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# SECTION 2 – CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent   # scripts live in src/
DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"

CONFIG = {
    # ---- paths ----
    "data_file":    DATA_DIR / "processed" / "Combined_df.csv",
    "results_dir":  REPO_ROOT / "results",
    "smiles_col":   "SMILES",
    "target_col":   "PCE",

    # ---- dataset / evaluation ----
    "top_k_threshold": 12.0,      # PCE > 12 % → high-efficiency pool: 370 of 1,481 (top 25 %)
    "n_seed":          10,         # initial labeled seed molecules
    "n_iterations":    100,        # AL iterations per run
    "n_random_seeds":  5,          # independent experimental repeats

    # ---- preprocessing ----
    "variance_threshold":   0.10,
    "correlation_threshold": 0.80,
    "pca_explained_variance": 0.95,
    "pca_max_components":    50,   # cap to keep GPR tractable

    # ---- GPR ----
    "gpr_n_restarts":  5,
    "gpr_alpha":       1e-6,       # nugget for numerical stability
    "ei_jitter":       0.01,       # ξ in EI formula
    "gpr_random_state": None,      # None reproduces the paper's runs (unseeded optimizer
                                   # restarts); set an int for bit-reproducible reruns
    "log_gpr_kernel":  True,       # store fitted kernel hyperparameters per iteration

    # ---- fingerprint settings (match data/features/*.csv used for the paper) ----
    "morgan_radius":   1,          # "Morgan"   = Morgan radius 1 (ECFP2-type)
    "morgan_n_bits":   2048,
    "extended_radius": 2,          # "Extended" = Morgan radius 2 (ECFP4-type)
    "extended_n_bits": 2048,
    "avalon_n_bits":   512,        # Avalon: hashed substructure FP (NOT circular)
    "apf_n_bits":      4096,       # RDKit atom-pair generator
    "pf_n_bits":       2048,       # RDKit PatternFingerprint (substructure patterns)

    # ---- GCN ----
    "gcn_hidden_dim":  64,
    "gcn_epochs":      60,
    "gcn_batch_size":  32,
    "gcn_lr":          1e-2,
    "gcn_mc_passes":   15,         # MC-Dropout forward passes for uncertainty
    "gcn_dropout":     0.20,

    # ---- reproducibility ----
    "global_seed": 42,
}

# Machine-readable parameter file (single source of truth for the paper's settings).
CONFIG_FILE = REPO_ROOT / "config" / "benchmark_config.json"
if CONFIG_FILE.exists():
    _params = json.loads(CONFIG_FILE.read_text())["parameters"]
    _unknown = set(_params) - set(CONFIG)
    if _unknown:
        raise KeyError(f"Unknown keys in {CONFIG_FILE}: {sorted(_unknown)}")
    CONFIG.update(_params)

# Atom types observed in the full 1 481-molecule dataset
ATOM_TYPES = ["Al", "B", "Br", "C", "Cl", "F", "Ge", "I",
              "N",  "O", "P",  "S", "Se", "Si", "Te", "Zn"]

# ==============================================================================
# SECTION 3 – UTILITY HELPERS
# ==============================================================================

def mol_from_smiles(smi: str):
    """Parse SMILES safely; return None on failure."""
    try:
        return Chem.MolFromSmiles(str(smi).strip())
    except Exception:
        return None


def to_float32_safe(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a descriptor DataFrame to a float32 numpy array.

    Steps
    -----
    1. Coerce every column to numeric – mordred 'Missing' objects and any
       other non-numeric values become NaN.
    2. Fill NaN / inf with 0.
    3. Clip to float32 range to prevent overflow.
    """
    arr = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr = arr.replace([np.inf, -np.inf], 0.0)
    fmax = np.finfo(np.float32).max
    arr = arr.clip(-fmax, fmax)
    return arr.values.astype(np.float32)


# ==============================================================================
# SECTION 4 – FINGERPRINT COMPUTATION
# ==============================================================================

def _fp_to_df(rows, prefix, n_bits):
    return pd.DataFrame(rows, columns=[f"{prefix}_{i}" for i in range(n_bits)])


def compute_rdkit_descriptors(smiles_list: list) -> pd.DataFrame:
    """All 2-D RDKit descriptors (210 in RDKit 2024.03), computed on
    hydrogen-explicit molecules exactly as in the data-generation notebook."""
    desc_names = [d[0] for d in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            rows.append(list(calc.CalcDescriptors(Chem.AddHs(mol))) if mol
                        else [np.nan] * len(desc_names))
        except Exception:
            rows.append([np.nan] * len(desc_names))
    return pd.DataFrame(rows, columns=desc_names)


def compute_maccs_keys(smiles_list: list) -> pd.DataFrame:
    """MACCS keys – 167-bit substructure fingerprint."""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            rows.append(list(MACCSkeys.GenMACCSKeys(mol)) if mol else [0] * 167)
        except Exception:
            rows.append([0] * 167)
    return _fp_to_df(rows, "MACCS", 167)


def compute_morgan(smiles_list: list, radius: int = 1, n_bits: int = 2048) -> pd.DataFrame:
    """Morgan circular fingerprint; radius 1 (ECFP2-type) as used for "Morgan"."""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "Morgan", n_bits)


def compute_atom_pair_fp(smiles_list: list, n_bits: int = 4096) -> pd.DataFrame:
    """Atom-pair fingerprint (RDKit atom-pair generator, 4096 bits as used in the paper)."""
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = gen.GetFingerprint(mol) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "APF", n_bits)


def compute_pattern_fp(smiles_list: list, n_bits: int = 2048) -> pd.DataFrame:
    """RDKit PatternFingerprint ("PF"): substructure-screening fingerprint built
    from generic SMARTS patterns. (Not a path-based RDKFingerprint.)"""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = PatternFingerprint(mol, fpSize=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "PF", n_bits)


def compute_extended_fp(smiles_list: list, radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """"Extended" = Morgan fingerprint at radius 2 (ECFP4-type): the larger
    neighbourhood relative to the radius-1 "Morgan" entry."""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "Extended", n_bits)


def compute_avalon_fp(smiles_list: list, n_bits: int = 512) -> pd.DataFrame:
    """
    Avalon fingerprint via pyAvalonTools: a 512-bit HASHED SUBSTRUCTURE
    fingerprint (atoms, bonds, rings, short paths). It is not circular.
    Used only if data/features/avalon_combined.csv is absent.
    """
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "Avalon", n_bits)


def compute_mordred_descriptors(smiles_list: list) -> pd.DataFrame:
    """
    Mordred descriptor set (1,826 columns), computed as in the notebook with
    ignore_3D=False but without 3-D conformers, so 3-D descriptors are
    'Missing'. Missing values are coerced to NaN in to_float32_safe(), filled
    with 0, and the resulting constant columns are removed by the variance
    filter.
    """
    calc = MordredCalc(mordred_desc, ignore_3D=False)
    mols = [mol_from_smiles(smi) for smi in smiles_list]
    df = calc.pandas(mols, quiet=True)
    # Keep column names for downstream variance/correlation filtering
    return df


# ==============================================================================
# SECTION 5 – PREPROCESSING PIPELINE
# ==============================================================================

def remove_low_variance(X: np.ndarray, threshold: float = 0.10):
    sel = VarianceThreshold(threshold)
    sel.fit(X)
    return X[:, sel.get_support()], sel.get_support()


def remove_high_correlation(X: np.ndarray, threshold: float = 0.80):
    """Drop one member of each highly-correlated pair (only for small FPs)."""
    df = pd.DataFrame(X)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if upper[c].max() > threshold]
    keep = [c for c in df.columns if c not in to_drop]
    return df[keep].values, keep


def preprocess_features(
    raw_df: pd.DataFrame,
    apply_corr_filter: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline returning a float32 numpy array.

    Order of operations
    -------------------
    1. Coerce to float32, clip overflow, fill NaN/inf → (N, D_raw)
    2. Variance threshold filter                       → (N, D_var)
    3. Correlation filter  [only if apply_corr_filter] → (N, D_corr)
    4. Standardise (fit on entire pool – justified
       because only PCE labels, not structures, are
       withheld in the AL setting)                     → (N, D_corr)
    5. PCA to ≤ 50 components retaining 95 % variance  → (N, D_pca)

    Returns
    -------
    np.ndarray  shape (N, D_pca)
    """
    X = to_float32_safe(raw_df)
    print(f"    raw shape          : {X.shape}")

    # -- Step 2: variance filter --
    X, _ = remove_low_variance(X, threshold=CONFIG["variance_threshold"])
    print(f"    after var filter   : {X.shape}")

    # -- Step 3: correlation filter (only for compact FPs) --
    if apply_corr_filter and X.shape[1] <= 300:
        X, _ = remove_high_correlation(X, threshold=CONFIG["correlation_threshold"])
        print(f"    after corr filter  : {X.shape}")

    # -- Step 4: standardise --
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -- Step 5: PCA (only when dimensionality exceeds PCA cap) --
    if X.shape[1] > CONFIG["pca_max_components"]:
        n_comp = min(CONFIG["pca_max_components"], X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=CONFIG["global_seed"])
        X = pca.fit_transform(X)
        # Trim to explained variance target
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, CONFIG["pca_explained_variance"]) + 1)
        n_keep = min(n_keep, n_comp)
        X = X[:, :n_keep]
        print(f"    after PCA          : {X.shape}  "
              f"({cumvar[n_keep-1]*100:.1f} % variance retained)")
        if cumvar[n_keep - 1] < CONFIG["pca_explained_variance"]:
            print(f"    [NOTE] {CONFIG['pca_max_components']}-component cap reached: "
                  f"only {cumvar[n_keep-1]*100:.1f} % of variance retained "
                  f"(target {CONFIG['pca_explained_variance']*100:.0f} %).")

    return X.astype(np.float32)


# ==============================================================================
# SECTION 6 – FINGERPRINT REGISTRY
# ==============================================================================
# Each entry: callable that takes smiles_list and returns a pd.DataFrame,
# plus a flag for the correlation filter.

FINGERPRINT_REGISTRY: dict = {
    "RDKit": {
        "fn": compute_rdkit_descriptors,
        "corr_filter": True,         # small FP → correlation filtering sensible
        "csv_file": "rdkit_combined_df.csv",
        "drop_cols": ["Unnamed: 0"],
    },
    "MACCS": {
        "fn": compute_maccs_keys,
        "corr_filter": True,
        "csv_file": "maccs_combined_df.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "Morgan": {
        "fn": lambda s: compute_morgan(s, CONFIG["morgan_radius"], CONFIG["morgan_n_bits"]),  # radius 1
        "corr_filter": False,
        "csv_file": "morgan_combined_df.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "APF": {
        "fn": lambda s: compute_atom_pair_fp(s, CONFIG["apf_n_bits"]),
        "corr_filter": False,
        "csv_file": "apf_combined.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "PF": {
        "fn": lambda s: compute_pattern_fp(s, CONFIG["pf_n_bits"]),
        "corr_filter": False,
        "csv_file": "pf_combined_df.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "Extended": {
        "fn": lambda s: compute_extended_fp(s, CONFIG["extended_radius"], CONFIG["extended_n_bits"]),
        "corr_filter": False,
        "csv_file": "extended_combined_df.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "Avalon": {
        "fn": lambda s: compute_avalon_fp(s, CONFIG["avalon_n_bits"]),
        "corr_filter": False,
        "csv_file": "avalon_combined.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "Mordred": {
        "fn": compute_mordred_descriptors,
        "corr_filter": False,        # too many features for pairwise correlation
        "csv_file": "mordred_combined_df.csv",
        "drop_cols": ["Unnamed: 0"],
    },
}


# ==============================================================================
# SECTION 7 – SURROGATE MODEL & ACQUISITION FUNCTION
# ==============================================================================

def build_gpr(random_state=None) -> GaussianProcessRegressor:
    """
    GPR with Constant × RBF + White noise kernel.
    normalize_y=True improves numerical conditioning on PCE targets.
    """
    kernel = (
        CK(1.0, constant_value_bounds=(1e-4, 1e2))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=CONFIG["gpr_n_restarts"],
        normalize_y=True,
        alpha=CONFIG["gpr_alpha"],
        random_state=random_state,
    )


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    EI(x) = (μ(x) − f* − ξ) Φ(Z) + σ(x) φ(Z)
    where Z = (μ(x) − f* − ξ) / σ(x).
    """
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - f_best - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-9] = 0.0
    return ei


def mean_nlpd(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Mean Negative Log Predictive Density – measures surrogate calibration.
    Lower is better.
    """
    var = sigma ** 2 + 1e-9
    nlpd = 0.5 * (np.log(2.0 * np.pi * var) + (y_true - mu) ** 2 / var)
    return float(np.nanmean(nlpd))


# ==============================================================================
# SECTION 8 – GPR ACTIVE LEARNING LOOP
# ==============================================================================

def run_gpr_al_loop(
    X_full: np.ndarray,
    y_full: np.ndarray,
    top_k_mask: np.ndarray,
    seed: int,
    rep_name: str,
    verbose: bool = True,
) -> dict:
    """
    Run one complete GPR + EI active learning experiment.

    Parameters
    ----------
    X_full      : (N, D) preprocessed feature matrix for all molecules
    y_full      : (N,)   PCE values
    top_k_mask  : (N,)   boolean – True where PCE > threshold
    seed        : random seed for this run
    rep_name    : name of fingerprint representation (for logging)

    Returns
    -------
    dict with keys:
        discovery_curve   – list[float], fraction of top-K found per iteration
        calibration_curve – list[float], mean NLPD per iteration
        best_pce_curve    – list[float], best PCE found so far per iteration
        random_discovery  – list[float], discovery curve for random baseline
    """
    rng = np.random.default_rng(seed)
    N = len(y_full)
    n_top = int(top_k_mask.sum())

    # ---- Initialise pools ----
    all_idx = np.arange(N)
    seed_idx = rng.choice(N, size=CONFIG["n_seed"], replace=False)
    labeled  = list(seed_idx)
    unlabeled = [i for i in all_idx if i not in set(labeled)]

    # ---- Parallel random baseline (same seed → same initial set) ----
    rng_rand = np.random.default_rng(seed)
    rand_labeled   = list(rng_rand.choice(N, size=CONFIG["n_seed"], replace=False))
    rand_unlabeled = [i for i in all_idx if i not in set(rand_labeled)]

    discovery_curve   = []
    calibration_curve = []
    best_pce_curve    = []
    random_discovery  = []
    kernel_log        = []   # fitted [constant, length_scale, noise_level] per iteration
    gpr_seed = (None if CONFIG["gpr_random_state"] is None
                else int(CONFIG["gpr_random_state"]) + seed)

    for itr in range(CONFIG["n_iterations"]):
        X_train = X_full[labeled]
        y_train = y_full[labeled]
        X_pool  = X_full[unlabeled]
        y_pool  = y_full[unlabeled]

        # ---- Train GPR ----
        gpr = build_gpr(random_state=gpr_seed)
        try:
            gpr.fit(X_train, y_train)
            mu_pool, sigma_pool = gpr.predict(X_pool, return_std=True)
            kernel_log.append(np.exp(gpr.kernel_.theta).tolist())
        except Exception as exc:
            # Fallback to random selection if GPR fails (rare with PCA preprocessing)
            if verbose:
                print(f"    [WARN] GPR failed at iter {itr}: {exc}. Random fallback.")
            chosen_local = int(rng.integers(len(unlabeled)))
            chosen_global = unlabeled.pop(chosen_local)
            labeled.append(chosen_global)
            discovery_curve.append(top_k_mask[labeled].sum() / n_top)
            calibration_curve.append(np.nan)
            kernel_log.append([np.nan, np.nan, np.nan])
            best_pce_curve.append(float(y_full[labeled].max()))
            rand_global = rand_unlabeled.pop(int(rng_rand.integers(len(rand_unlabeled))))
            rand_labeled.append(rand_global)
            random_discovery.append(top_k_mask[rand_labeled].sum() / n_top)
            continue

        # ---- EI acquisition ----
        f_best = float(y_train.max())
        ei = expected_improvement(mu_pool, sigma_pool, f_best, xi=CONFIG["ei_jitter"])
        chosen_local  = int(np.argmax(ei))
        chosen_global = unlabeled.pop(chosen_local)
        labeled.append(chosen_global)

        # ---- Record metrics ----
        discovery_curve.append(top_k_mask[labeled].sum() / n_top)
        calibration_curve.append(mean_nlpd(mu_pool, sigma_pool, y_pool))
        best_pce_curve.append(float(y_full[labeled].max()))

        # ---- Random baseline step ----
        rand_global = rand_unlabeled.pop(int(rng_rand.integers(len(rand_unlabeled))))
        rand_labeled.append(rand_global)
        random_discovery.append(top_k_mask[rand_labeled].sum() / n_top)

        if verbose and (itr + 1) % 25 == 0:
            pct = 100 * discovery_curve[-1]
            print(f"    [{rep_name}] seed={seed} iter={itr+1:3d} | "
                  f"top-K found: {pct:.1f}% | best PCE: {best_pce_curve[-1]:.2f}%")

    out = {
        "discovery_curve":   discovery_curve,
        "calibration_curve": calibration_curve,
        "best_pce_curve":    best_pce_curve,
        "random_discovery":  random_discovery,
    }
    if CONFIG["log_gpr_kernel"]:
        out["gpr_kernel_params"] = kernel_log
    return out


# ==============================================================================
# SECTION 9 – GCN MODEL + ACTIVE LEARNING LOOP
# ==============================================================================

class GCNRegressorWithDropout(torch.nn.Module):
    """
    Two-layer GCN with dropout for MC uncertainty estimation.
    Input node features: one-hot atom type (16 dims) +
                         degree (1) + formal charge (1) + is_aromatic (1) = 19 dims.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.drop  = torch.nn.Dropout(p=dropout)
        self.fc1   = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2   = torch.nn.Linear(hidden_dim // 2, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, ei))
        x = self.drop(x)
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(self.drop(x)))
        return self.fc2(x).squeeze(-1)


def smiles_to_pyg(smi: str, y_val: float = None) -> Data | None:
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        onehot = [1 if atom.GetSymbol() == t else 0 for t in ATOM_TYPES]
        onehot += [
            atom.GetDegree() / 10.0,
            atom.GetFormalCharge() / 4.0,
            int(atom.GetIsAromatic()),
        ]
        node_feats.append(onehot)

    x = torch.tensor(node_feats, dtype=torch.float)

    # Edges (bidirectional)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j];  dst += [j, i]

    if not src:        # isolated atom(s) – skip
        return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    d = Data(x=x, edge_index=edge_index)
    if y_val is not None:
        d.y = torch.tensor([y_val], dtype=torch.float)
    return d


def build_pyg_dataset(smiles_list: list, y_values: np.ndarray):
    """Convert all SMILES to PyG Data objects; return list + original indices."""
    data_list, orig_idx = [], []
    for i, (smi, y) in enumerate(zip(smiles_list, y_values)):
        d = smiles_to_pyg(smi, float(y))
        if d is not None:
            data_list.append(d)
            orig_idx.append(i)
    return data_list, orig_idx


def train_gcn_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def mc_predict(model, loader, device, n_passes: int = 15):
    """
    MC-Dropout inference: run n_passes stochastic forward passes.
    Returns (mu, sigma) for every molecule in loader.
    """
    model.train()   # keep dropout active at inference time
    all_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pass_preds = []
            for batch in loader:
                batch = batch.to(device)
                pass_preds.extend(model(batch).cpu().numpy().tolist())
            all_preds.append(pass_preds)
    arr = np.array(all_preds)              # (n_passes, N_pool)
    return arr.mean(axis=0), arr.std(axis=0)


def run_gcn_al_loop(
    smiles_list: list,
    y_full: np.ndarray,
    top_k_mask: np.ndarray,
    seed: int,
    verbose: bool = True,
) -> dict:
    """
    GCN + MC-Dropout + EI active learning loop.
    A fresh GCN is trained from scratch at each iteration on the growing
    labeled set (simulating an oracle-in-the-loop setting).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    all_pyg, orig_idx = build_pyg_dataset(smiles_list, y_full)
    orig_idx_arr = np.array(orig_idx)
    N_valid = len(all_pyg)
    n_top = int(top_k_mask.sum())
    input_dim = all_pyg[0].x.shape[1]

    labeled   = list(rng.choice(N_valid, size=CONFIG["n_seed"], replace=False))
    unlabeled = [i for i in range(N_valid) if i not in set(labeled)]

    discovery_curve  = []
    best_pce_curve   = []

    for itr in range(CONFIG["n_iterations"]):
        # ---- Train GCN on labeled set ----
        train_loader = PyGDataLoader(
            [all_pyg[i] for i in labeled],
            batch_size=CONFIG["gcn_batch_size"], shuffle=True,
        )
        model = GCNRegressorWithDropout(
            input_dim, CONFIG["gcn_hidden_dim"], CONFIG["gcn_dropout"]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["gcn_lr"])

        for _ in range(CONFIG["gcn_epochs"]):
            train_gcn_one_epoch(model, train_loader, optimizer, device)

        # ---- MC-Dropout uncertainty over unlabeled pool ----
        pool_loader = PyGDataLoader(
            [all_pyg[i] for i in unlabeled],
            batch_size=CONFIG["gcn_batch_size"], shuffle=False,
        )
        mu_pool, sigma_pool = mc_predict(
            model, pool_loader, device, n_passes=CONFIG["gcn_mc_passes"]
        )

        # ---- EI acquisition ----
        labeled_pce = np.array([all_pyg[i].y.item() for i in labeled])
        f_best = float(labeled_pce.max())
        ei = expected_improvement(mu_pool, sigma_pool, f_best, xi=CONFIG["ei_jitter"])
        chosen_local  = int(np.argmax(ei))
        chosen_global = unlabeled.pop(chosen_local)
        labeled.append(chosen_global)

        # ---- Record metrics ----
        # Map local GCN indices back to original molecule indices
        orig_labeled = orig_idx_arr[labeled].tolist()
        n_found = int(top_k_mask[orig_labeled].sum())
        discovery_curve.append(n_found / n_top)
        best_pce_curve.append(float(y_full[orig_labeled].max()))

        if verbose and (itr + 1) % 25 == 0:
            print(f"    [GCN] seed={seed} iter={itr+1:3d} | "
                  f"top-K found: {100*discovery_curve[-1]:.1f}% | "
                  f"best PCE: {best_pce_curve[-1]:.2f}%")

    return {
        "discovery_curve": discovery_curve,
        "best_pce_curve":  best_pce_curve,
    }


# ==============================================================================
# SECTION 10 – RESULTS: SAVING, PLOTTING, SUMMARY TABLE
# ==============================================================================

def _jsonable(v):
    """Convert (nested) lists / numpy scalars to plain floats for JSON."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer, float, int)):
        return float(v)
    return v


def save_results(all_results: dict, results_dir: Path):
    out = {}
    for rep, seeds_data in all_results.items():
        out[rep] = {}
        for seed, data in seeds_data.items():
            out[rep][str(seed)] = {k: _jsonable(vals) for k, vals in data.items()}
    path = results_dir / "al_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {path}")


# Figures are produced by src/make_figures.py (also usable stand-alone from
# results/al_results.json). Labels use "high-efficiency pool (PCE > 12 %)";
# NLPD is plotted as mean and median over seeds on a log axis.
import make_figures  # noqa: E402  (same directory as this script)


_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # NumPy 1.x / 2.x


def print_summary_table(all_results: dict):
    """Hits@50 and normalised AUC for every representation."""
    header = f"{'Representation':<15} {'Hits@50 (mean±std)':<26} {'AUC (mean±std)':<24}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for rep, seeds_data in all_results.items():
        hits50, aucs = [], []
        for data in seeds_data.values():
            dc = data.get("discovery_curve", [])
            if len(dc) >= 50:
                hits50.append(dc[49])
            if dc:
                aucs.append(_trapz(dc) / len(dc))

        h_mu = np.mean(hits50) if hits50 else float("nan")
        h_sd = np.std(hits50)  if hits50 else float("nan")
        a_mu = np.mean(aucs)   if aucs   else float("nan")
        a_sd = np.std(aucs)    if aucs   else float("nan")

        print(f"{rep:<15} {h_mu:.3f} ± {h_sd:.3f}              "
              f"{a_mu:.3f} ± {a_sd:.3f}")

    print("=" * len(header))


# ==============================================================================
# SECTION 11 – MAIN ORCHESTRATION
# ==============================================================================

def main():
    np.random.seed(CONFIG["global_seed"])
    torch.manual_seed(CONFIG["global_seed"])

    results_dir: Path = CONFIG["results_dir"]
    fp_cache_dir: Path = results_dir / "fp_cache"
    results_dir.mkdir(parents=True, exist_ok=True)
    fp_cache_dir.mkdir(exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 11.1  Load dataset
    # ------------------------------------------------------------------
    print("=" * 65)
    print("LOADING DATASET")
    print("=" * 65)
    df = pd.read_csv(CONFIG["data_file"])
    smiles_list = df[CONFIG["smiles_col"]].astype(str).tolist()
    y_full      = df[CONFIG["target_col"]].values.astype(np.float32)
    top_k_mask  = y_full > CONFIG["top_k_threshold"]

    print(f"  Molecules     : {len(df)}")
    print(f"  PCE range     : {y_full.min():.2f} – {y_full.max():.2f} %")
    print(f"  Mean ± std    : {y_full.mean():.2f} ± {y_full.std():.2f} %")
    print(f"  High-efficiency pool (PCE > {CONFIG['top_k_threshold']}%): "
          f"{top_k_mask.sum()} molecules ({100*top_k_mask.mean():.1f} %)")

    # Generate PCE histogram (Figure 1)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    make_figures.plot_histogram(np.asarray(y_full, dtype=float), results_dir / "figures")

    # ------------------------------------------------------------------
    # 11.2  Compute / load all fingerprint arrays
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("LOADING / COMPUTING MOLECULAR FINGERPRINTS")
    print("=" * 65)

    fingerprint_arrays: dict = {}

    for fp_name, fp_cfg in FINGERPRINT_REGISTRY.items():
        cache_path = fp_cache_dir / f"{fp_name}.npy"

        if cache_path.exists():
            print(f"\n[{fp_name}]  loading from cache …")
            X = np.load(cache_path)
            print(f"  cached shape: {X.shape}")
            fingerprint_arrays[fp_name] = X
            continue

        csv_path = FEATURES_DIR / fp_cfg.get("csv_file", "")
        if fp_cfg.get("csv_file") and csv_path.exists():
            print(f"\n[{fp_name}]  loading from precomputed CSV …")
            t0 = time.time()
            try:
                raw_df = pd.read_csv(csv_path, low_memory=False)
                drop = [c for c in fp_cfg.get("drop_cols", []) if c in raw_df.columns]
                raw_df = raw_df.drop(columns=drop)
                X = preprocess_features(
                    raw_df,
                    apply_corr_filter=fp_cfg.get("corr_filter", False),
                )
                np.save(cache_path, X)
                print(f"  final shape: {X.shape}  ({time.time()-t0:.1f} s)")
                fingerprint_arrays[fp_name] = X
            except Exception as exc:
                print(f"  [ERROR] {fp_name} CSV load failed: {exc}")
                fingerprint_arrays[fp_name] = None
            continue

        print(f"\n[{fp_name}]  computing fingerprints …")
        t0 = time.time()
        try:
            raw_df = fp_cfg["fn"](smiles_list)
            X = preprocess_features(
                raw_df,
                apply_corr_filter=fp_cfg.get("corr_filter", False),
            )
            np.save(cache_path, X)
            print(f"  final shape: {X.shape}  ({time.time()-t0:.1f} s)")
            fingerprint_arrays[fp_name] = X
        except Exception as exc:
            print(f"  [ERROR] {fp_name} failed: {exc}")
            fingerprint_arrays[fp_name] = None

    # ------------------------------------------------------------------
    # 11.3  GPR Active Learning loop – all fingerprint representations
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RUNNING GPR + EI ACTIVE LEARNING (fingerprint representations)")
    print("=" * 65)

    all_results: dict = {}

    for fp_name, X_full in fingerprint_arrays.items():
        if X_full is None:
            print(f"\n[SKIP] {fp_name} – fingerprint unavailable.\n")
            continue

        print(f"\n{'─'*50}")
        print(f" Representation : {fp_name}  |  feature dim : {X_full.shape[1]}")
        print(f"{'─'*50}")
        all_results[fp_name] = {}

        for seed in range(CONFIG["n_random_seeds"]):
            result = run_gpr_al_loop(
                X_full, y_full, top_k_mask,
                seed=seed,
                rep_name=fp_name,
                verbose=True,
            )
            all_results[fp_name][seed] = result

    # ------------------------------------------------------------------
    # 11.4  GCN Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RUNNING GCN + MC-DROPOUT + EI ACTIVE LEARNING (graph baseline)")
    print("=" * 65)

    all_results["GCN"] = {}
    for seed in range(CONFIG["n_random_seeds"]):
        print(f"\n[GCN]  seed {seed}/{CONFIG['n_random_seeds']-1}")
        result = run_gcn_al_loop(
            smiles_list, y_full, top_k_mask,
            seed=seed, verbose=True,
        )
        all_results["GCN"][seed] = result

    # ------------------------------------------------------------------
    # 11.5  Save results & generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SAVING RESULTS & GENERATING PLOTS")
    print("=" * 65)

    save_results(all_results, results_dir)
    fig_dir = results_dir / "figures"
    make_figures.plot_discovery(all_results, fig_dir)
    make_figures.plot_best_pce(all_results, fig_dir)
    make_figures.plot_calibration(all_results, fig_dir)
    make_figures.plot_nlpd_per_seed(all_results, fig_dir)
    print_summary_table(all_results)

    print(f"\nAll experiments complete.")
    print(f"Results directory: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
