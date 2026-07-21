#!/usr/bin/env python
"""
active_learning_benchmark.py
==============================
Bayesian Active Learning benchmark for OSC PCE discovery.

Benchmarks 8 molecular fingerprint representations + a GCN baseline within
a unified GPR + Expected Improvement (EI) active learning loop.

Dataset  : Combined_df.csv  (1,481 molecules, SMILES + PCE)
Paper    : "Benchmarking Molecular Representations in Bayesian Active Learning
            for Accelerated Discovery of High-Efficiency Organic Solar Cells"

Issues addressed vs. individual notebooks
------------------------------------------
  - DtypeWarning in avalon_combined.csv           → recompute from SMILES via pyAvalonTools
  - mordred Missing/error descriptor values        → coerce to numeric + fillna(0)
  - float32 overflow in descriptor matrices        → explicit clipping before GPR
  - GPR kernel crash on high-dimensional inputs   → PCA (≤ 50 components, 95 % var)
  - correlation filter too slow for large FPs      → only applied to FPs with < 300 features
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
DATA_DIR = Path(__file__).parent

CONFIG = {
    # ---- paths ----
    "data_file":    DATA_DIR / "Combined_df.csv",
    "results_dir":  DATA_DIR / "al_results",
    "smiles_col":   "SMILES",
    "target_col":   "PCE",

    # ---- dataset / evaluation ----
    "top_k_threshold": 12.0,      # PCE > 12 % → "top-10 %" of 1 481 molecules
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

    # ---- fingerprint settings ----
    "morgan_radius":   2,
    "morgan_n_bits":   2048,
    "avalon_n_bits":   512,
    "fp_n_bits":       2048,       # APF / PF / Extended shared size

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
    """2-D RDKit physicochemical descriptors (~200 features after filtering)."""
    desc_names = [d[0] for d in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            rows.append(list(calc.CalcDescriptors(mol)) if mol else [np.nan] * len(desc_names))
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


def compute_morgan(smiles_list: list, radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """Morgan circular fingerprint (ECFP4-equivalent)."""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "Morgan", n_bits)


def compute_atom_pair_fp(smiles_list: list, n_bits: int = 2048) -> pd.DataFrame:
    """Atom-pair fingerprint – encodes atom-pair topology + distance."""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "APF", n_bits)


def compute_path_fp(smiles_list: list, n_bits: int = 2048) -> pd.DataFrame:
    """RDKit path-based (topological) fingerprint."""
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits) if mol else None
            rows.append(list(fp) if fp else [0] * n_bits)
        except Exception:
            rows.append([0] * n_bits)
    return _fp_to_df(rows, "PF", n_bits)


def compute_extended_fp(smiles_list: list, radius: int = 3, n_bits: int = 2048) -> pd.DataFrame:
    """Extended Morgan fingerprint at radius-3 (broader neighbourhood)."""
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
    Avalon fingerprint via pyAvalonTools.

    This recomputes from SMILES directly, avoiding the DtypeWarning that
    arises when loading the pre-computed avalon_combined.csv (which mixes
    string SMILES and integer bit columns with no explicit dtype).
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
    Mordred 2-D descriptor set (~1 600+ features).

    mordred returns 'mordred.error.Missing' objects for failed descriptors.
    These are coerced to NaN in to_float32_safe() and then filled with 0.
    """
    calc = MordredCalc(mordred_desc, ignore_3D=True)
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
        "fn": lambda s: compute_morgan(s, CONFIG["morgan_radius"], CONFIG["morgan_n_bits"]),
        "corr_filter": False,
        "csv_file": "morgan_combined_df.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "APF": {
        "fn": lambda s: compute_atom_pair_fp(s, CONFIG["fp_n_bits"]),
        "corr_filter": False,
        "csv_file": "apf_combined.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "PF": {
        "fn": lambda s: compute_path_fp(s, CONFIG["fp_n_bits"]),
        "corr_filter": False,
        "csv_file": "pf_combined_df.csv",
        "drop_cols": ["SMILES", "PCE"],
    },
    "Extended": {
        "fn": lambda s: compute_extended_fp(s, radius=3, n_bits=CONFIG["fp_n_bits"]),
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

def build_gpr() -> GaussianProcessRegressor:
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

    for itr in range(CONFIG["n_iterations"]):
        X_train = X_full[labeled]
        y_train = y_full[labeled]
        X_pool  = X_full[unlabeled]
        y_pool  = y_full[unlabeled]

        # ---- Train GPR ----
        gpr = build_gpr()
        try:
            gpr.fit(X_train, y_train)
            mu_pool, sigma_pool = gpr.predict(X_pool, return_std=True)
        except Exception as exc:
            # Fallback to random selection if GPR fails (rare with PCA preprocessing)
            if verbose:
                print(f"    [WARN] GPR failed at iter {itr}: {exc}. Random fallback.")
            chosen_local = int(rng.integers(len(unlabeled)))
            chosen_global = unlabeled.pop(chosen_local)
            labeled.append(chosen_global)
            discovery_curve.append(top_k_mask[labeled].sum() / n_top)
            calibration_curve.append(np.nan)
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

    return {
        "discovery_curve":   discovery_curve,
        "calibration_curve": calibration_curve,
        "best_pce_curve":    best_pce_curve,
        "random_discovery":  random_discovery,
    }


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

def save_results(all_results: dict, results_dir: Path):
    out = {}
    for rep, seeds_data in all_results.items():
        out[rep] = {}
        for seed, data in seeds_data.items():
            out[rep][str(seed)] = {
                k: [float(v) for v in vals] if isinstance(vals, list) else vals
                for k, vals in data.items()
            }
    path = results_dir / "al_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {path}")


def plot_discovery_curves(all_results: dict, results_dir: Path):
    """Save two separate figures: discovery rate, and best PCE found vs oracle calls."""
    colors = plt.cm.tab10.colors
    iters = np.arange(1, CONFIG["n_iterations"] + 1)

    rep_names = list(all_results.keys())

    # ===== Figure 1: mean discovery curves =====
    fig, ax = plt.subplots(figsize=(7, 5))
    random_plotted = False

    for idx, rep in enumerate(rep_names):
        seeds_data = all_results[rep]
        color = colors[idx % len(colors)]

        dc_all = np.array([
            v["discovery_curve"] for v in seeds_data.values()
            if "discovery_curve" in v and len(v["discovery_curve"]) == CONFIG["n_iterations"]
        ])
        if dc_all.size:
            mu, sd = dc_all.mean(0), dc_all.std(0)
            ax.plot(iters, mu, label=rep, color=color, lw=1.8)
            ax.fill_between(iters, mu - sd, mu + sd, alpha=0.12, color=color)

        # -- Random baseline (plot once from first rep that has it) --
        if not random_plotted:
            rd_all = np.array([
                v["random_discovery"] for v in seeds_data.values()
                if "random_discovery" in v and len(v["random_discovery"]) == CONFIG["n_iterations"]
            ])
            if rd_all.size:
                mu_r, sd_r = rd_all.mean(0), rd_all.std(0)
                ax.plot(iters, mu_r, "k--", label="Random", lw=1.5)
                ax.fill_between(iters, mu_r - sd_r, mu_r + sd_r, alpha=0.08, color="k")
                random_plotted = True

    ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=9)
    ax.set_ylabel("Fraction of Top-10% Molecules Discovered", fontsize=9)
    ax.set_title("Discovery Curves — Molecular Representation Benchmark", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, CONFIG["n_iterations"])
    #### ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=1.0, color="gray", ls=":", lw=1)

    plt.tight_layout()
    path = results_dir / "mean_discovery_curves.png"
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    print(f"Discovery plot saved → {path}")

    # ===== Figure 2: mean best PCE discovered =====
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, rep in enumerate(rep_names):
        seeds_data = all_results[rep]
        color = colors[idx % len(colors)]

        bp_all = np.array([
            v["best_pce_curve"] for v in seeds_data.values()
            if "best_pce_curve" in v and len(v["best_pce_curve"]) == CONFIG["n_iterations"]
        ])
        if bp_all.size:
            mu, sd = bp_all.mean(0), bp_all.std(0)
            ax.plot(iters, mu, label=rep, color=color, lw=1.8)
            ax.fill_between(iters, mu - sd, mu + sd, alpha=0.12, color=color)

    ax.set_xlabel("Oracle Calls (Active Learning Iterations)", fontsize=9)
    ax.set_ylabel("Best PCE Found (%)", fontsize=9)
    ax.set_title("Best PCE Discovered vs. Oracle Calls", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, CONFIG["n_iterations"])
    #### ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = results_dir / "mean_best_pce_discovered.png"
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    print(f"Best-PCE plot saved → {path}")


def plot_calibration_curves(all_results: dict, results_dir: Path):
    """Mean NLPD (surrogate calibration) vs iteration for each representation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    iters = np.arange(1, CONFIG["n_iterations"] + 1)

    for idx, (rep, seeds_data) in enumerate(all_results.items()):
        cal_all = []
        for v in seeds_data.values():
            cc = v.get("calibration_curve", [])
            if len(cc) == CONFIG["n_iterations"]:
                clean = [x if (x is not None and not np.isnan(x)) else np.nan for x in cc]
                cal_all.append(clean)

        if not cal_all:
            continue

        arr = np.array(cal_all, dtype=float)
        mu  = np.nanmean(arr, axis=0)
        sd  = np.nanstd(arr, axis=0)
        color = colors[idx % len(colors)]
        ax.plot(iters, mu, label=rep, color=color, lw=1.8)
        ax.fill_between(iters, mu - sd, mu + sd, alpha=0.12, color=color)

    ax.set_xlabel("Oracle Calls", fontsize=9)
    ax.set_ylabel("Mean NLPD (lower = better calibration)", fontsize=9)
    ax.set_title("Surrogate Calibration over Active Learning Iterations", fontsize=11)
    ax.legend(fontsize=9)
    #### ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = results_dir / "calibration_curves.png"
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved → {path}")


def plot_pce_histogram(y_full: np.ndarray, results_dir: Path):
    """PCE distribution plot for Figure 1 in the paper."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_full, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(CONFIG["top_k_threshold"], color="crimson", ls="--", lw=1.8,
               label=f"Top-10% threshold (PCE > {CONFIG['top_k_threshold']}%)")
    ax.set_xlabel("Power Conversion Efficiency, PCE (%)", fontsize=11)
    ax.set_ylabel("Number of Molecules", fontsize=11)
    ax.set_title("Distribution of PCE Values in the 1 481-Molecule Dataset", fontsize=11)
    ax.legend(fontsize=9)
    #### ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = results_dir / "pce_histogram.png"
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    print(f"PCE histogram saved → {path}")


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
                aucs.append(np.trapz(dc) / len(dc))

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
    print(f"  Top-K (> {CONFIG['top_k_threshold']}%): "
          f"{top_k_mask.sum()} molecules ({100*top_k_mask.mean():.1f} %)")

    # Generate PCE histogram (Figure 1)
    plot_pce_histogram(y_full, results_dir)

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

        csv_path = DATA_DIR / fp_cfg.get("csv_file", "")
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
    plot_discovery_curves(all_results, results_dir)
    plot_calibration_curves(all_results, results_dir)
    print_summary_table(all_results)

    print(f"\nAll experiments complete.")
    print(f"Results directory: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
