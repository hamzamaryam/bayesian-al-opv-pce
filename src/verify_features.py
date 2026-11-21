#!/usr/bin/env python
"""
verify_features.py
==================
Audit the provenance of every handcrafted representation used in the paper.

For each representation this script
  1. regenerates the fingerprint from SMILES with the settings stated in the
     manuscript / config (Morgan r=1, Extended r=2, PF = PatternFingerprint,
     APF = 4096-bit atom-pair generator, Avalon 512, MACCS 167) and reports the
     fraction of molecules whose bit vector is identical to data/features/*.csv;
  2. re-applies the benchmark preprocessing (variance filter -> correlation
     filter for <= 300 features -> standardisation -> PCA, 95 % capped at 50
     components) to the CSV and checks it against results/fp_cache/*.npy;
  3. reports the dimensionality after each step, the PCA variance retained,
     the number of components a full 95 % would need, and the number of
     distinct feature vectors among the 1,481 molecules.

Results are printed and written to results/feature_provenance.json.
Needs: numpy, pandas, scikit-learn, rdkit (no torch).

    python src/verify_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
P = json.loads((REPO / "config" / "benchmark_config.json").read_text())["parameters"]

REGISTRY = {  # name: (csv, columns to drop, correlation filter allowed)
    "RDKit":    ("rdkit_combined_df.csv",    ["Unnamed: 0"],      True),
    "MACCS":    ("maccs_combined_df.csv",    ["SMILES", "PCE"],   True),
    "Morgan":   ("morgan_combined_df.csv",   ["SMILES", "PCE"],   False),
    "APF":      ("apf_combined.csv",         ["SMILES", "PCE"],   False),
    "PF":       ("pf_combined_df.csv",       ["SMILES", "PCE"],   False),
    "Extended": ("extended_combined_df.csv", ["SMILES", "PCE"],   False),
    "Avalon":   ("avalon_combined.csv",      ["SMILES", "PCE"],   False),
    "Mordred":  ("mordred_combined_df.csv",  ["Unnamed: 0"],      False),
}


def fingerprint_fns():
    from rdkit import Chem, RDLogger
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as G
    from rdkit.Chem.rdmolops import PatternFingerprint
    RDLogger.DisableLog("rdApp.*")
    morgan = G.GetMorganGenerator(radius=P["morgan_radius"], fpSize=P["morgan_n_bits"])
    ext = G.GetMorganGenerator(radius=P["extended_radius"], fpSize=P["extended_n_bits"])
    apf = G.GetAtomPairGenerator(fpSize=P["apf_n_bits"])
    bits = lambda fp: np.array(list(fp), dtype=np.int8)
    return Chem, {
        "MACCS":    lambda m: bits(MACCSkeys.GenMACCSKeys(m)),
        "Morgan":   lambda m: bits(morgan.GetFingerprint(m)),
        "Extended": lambda m: bits(ext.GetFingerprint(m)),
        "APF":      lambda m: bits(apf.GetFingerprint(m)),
        "PF":       lambda m: bits(PatternFingerprint(m, fpSize=P["pf_n_bits"])),
        "Avalon":   lambda m: bits(pyAvalonTools.GetAvalonFP(m, nBits=P["avalon_n_bits"])),
    }


def preprocess(df, corr_allowed):
    X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)
    fmax = np.finfo(np.float32).max
    X = X.clip(-fmax, fmax).values.astype(np.float32)
    info = {"raw_dim": int(X.shape[1])}
    X = X[:, VarianceThreshold(P["variance_threshold"]).fit(X).get_support()]
    info["after_variance_filter"] = int(X.shape[1])
    if corr_allowed and X.shape[1] <= 300:
        d = pd.DataFrame(X)
        c = d.corr().abs()
        up = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1))
        drop = [k for k in up.columns if up[k].max() > P["correlation_threshold"]]
        X = d[[k for k in d.columns if k not in drop]].values
        info["after_correlation_filter"] = int(X.shape[1])
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > P["pca_max_components"]:
        full = np.cumsum(PCA(svd_solver="full").fit(X).explained_variance_ratio_)
        info["components_for_full_target"] = int(np.searchsorted(full, P["pca_explained_variance"]) + 1)
        n = min(P["pca_max_components"], X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n, random_state=P["global_seed"])
        Xp = pca.fit_transform(X)
        cv = np.cumsum(pca.explained_variance_ratio_)
        k = min(int(np.searchsorted(cv, P["pca_explained_variance"]) + 1), n)
        X = Xp[:, :k]
        info["pca_variance_retained"] = round(float(cv[k - 1]), 4)
    info["gpr_input_dim"] = int(X.shape[1])
    return X.astype(np.float32), info


def main():
    smiles = pd.read_csv(REPO / "data" / "processed" / "Combined_df.csv")["SMILES"].tolist()
    Chem, fns = fingerprint_fns()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    report = {}
    for name, (csv, drop, corr) in REGISTRY.items():
        df = pd.read_csv(REPO / "data" / "features" / csv, low_memory=False)
        feats = df.drop(columns=[c for c in drop if c in df.columns])
        entry = {"csv": csv}
        if name in fns:
            regen = np.array([fns[name](m) for m in mols])
            csv_bits = feats.values.astype(np.int8)
            entry["regenerated_rows_identical"] = (
                round(float((regen == csv_bits).all(1).mean()), 4)
                if regen.shape == csv_bits.shape else f"shape mismatch {regen.shape} vs {csv_bits.shape}")
        X, info = preprocess(feats, corr)
        entry.update(info)
        cache = REPO / "results" / "fp_cache" / f"{name}.npy"
        # distinct feature vectors among the molecules, counted on the array the
        # GPR actually saw (the cache) when available
        D = np.load(cache) if cache.exists() else X
        entry["distinct_inputs"] = int(len(np.unique(np.round(D, 5), axis=0)))
        if cache.exists():
            C = np.load(cache)
            if C.shape == X.shape:
                r = [abs(np.corrcoef(C[:, i], X[:, i])[0, 1]) for i in range(C.shape[1])]
                entry["cache_matches_csv_min_abs_r"] = round(float(np.min(r)), 4)
            else:
                entry["cache_matches_csv_min_abs_r"] = f"shape mismatch {C.shape} vs {X.shape}"
        report[name] = entry
        print(f"{name:9s} " + ", ".join(f"{k}={v}" for k, v in entry.items() if k != "csv"))
    out = REPO / "results" / "feature_provenance.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
