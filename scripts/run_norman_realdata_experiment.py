#!/usr/bin/env python3
"""Norman Perturb-seq diagnostics on one shared processed data set.

Unless --no-normalize or --self-test is given, cells are library-size
normalized to 1e4 counts and log1p transformed. The most variable genes are
chosen from control and single-perturbation cells only. Three diagnostics
then run on the same genes and context means:

* additive baseline: predict each held-out double mean by mu_i + mu_j - mu_0
  and report R^2 relative to the control predictor mu_0;
* direction/magnitude decomposition of that additive prediction, from
  direction_magnitude_decomposition.py;
* low-rank check: after PCA fitted to control and single cells, estimate a
  Ledoit-Wolf precision matrix per context and compute rho_2 of
  Theta_i - Theta_0, the share of its squared singular values in the top two.

Accepted inputs are a processed .h5ad file, the four GEO GSE133344 filtered
files in one directory, or --self-test, which generates synthetic data. The
data file is not bundled with the repository.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from experiment_utils import write_config, file_sha256
from direction_magnitude_decomposition import evaluate_pairs, summarize as summarize_direction, make_plot

CONTROL = {"ctrl", "control", "non-targeting", "nontargeting", "non_targeting", "nt", "ntc", "negctrl", "negative", "negative_control", "mock", "empty", "unperturbed", "none", "nan", "na", "null", "safe", "no_guide"}


@dataclass
class Data:
    X: object
    labels: np.ndarray
    genes: np.ndarray
    source: str


def is_control(token: str) -> bool:
    value = token.strip().lower().replace("_", "-")
    return value in CONTROL or value.startswith("ctrl") or value.startswith("control") or "non-target" in value


def parse_targets(label: object) -> tuple[str, ...]:
    text = str(label).strip()
    if not text or is_control(text):
        return tuple()
    parts = re.split(r"\+|/|\||__|--|,|;", text)
    targets = []
    for part in parts:
        part = re.sub(r"^sg", "", part.strip(), flags=re.I)
        if part and not is_control(part) and part.upper() not in {x.upper() for x in targets}:
            targets.append(part)
    return tuple(targets)


def infer_label_column(obs: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in obs:
            raise ValueError(f"Missing label column {requested!r}; available: {list(obs.columns)}")
        return requested
    preferred = ["guide_merged", "condition", "perturbation", "perturbation_name", "guide_identity", "gene", "target", "cov_drug_dose_name"]
    for column in preferred:
        if column in obs:
            parsed = [parse_targets(x) for x in obs[column].astype(str)]
            lengths = {len(x) for x in parsed}
            if 0 in lengths and 1 in lengths:
                return column
    candidates = []
    for column in obs.columns:
        if pd.api.types.is_object_dtype(obs[column]) or isinstance(obs[column].dtype, pd.CategoricalDtype):
            parsed = [parse_targets(x) for x in obs[column].astype(str)]
            lengths = {len(x) for x in parsed}
            candidates.append((int(0 in lengths) + int(1 in lengths) + int(2 in lengths), column))
    if not candidates:
        raise ValueError("Could not infer perturbation labels; pass --label-column")
    return max(candidates)[1]


def load_h5ad(path: Path, label_column: Optional[str], layer: Optional[str], *,
              n_genes: int = 2000, normalize: bool = True, seed: int = 0,
              target_separator: Optional[str] = None, block_size: int = 256) -> Data:
    """Normalize and select genes in column blocks, then load the selected matrix."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install anndata to read .h5ad files") from exc
    adata = ad.read_h5ad(path, backed="r")
    try:
        column = infer_label_column(adata.obs, label_column)
        matrix = adata.layers[layer] if layer else adata.X
        labels = adata.obs[column].astype(str).to_numpy()
        _, kinds = context_arrays(labels, target_separator)
        rows = np.flatnonzero((kinds == 'control') | (kinds == 'single'))
        if not len(rows):
            raise ValueError('No control or single-perturbation cells; check labels')
        if len(rows) > 50000:
            rows = np.random.default_rng(seed).choice(rows, 50000, replace=False)

        factors = None
        if normalize:
            totals = np.zeros(adata.n_obs, dtype=np.float64)
            for start in range(0, adata.n_vars, block_size):
                block = matrix[:, start:start + block_size]
                totals += np.asarray(block.sum(axis=1)).ravel()
            totals[totals <= 0] = 1
            factors = (1e4 / totals).astype(np.float32)

        def read_columns(columns):
            block = matrix[:, columns]
            if factors is None:
                return block
            if sparse.issparse(block):
                block = block.tocsr().astype(np.float32, copy=False)
                block.data *= np.repeat(factors, np.diff(block.indptr))
                np.log1p(block.data, out=block.data)
            else:
                block = np.asarray(block, dtype=np.float32) * factors[:, None]
                np.log1p(block, out=block)
            return block

        variance = np.empty(adata.n_vars, dtype=np.float64)
        for start in range(0, adata.n_vars, block_size):
            stop = min(start + block_size, adata.n_vars)
            block = read_columns(slice(start, stop))[rows]
            if sparse.issparse(block):
                mean = np.asarray(block.mean(axis=0)).ravel()
                mean2 = np.asarray(block.multiply(block).mean(axis=0)).ravel()
            else:
                mean, mean2 = block.mean(axis=0), (block ** 2).mean(axis=0)
            variance[start:stop] = np.maximum(mean2 - mean**2, 0)
        selected = np.argsort(variance)[-min(n_genes, adata.n_vars):][::-1]
        blocks = [read_columns(np.sort(selected[start:start + block_size]))
                  for start in range(0, len(selected), block_size)]
        loaded_order = np.concatenate([np.sort(selected[start:start + block_size])
                                       for start in range(0, len(selected), block_size)])
        positions = {gene: i for i, gene in enumerate(loaded_order)}
        order = [positions[gene] for gene in selected]
        X = sparse.hstack(blocks, format='csr') if sparse.issparse(blocks[0]) else np.hstack(blocks)
        return Data(X=X[:, order], labels=labels,
                    genes=np.asarray(adata.var_names.astype(str))[selected],
                    source=f"h5ad:{path.name}; label={column}")
    finally:
        adata.file.close()


def geo_file(directory: Path, suffix: str) -> Path:
    exact = directory / f"GSE133344_{suffix}"
    if exact.exists():
        return exact
    matches = list(directory.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely locate *{suffix} in {directory}")


def load_geo(directory: Path) -> Data:
    matrix_path = geo_file(directory, "filtered_matrix.mtx.gz")
    genes_path = geo_file(directory, "filtered_genes.tsv.gz")
    barcodes_path = geo_file(directory, "filtered_barcodes.tsv.gz")
    identities_path = geo_file(directory, "filtered_cell_identities.csv.gz")
    print("Reading GEO sparse matrix...", file=sys.stderr)
    matrix = mmread(matrix_path).tocsr()
    genes_frame = pd.read_csv(genes_path, sep="\t", header=None)
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str).to_numpy()
    identities = pd.read_csv(identities_path)
    barcode_col = next((c for c in identities if "barcode" in c.lower() or c.lower() == "cell"), None)
    label_col = next((c for c in identities if c != barcode_col and any(k in c.lower() for k in ["guide", "gene", "perturb", "identity", "condition"])), None)
    label_col = label_col or identities.columns[-1]
    if barcode_col:
        mapping = dict(zip(identities[barcode_col].astype(str), identities[label_col].astype(str)))
        labels = np.array([mapping.get(x, "unknown") for x in barcodes])
    else:
        labels = identities[label_col].astype(str).to_numpy()
    n_genes, n_cells = len(genes_frame), len(barcodes)
    if matrix.shape == (n_genes, n_cells):
        matrix = matrix.T.tocsr()
    elif matrix.shape != (n_cells, n_genes):
        raise ValueError(f"Unexpected matrix shape {matrix.shape}")
    gene_col = 1 if genes_frame.shape[1] > 1 else 0
    return Data(matrix, labels, genes_frame.iloc[:, gene_col].astype(str).to_numpy(), f"GEO:GSE133344; label={label_col}")


def normalize_log1p(X, target_sum: float = 1e4):
    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(np.float32)
        totals = np.asarray(X.sum(axis=1)).ravel()
        totals[totals <= 0] = 1
        X = sparse.diags(target_sum / totals).dot(X).tocsr()
        X.data = np.log1p(X.data)
        return X
    X = np.asarray(X, dtype=np.float32).copy()
    totals = X.sum(axis=1)
    totals[totals <= 0] = 1
    X *= (target_sum / totals)[:, None]
    return np.log1p(X)


def select_variable_genes(X, n_genes: int, max_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.arange(X.shape[0])
    if len(rows) > max_cells:
        rows = rng.choice(rows, max_cells, replace=False)
    sub = X[rows]
    if sparse.issparse(sub):
        mean = np.asarray(sub.mean(axis=0)).ravel()
        mean2 = np.asarray(sub.multiply(sub).mean(axis=0)).ravel()
    else:
        mean, mean2 = np.mean(sub, axis=0), np.mean(np.asarray(sub) ** 2, axis=0)
    variance = np.maximum(mean2 - mean**2, 0)
    n_genes = min(n_genes, len(variance))
    return np.argsort(variance)[-n_genes:][::-1]


def group_mean(X, mask: np.ndarray) -> np.ndarray:
    sub = X[mask]
    return np.asarray(sub.mean(axis=0)).ravel() if sparse.issparse(sub) else np.asarray(sub).mean(axis=0)


def r2_vs_control(true: np.ndarray, pred: np.ndarray, control: np.ndarray) -> float:
    denominator = float(np.sum((true - control) ** 2))
    return np.nan if denominator <= 1e-12 else 1 - float(np.sum((true - pred) ** 2)) / denominator


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    return np.nan if np.std(x) < 1e-12 or np.std(y) < 1e-12 else float(pearsonr(x, y).statistic)


def context_arrays(labels: np.ndarray, target_separator: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    if target_separator:
        labels = np.array([str(x).replace(target_separator, '+') for x in labels])
    targets = [parse_targets(x) for x in labels]
    names = np.array(["+".join(x) if x else "control" for x in targets], dtype=object)
    kinds = np.array(["control" if len(x) == 0 else "single" if len(x) == 1 else "double" if len(x) == 2 else "other" for x in targets], dtype=object)
    return names, kinds


def additive_analysis(X, names: np.ndarray, kinds: np.ndarray, outdir: Path, min_control: int, min_single: int, min_double: int) -> pd.DataFrame:
    control_mask = kinds == "control"
    if control_mask.sum() < min_control:
        raise ValueError(f"Only {control_mask.sum()} control cells; check labels")
    mu0 = group_mean(X, control_mask)
    singles = {}
    counts = {}
    for name in sorted(set(names[kinds == "single"])):
        mask = (kinds == "single") & (names == name)
        if mask.sum() >= min_single:
            singles[str(name)] = group_mean(X, mask)
            counts[str(name)] = int(mask.sum())
    rows = []
    examples = []
    for name in sorted(set(names[kinds == "double"])):
        genes = str(name).split("+")
        if len(genes) != 2 or any(g not in singles for g in genes):
            continue
        mask = (kinds == "double") & (names == name)
        if mask.sum() < min_double:
            continue
        true = group_mean(X, mask)
        pred = singles[genes[0]] + singles[genes[1]] - mu0
        effect = np.abs(true - mu0)
        top = np.argsort(effect)[-min(100, len(effect)):]
        r2 = r2_vs_control(true, pred, mu0)
        rows.append({
            "pair": name, "gene_1": genes[0], "gene_2": genes[1],
            "n_double_cells": int(mask.sum()), "n_single_1_cells": counts[genes[0]], "n_single_2_cells": counts[genes[1]],
            "r2_mean_additive_vs_control": r2,
            "pearson_mean_additive": safe_corr(true, pred),
            "rmse_control": math.sqrt(float(np.mean((true - mu0) ** 2))),
            "rmse_additive": math.sqrt(float(np.mean((true - pred) ** 2))),
            "r2_top100_additive_vs_control": r2_vs_control(true[top], pred[top], mu0[top]),
        })
        examples.append((str(name), true, pred, r2))
    result = pd.DataFrame(rows).sort_values("r2_mean_additive_vs_control", ascending=False) if rows else pd.DataFrame()
    result.to_csv(outdir / "results/double_prediction_metrics.csv", index=False)
    if len(result):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(result.r2_mean_additive_vs_control, bins=max(5, min(20, len(result))))
        ax.axvline(0, linestyle="--", label="control baseline")
        ax.axvline(result.r2_mean_additive_vs_control.median(), linestyle=":", label=f"median={result.r2_mean_additive_vs_control.median():.3f}")
        ax.set_xlabel(r"additive $R^2_{mean}$ vs control")
        ax.set_ylabel("double perturbations")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "figures/double_prediction_histogram.png", dpi=180)
        plt.close(fig)
    return result


def lowrank_analysis(X, names: np.ndarray, kinds: np.ndarray, outdir: Path, n_pcs: int, min_single: int, seed: int) -> pd.DataFrame:
    use = (kinds == "control") | (kinds == "single")
    Xuse = X[use]
    if sparse.issparse(Xuse):
        Xuse = Xuse.toarray()
    n_pcs = min(n_pcs, Xuse.shape[1], Xuse.shape[0] - 1)
    reducer = PCA(n_components=n_pcs, svd_solver="randomized", random_state=seed)
    Y = reducer.fit_transform(Xuse)
    names_use, kinds_use = names[use], kinds[use]
    control = Y[kinds_use == "control"]
    theta0 = LedoitWolf().fit(control).precision_
    order = np.random.default_rng(seed).permutation(len(control))
    first, second = np.array_split(order, 2)
    null_difference = (LedoitWolf().fit(control[first]).precision_
                       - LedoitWolf().fit(control[second]).precision_)
    null_singular = np.linalg.svd(null_difference, compute_uv=False)
    null_energy = float(np.sum(null_singular**2))
    null = {
        "seed": seed, "n_pcs": n_pcs,
        "n_control_first": len(first), "n_control_second": len(second),
        "rho2": float(np.sum(null_singular[:2]**2) / null_energy) if null_energy > 1e-12 else None,
        "frobenius_norm": float(np.sqrt(null_energy)),
    }
    (outdir / "results/control_split_baseline.json").write_text(json.dumps(null, indent=2) + "\n")
    rows = []
    for name in sorted(set(names_use[kinds_use == "single"])):
        mask = (kinds_use == "single") & (names_use == name)
        if mask.sum() < max(min_single, n_pcs + 5):
            continue
        difference = LedoitWolf().fit(Y[mask]).precision_ - theta0
        singular = np.linalg.svd(difference, compute_uv=False)
        energy = float(np.sum(singular**2))
        value = float(np.sum(singular[:2] ** 2) / energy) if energy > 1e-12 else np.nan
        rows.append({"single_perturbation": name, "n_cells": int(mask.sum()), "rho2_top2_precision_difference": value, "rank_leakage": 1 - value})
    result = pd.DataFrame(rows).sort_values("rho2_top2_precision_difference", ascending=False) if rows else pd.DataFrame()
    result.to_csv(outdir / "results/lowrank_audit_metrics.csv", index=False)
    if len(result):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(result.rho2_top2_precision_difference, bins=max(5, min(20, len(result))))
        ax.axvline(1, linestyle="--", label="population rank <=2 reference")
        ax.axvline(result.rho2_top2_precision_difference.median(), linestyle=":", label=f"median={result.rho2_top2_precision_difference.median():.3f}")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(r"$\rho_2(\widehat\Theta_i-\widehat\Theta_0)$")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "figures/lowrank_audit_histogram.png", dpi=180)
        plt.close(fig)
    return result


def self_test_data(seed: int) -> Data:
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:03d}" for i in range(100)])
    targets = ["A", "B", "C", "D"]
    base = rng.normal(1.5, 0.2, len(genes))
    effects = {g: rng.normal(0, 0.25, len(genes)) for g in targets}
    rows, labels = [], []
    def add(label: str, mean: np.ndarray, n: int) -> None:
        rows.append(rng.normal(mean, 0.3, size=(n, len(genes))))
        labels.extend([label] * n)
    add("ctrl", base, 160)
    for g in targets:
        add(g, base + effects[g], 70)
    for i, first in enumerate(targets):
        for second in targets[i + 1:]:
            add(f"{first}+{second}", base + effects[first] + effects[second] + rng.normal(0, 0.03, len(genes)), 45)
    return Data(np.vstack(rows).astype(np.float32), np.array(labels), genes, "synthetic self-test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5ad", type=Path, help="processed AnnData file")
    source.add_argument("--geo-dir", type=Path, help="directory holding the four GSE133344 filtered files")
    source.add_argument("--self-test", action="store_true", help="run on generated synthetic data")
    parser.add_argument("--outdir", type=Path, default=Path("norman_results"), help="output directory")
    parser.add_argument("--label-column", help="adata.obs column with perturbation labels; inferred if omitted")
    parser.add_argument("--target-separator", help="separator between targets in labels, e.g. '_' for scPerturb")
    parser.add_argument("--layer", help="AnnData layer to use instead of adata.X")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip library-size normalization and log1p; use for already-processed data")
    parser.add_argument("--n-hvgs", type=int, default=2000,
                        help="number of most variable genes, chosen from control and single cells")
    parser.add_argument("--n-pcs", type=int, default=20, help="principal components for the low-rank check")
    parser.add_argument("--min-cells-control", type=int, default=200, help="minimum control cells")
    parser.add_argument("--min-cells-single", type=int, default=50, help="minimum cells per single perturbation")
    parser.add_argument("--min-cells-double", type=int, default=30, help="minimum cells per double perturbation")
    parser.add_argument("--seed", type=int, default=0, help="seed for gene subsampling, PCA, and self-test data")
    parser.add_argument("--skip-lowrank", action="store_true", help="skip the PCA and precision-matrix check")
    args = parser.parse_args()
    if args.h5ad:
        data = load_h5ad(args.h5ad, args.label_column, args.layer,
                         n_genes=args.n_hvgs, normalize=not args.no_normalize,
                         seed=args.seed, target_separator=args.target_separator)
        X = data.X
        selected = np.arange(X.shape[1])
    else:
        data = self_test_data(args.seed) if args.self_test else load_geo(args.geo_dir)
        X = data.X if (args.no_normalize or args.self_test) else normalize_log1p(data.X)
        _, kinds = context_arrays(data.labels, args.target_separator)
        training = (kinds == 'control') | (kinds == 'single')
        selected = select_variable_genes(X[training], min(args.n_hvgs, X.shape[1]), 50000, args.seed)
        X = X[:, selected]
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results").mkdir(exist_ok=True)
    (args.outdir / "figures").mkdir(exist_ok=True)
    names, kinds = context_arrays(data.labels, args.target_separator)
    pd.DataFrame({"gene": data.genes[selected]}).to_csv(args.outdir / "results/selected_genes.csv", index=False)
    pd.DataFrame({"context": names, "kind": kinds}).value_counts().rename("n_cells").reset_index().to_csv(args.outdir / "results/context_counts.csv", index=False)
    additive = additive_analysis(X, names, kinds, args.outdir, 50 if args.self_test else args.min_cells_control, 30 if args.self_test else args.min_cells_single, 20 if args.self_test else args.min_cells_double)
    # The direction/magnitude metrics use the same selected genes and context means as the additive analysis.
    contexts = sorted(set(names))
    means = pd.DataFrame([group_mean(X, names == c) for c in contexts],
                         index=contexts, columns=data.genes[selected], dtype=np.float64)
    means.index.name = "context"
    means.to_csv(args.outdir / "results/context_means.csv")
    if len(additive):
        pairs = additive[["pair", "gene_1", "gene_2"]].rename(
            columns={"pair": "double", "gene_1": "single_1", "gene_2": "single_2"})
        pairs.to_csv(args.outdir / "results/pairs.csv", index=False)
        direction = evaluate_pairs(means, "control", pairs)
        direction.to_csv(args.outdir / "results/direction_magnitude_metrics.csv", index=False)
        (args.outdir / "results/direction_magnitude_summary.json").write_text(
            json.dumps(summarize_direction(direction), indent=2) + "\n")
        make_plot(direction, args.outdir / "figures/direction_magnitude.pdf",
                  args.outdir / "figures/direction_magnitude.png", False)
    audit = pd.DataFrame()
    if not args.skip_lowrank:
        audit = lowrank_analysis(X, names, kinds, args.outdir, min(args.n_pcs, 5) if args.self_test else args.n_pcs, 30 if args.self_test else args.min_cells_single, args.seed)
    metadata = {
        "source": data.source, "n_cells": int(X.shape[0]), "n_genes_used": int(X.shape[1]), "seed": args.seed,
        "n_controls": int(np.sum(kinds == "control")),
        "n_eligible_singles": sum(int(np.sum(names == c)) >= (30 if args.self_test else args.min_cells_single)
                                  for c in set(names[kinds == "single"])),
        "n_evaluated_doubles": int(len(additive)), "n_audited_singles": int(len(audit)),
        "gene_selection_contexts": "control and singles only",
        "projection": "centered PCA, randomized solver; fit to control and singles",
        "precision_estimator": "LedoitWolf per context",
    }
    if args.h5ad:
        metadata["input_sha256"] = file_sha256(args.h5ad)
    (args.outdir / "results/metadata.json").write_text(json.dumps(metadata, indent=2))
    write_config(args.outdir / "results/config.json", vars(args),
                 "Genes and PCA are fitted to controls and singles. "
                 "Top-100 metrics select genes by observed double effects.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
