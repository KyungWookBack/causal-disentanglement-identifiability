#!/usr/bin/env python3
"""Oracle soft-intervention distribution-shift probe.

Latent coordinates of a five-node nonlinear SCM are observed directly. For
each intervention target i and each other coordinate j, the score is the
squared 1-D RBF MMD between observational and interventional samples of z_j,
and the label is whether i is an ancestor of j. The MMD is the biased
V-statistic with a median-heuristic bandwidth. By default every sample enters
the kernel sums; --mmd-max-n caps each side only when set to a positive value.

The F1 cutoff is a quantile (0.95 by default) of the five control-control MMD
scores computed at the same sample size. This cutoff is exploratory and has
no calibrated false-positive rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from experiment_utils import write_config

D = 5
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
ADJ = np.zeros((D, D), dtype=int)
for i, j in EDGES:
    ADJ[i, j] = 1
TC = ADJ.copy()
for k in range(D):
    TC = np.logical_or(TC, np.outer(TC[:, k], TC[k, :])).astype(int)


def sample_scm(n: int, delta: float, target: int | None, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(size=(n, D))
    if target is not None:
        e[:, target] += delta
    z = np.zeros_like(e)
    z[:, 0] = e[:, 0]
    z[:, 1] = 0.9 * np.tanh(z[:, 0]) + 0.25 * (z[:, 0] ** 2 - 1) + e[:, 1]
    z[:, 2] = -0.8 * np.sin(z[:, 0]) + 0.20 * (z[:, 0] ** 2 - 1) + e[:, 2]
    z[:, 3] = 0.7 * np.tanh(z[:, 1]) - 0.6 * np.tanh(z[:, 2]) + 0.15 * (z[:, 1] ** 2 - z[:, 2] ** 2) + e[:, 3]
    z[:, 4] = 0.9 * np.tanh(z[:, 3]) + 0.20 * (z[:, 3] ** 2 - 1) + e[:, 4]
    return z


def subsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x).ravel()
    return x if max_n == 0 or len(x) <= max_n else x[rng.choice(len(x), max_n, replace=False)]


def rbf_mean(x: np.ndarray, y: np.ndarray, gamma: float, block_size: int = 256) -> float:
    """Exact kernel mean, summing row blocks to bound working memory."""
    total = 0.0
    for start in range(0, len(x), block_size):
        difference = x[start:start + block_size, None] - y[None, :]
        np.square(difference, out=difference)
        difference *= -gamma
        np.exp(difference, out=difference)
        total += float(difference.sum())
    return total / (len(x) * len(y))


def mmd2_rbf(x: np.ndarray, y: np.ndarray, max_n: int, rng: np.random.Generator) -> float:
    """Biased (V-statistic) RBF MMD squared with a median-heuristic bandwidth.

    The bandwidth is set from at most 400 pooled points to bound its cost; the
    kernel sums themselves use every point that `subsample` returns.
    """
    x, y = subsample(x, max_n, rng), subsample(y, max_n, rng)
    z = np.concatenate([x, y])
    z_band = subsample(z, min(400, len(z)), rng)
    sq = (z_band[:, None] - z_band[None, :]) ** 2
    upper = sq[np.triu_indices_from(sq, k=1)]
    gamma = 1.0 / (2 * max(float(np.median(upper)), 1e-8))
    return max(0.0, rbf_mean(x, x, gamma) + rbf_mean(y, y, gamma)
               - 2 * rbf_mean(x, y, gamma))


def run_grid(ns: tuple[int, ...], deltas: tuple[float, ...], repeats: int, max_n: int, null_q: float, seed: int) -> pd.DataFrame:
    rows = []
    for delta in deltas:
        for n in ns:
            for rep in range(repeats):
                rng = np.random.default_rng(seed + n * 37 + rep * 100003 + int(delta * 1000))
                z0a, z0b = sample_scm(n, 0, None, rng), sample_scm(n, 0, None, rng)
                null = [mmd2_rbf(z0a[:, j], z0b[:, j], max_n, rng) for j in range(D)]
                threshold = float(np.quantile(null, null_q))
                scores, labels = [], []
                for target in range(D):
                    zi = sample_scm(n, delta, target, rng)
                    for j in range(D):
                        if j == target:
                            continue
                        scores.append(mmd2_rbf(z0a[:, j], zi[:, j], max_n, rng))
                        labels.append(TC[target, j])
                scores, labels = np.asarray(scores), np.asarray(labels)
                rows.append({
                    "delta": delta, "n": n, "rep": rep,
                    "ancestor_auc": roc_auc_score(labels, scores),
                    "tc_f1": f1_score(labels, scores > threshold, zero_division=0),
                    "null_threshold": threshold,
                    "effective_n": n if max_n == 0 else min(n, max_n),
                })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["delta", "n"], as_index=False).agg(
        ancestor_auc_mean=("ancestor_auc", "mean"),
        ancestor_auc_sd=("ancestor_auc", "std"),
        tc_f1_mean=("tc_f1", "mean"),
        tc_f1_sd=("tc_f1", "std"),
    )


def heatmaps(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    deltas, ns = sorted(summary.delta.unique()), sorted(summary.n.unique())
    auc = summary.pivot(index="delta", columns="n", values="ancestor_auc_mean").loc[deltas, ns].to_numpy()
    f1 = summary.pivot(index="delta", columns="n", values="tc_f1_mean").loc[deltas, ns].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    image = None
    for ax, matrix, title in zip(axes, [auc, f1], ["Ancestor AUC", r"TC $F_1$"]):
        image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ns)), ns)
        ax.set_yticks(range(len(deltas)), deltas)
        ax.set_xlabel("samples per context")
        ax.set_ylabel(r"shift $\delta$")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes, fraction=0.04, pad=0.04)
    fig.suptitle("Soft-intervention distribution-shift probe")
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.18, top=0.80, wspace=0.28)
    fig.savefig(outdir / "soft_mmd_probe.pdf")
    fig.savefig(outdir / "soft_mmd_probe.png", dpi=220)
    plt.close(fig)


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("soft_mmd_results"), help="output directory")
    parser.add_argument("--sample-sizes", type=ints, default=ints("250,500,1000"),
                        help="comma-separated samples per context")
    parser.add_argument("--deltas", type=floats, default=floats("0.25,0.5,1,2"),
                        help="comma-separated mean shifts added to the target's noise")
    parser.add_argument("--repeats", type=int, default=30, help="repetitions per (delta, n) cell")
    parser.add_argument("--mmd-max-n", type=int, default=0,
                        help="0 uses every sample in the kernel sums (default); "
                             "a positive value subsamples each side to at most this many points")
    parser.add_argument("--null-quantile", type=float, default=0.95,
                        help="quantile of the five control-control MMD scores used as the F1 cutoff")
    parser.add_argument("--seed", type=int, default=20260607, help="base random seed")
    parser.add_argument("--smoke-test", action="store_true", help="run a tiny grid to check the pipeline")
    args = parser.parse_args()
    if args.smoke_test:
        args.sample_sizes, args.deltas, args.repeats, args.mmd_max_n = (60, 100), (0.5, 1.0), 2, 0
    if args.mmd_max_n < 0 or args.repeats < 1 or any(n < 2 for n in args.sample_sizes):
        parser.error("Use positive repeats, sample sizes >=2, and a nonnegative MMD cap")
    if not 0 < args.null_quantile < 1:
        parser.error("--null-quantile must lie strictly between 0 and 1")
    if args.mmd_max_n and any(n > args.mmd_max_n for n in args.sample_sizes):
        print("MMD inputs are capped; use effective_n in raw.csv when interpreting sample size.", flush=True)
    raw = run_grid(args.sample_sizes, args.deltas, args.repeats, args.mmd_max_n, args.null_quantile, args.seed)
    summary = summarize(raw)
    (args.outdir / "results").mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.outdir / "results/raw.csv", index=False)
    summary.to_csv(args.outdir / "results/summary.csv", index=False)
    write_config(args.outdir / "results/config.json", vars(args),
                 "Oracle latent coordinates; biased RBF MMD squared (V-statistic). "
                 "All samples enter kernel sums by default; blocks only limit memory. "
                 "The median bandwidth uses at most 400 pooled observations per comparison. "
                 "The F1 cutoff is the 0.95 quantile of five control-control scores by default. "
                 "Defaults use n=250,500,1000 and 30 repetitions.")
    heatmaps(summary, args.outdir / "figures")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
