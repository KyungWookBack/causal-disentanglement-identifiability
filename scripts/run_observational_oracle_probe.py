#!/usr/bin/env python3
"""Oracle score-Hessian noise probe for the observational regime.

Latent coordinates of a five-node nonlinear SCM are observed directly, and
the true topological layers are given. At each peeling stage the active nodes
are those not yet removed, and the diagonal of the score Hessian is computed
analytically for them. Gaussian noise scaled by tau is added, and the nodes
whose Hessian entries have the smallest sample variance are selected as
leaves, with the true leaf count given. The probe reports leaf recall and
leaf AUC against the true leaves. It measures conditional leaf selection
only, not end-to-end layer recovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_utils import write_config

PARENTS = {0: [], 1: [0], 2: [0], 3: [1, 2], 4: [3]}
CHILDREN = {i: [] for i in range(5)}
for child, parents in PARENTS.items():
    for parent in parents:
        CHILDREN[parent].append(child)
TRUE_LAYERS = np.array([3, 2, 2, 1, 0])
SIGMA = np.array([0.70, 0.55, 0.60, 0.55, 0.50])


def sech2(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x) ** 2


def sample_latents(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 5)) * SIGMA
    z = np.zeros_like(e)
    z[:, 0] = e[:, 0]
    z[:, 1] = 0.90 * np.tanh(z[:, 0]) + 0.25 * z[:, 0] ** 2 + e[:, 1]
    z[:, 2] = -0.75 * np.sin(z[:, 0]) + 0.20 * z[:, 0] ** 2 + e[:, 2]
    z[:, 3] = 0.65 * np.tanh(z[:, 1]) + 0.55 * np.sin(z[:, 2]) + 0.18 * z[:, 1] ** 2 - 0.14 * z[:, 2] ** 2 + e[:, 3]
    z[:, 4] = 0.85 * np.tanh(z[:, 3]) + 0.25 * z[:, 3] ** 2 + e[:, 4]
    return z


def residuals(z: np.ndarray) -> np.ndarray:
    r = np.zeros_like(z)
    r[:, 0] = z[:, 0]
    r[:, 1] = z[:, 1] - (0.90 * np.tanh(z[:, 0]) + 0.25 * z[:, 0] ** 2)
    r[:, 2] = z[:, 2] - (-0.75 * np.sin(z[:, 0]) + 0.20 * z[:, 0] ** 2)
    r[:, 3] = z[:, 3] - (0.65 * np.tanh(z[:, 1]) + 0.55 * np.sin(z[:, 2]) + 0.18 * z[:, 1] ** 2 - 0.14 * z[:, 2] ** 2)
    r[:, 4] = z[:, 4] - (0.85 * np.tanh(z[:, 3]) + 0.25 * z[:, 3] ** 2)
    return r


def derivatives(child: int, parent: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = z[:, parent]
    if (child, parent) == (1, 0):
        return 0.90 * sech2(x) + 0.50 * x, -1.80 * sech2(x) * np.tanh(x) + 0.50
    if (child, parent) == (2, 0):
        return -0.75 * np.cos(x) + 0.40 * x, 0.75 * np.sin(x) + 0.40
    if (child, parent) == (3, 1):
        return 0.65 * sech2(x) + 0.36 * x, -1.30 * sech2(x) * np.tanh(x) + 0.36
    if (child, parent) == (3, 2):
        return 0.55 * np.cos(x) - 0.28 * x, -0.55 * np.sin(x) - 0.28
    if (child, parent) == (4, 3):
        return 0.85 * sech2(x) + 0.50 * x, -1.70 * sech2(x) * np.tanh(x) + 0.50
    raise ValueError((child, parent))


def oracle_diag_hessian(z: np.ndarray, active: list[int]) -> np.ndarray:
    r = residuals(z)
    active_set = set(active)
    out = np.zeros((len(z), len(active)))
    for col, node in enumerate(active):
        value = -np.ones(len(z)) / SIGMA[node] ** 2
        for child in CHILDREN[node]:
            if child in active_set:
                fp, fpp = derivatives(child, node, z)
                value += (r[:, child] * fpp - fp**2) / SIGMA[child] ** 2
        out[:, col] = value
    return out


def prepare_stages(z: np.ndarray) -> list[dict]:
    """Build each stage's active set, leaves, oracle Hessian, and noise scale.

    The noise scale is the median SD of the non-leaf Hessian columns, so tau
    is measured relative to the spread of the entries the leaves must beat.
    """
    stages = []
    for layer in range(TRUE_LAYERS.max() + 1):
        active = np.where(TRUE_LAYERS >= layer)[0].tolist()
        leaves = np.where(TRUE_LAYERS == layer)[0].tolist()
        hessian = oracle_diag_hessian(z, active)
        std = hessian.std(axis=0, ddof=1)
        nonleaf = [i for i, node in enumerate(active) if node not in leaves]
        scale = float(np.median(std[nonleaf])) if nonleaf else float(np.median(np.abs(hessian)))
        stages.append({"layer": layer, "active": active, "leaves": leaves, "hessian": hessian, "scale": max(scale, 1e-8)})
    return stages


def run_probe(n: int, repeats: int, taus: Iterable[float], seed: int) -> pd.DataFrame:
    rows = []
    taus = list(taus)
    for rep in range(repeats):
        z = sample_latents(n, seed + 1 + rep * 100003)
        stages = prepare_stages(z)
        for tau in taus:
            # Reseed per tau so every tau sees the same noise draws within a repetition.
            rng = np.random.default_rng(seed + rep * 100003)
            for stage in stages:
                if len(stage["leaves"]) == len(stage["active"]):
                    continue  # No non-leaf comparison in the final stage.
                noisy = stage["hessian"] + rng.normal(0, tau * stage["scale"], size=stage["hessian"].shape)
                variance = noisy.var(axis=0, ddof=1)
                selected = np.argsort(variance)[: len(stage["leaves"])]
                selected_nodes = [stage["active"][i] for i in selected]
                y = np.array([int(node in stage["leaves"]) for node in stage["active"]])
                rows.append({
                    "tau": float(tau), "rep": rep, "stage": stage["layer"],
                    "n_active": len(stage["active"]), "n_leaves": len(stage["leaves"]),
                    "leaf_recall": len(set(selected_nodes) & set(stage["leaves"])) / len(stage["leaves"]),
                    "leaf_auc": roc_auc_score(y, -variance),
                    "noise_scale": stage["scale"],
                })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    per_rep = raw.groupby(["tau", "rep"], as_index=False).agg(
        leaf_recall=("leaf_recall", "mean"), leaf_auc=("leaf_auc", "mean"))
    return per_rep.groupby("tau", as_index=False).agg(
        leaf_recall_mean=("leaf_recall", "mean"),
        leaf_recall_sd=("leaf_recall", "std"),
        leaf_auc_mean=("leaf_auc", "mean"),
        leaf_auc_sd=("leaf_auc", "std"),
    )


def plot(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(summary.tau, summary.leaf_recall_mean, marker="o", label="Conditional leaf recall")
    ax.plot(summary.tau, summary.leaf_auc_mean, marker="s", label="Conditional leaf AUC")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(0, max(summary.tau) * 1.2 if max(summary.tau) > 0 else 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"injected score-Hessian noise $\tau$")
    ax.set_ylabel("Mean across true stages and repetitions")
    ax.set_title("Oracle Hessian sensitivity (true stages given)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "observational_probe.pdf")
    fig.savefig(outdir / "observational_probe.png", dpi=220)
    plt.close(fig)


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("observational_results"), help="output directory")
    parser.add_argument("--n", type=int, default=300, help="samples per repetition")
    parser.add_argument("--repeats", type=int, default=200, help="repetitions; each draws new data from the fixed SCM")
    parser.add_argument("--taus", type=parse_floats, default=parse_floats("0,0.1,0.3,1,3,10,30"),
                        help="comma-separated noise strengths relative to the non-leaf Hessian SD")
    parser.add_argument("--seed", type=int, default=20260607, help="base random seed")
    parser.add_argument("--smoke-test", action="store_true", help="run a tiny grid to check the pipeline")
    args = parser.parse_args()
    if args.smoke_test:
        args.n, args.repeats, args.taus = 120, 3, [0.0, 1.0, 4.0]
    raw = run_probe(args.n, args.repeats, args.taus, args.seed)
    summary = summarize(raw)
    (args.outdir / "results").mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.outdir / "results/raw.csv", index=False)
    summary.to_csv(args.outdir / "results/summary.csv", index=False)
    write_config(args.outdir / "results/config.json", vars(args),
                 "Latent coordinates, true active sets and leaf counts are given. "
                 "Noise is scaled by the median true non-leaf Hessian SD at each stage. "
                 "Each repetition resamples the SCM; tau settings share data and noise draws. "
                 "Scores average the three nontrivial stages within each repetition. "
                 "This is conditional leaf selection, not end-to-end layer recovery.")
    plot(summary, args.outdir / "figures")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
