#!/usr/bin/env python3
"""Perfect-intervention low-rank precision-difference probe.

A linear Gaussian SCM on d=5 latent nodes is mixed into p=10 observed
coordinates. Each perfect intervention removes the target's incoming edges
and multiplies its noise variance. Precision matrices are pseudoinverses of
the sample covariances (rcond=1e-10) with no ridge, because X=GZ has rank
d<p. For each intervention the script reports rho_2, the share of the
squared singular values of Theta_i-Theta_0 carried by its top two, and the
rank leakage 1-rho_2; the population difference has rank at most two. A
second control sample gives an independent control-control difference norm
as a null reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_utils import write_config

D, P = 5, 10
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]


def parameters(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.zeros((D, D))
    for parent, child in EDGES:
        a[child, parent] = rng.uniform(0.35, 0.9) * rng.choice([-1.0, 1.0])
    omega = rng.uniform(1.0, 2.0, size=D)
    q, _ = np.linalg.qr(rng.normal(size=(P, D)))
    g = q[:, :D] * rng.uniform(0.8, 1.5, size=D)
    return a, omega, g


def sample(n: int, a: np.ndarray, omega: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(size=(n, D)) * np.sqrt(omega)
    z = np.linalg.solve(np.eye(D) - a, eps.T).T
    return z @ g.T


def intervene(a0: np.ndarray, omega0: np.ndarray, target: int, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    a, omega = a0.copy(), omega0.copy()
    a[target, :] = 0.0
    omega[target] *= multiplier
    return a, omega


def precision(x: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Pseudoinverse of the sample covariance on its estimated support, with no ridge.

    X=GZ has rank d<p, so the covariance is singular. A ridge would invert to
    eigenvalues of order 1/ridge in the null space, and their imperfect
    cancellation would dominate the precision differences.
    """
    xc = x - x.mean(axis=0)
    cov = xc.T @ xc / (len(x) - 1)
    cov = (cov + cov.T) / 2
    return np.linalg.pinv(cov, rcond=rcond, hermitian=True)


def rho2(difference: np.ndarray) -> float:
    """Share of the squared singular values carried by the top two."""
    singular = np.linalg.svd(difference, compute_uv=False)
    energy = float(np.sum(singular**2))
    return float(np.sum(singular[:2] ** 2) / energy) if energy > 1e-14 else np.nan


def run(ns: tuple[int, ...], repeats: int, multiplier: float, seed: int) -> pd.DataFrame:
    rows = []
    for n in ns:
        for rep in range(repeats):
            rng = np.random.default_rng(seed + n * 37 + rep * 100003)
            a0, omega0, g = parameters(rng)
            x0a, x0b = sample(n, a0, omega0, g, rng), sample(n, a0, omega0, g, rng)
            theta0a, theta0b = precision(x0a), precision(x0b)
            null_norm = float(np.linalg.norm(theta0b - theta0a, ord="fro"))
            for target in range(D):
                a1, omega1 = intervene(a0, omega0, target, multiplier)
                dmat = precision(sample(n, a1, omega1, g, rng)) - theta0a
                value = rho2(dmat)
                rows.append({
                    "n": n, "rep": rep, "target": target,
                    "rho2": value, "rank_leakage": 1 - value,
                    "intervention_norm_fro": float(np.linalg.norm(dmat, ord="fro")),
                    "control_control_null_norm_fro": null_norm,
                })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    per_rep = raw.groupby(["n", "rep"], as_index=False).agg(
        rank_leakage=("rank_leakage", "mean"),
        signal_norm=("intervention_norm_fro", "mean"),
        null_norm=("control_control_null_norm_fro", "first"),
    )
    return per_rep.groupby("n", as_index=False).agg(
        rank_leakage_mean=("rank_leakage", "mean"),
        rank_leakage_sd=("rank_leakage", "std"),
        signal_norm_mean=("signal_norm", "mean"),
        signal_norm_sd=("signal_norm", "std"),
        null_norm_mean=("null_norm", "mean"),
        null_norm_sd=("null_norm", "std"),
    )


def plot(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    x = summary.n.to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    ax1.errorbar(x, summary.rank_leakage_mean, yerr=summary.rank_leakage_sd.fillna(0), marker="o", capsize=3)
    ax1.set_xscale("log")
    ax1.set_xlabel("samples per context")
    ax1.set_ylabel(r"rank leakage $1-\rho_2(D_k)$")
    ax1.set_title("Rank-two concentration")
    ax1.grid(alpha=0.25)
    scale = max(float(summary.signal_norm_mean.iloc[-1]), 1e-12)
    ax2.errorbar(x, summary.signal_norm_mean / scale, marker="o", label="intervention-control")
    ax2.errorbar(x, summary.null_norm_mean / scale, marker="s", linestyle="--", label="control-control null")
    ax2.set_xscale("log")
    ax2.set_xlabel("samples per context")
    ax2.set_ylabel("Frobenius norm / signal scale")
    ax2.set_title("Signal and null magnitude")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "perfect_precision_probe.pdf")
    fig.savefig(outdir / "perfect_precision_probe.png", dpi=220)
    plt.close(fig)


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("perfect_results"), help="output directory")
    parser.add_argument("--sample-sizes", type=ints, default=ints("250,500,1000,3000,5000,10000"),
                        help="comma-separated samples per context")
    parser.add_argument("--repeats", type=int, default=50,
                        help="repetitions per sample size; SCM parameters are redrawn each time")
    parser.add_argument("--variance-multiplier", type=float, default=4.0,
                        help="factor applied to the intervened node's noise variance")
    parser.add_argument("--seed", type=int, default=20260607, help="base random seed")
    parser.add_argument("--smoke-test", action="store_true", help="run a tiny grid to check the pipeline")
    args = parser.parse_args()
    if args.smoke_test:
        args.sample_sizes, args.repeats = (100, 250), 2
    raw = run(args.sample_sizes, args.repeats, args.variance_multiplier, args.seed)
    summary = summarize(raw)
    (args.outdir / "results").mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.outdir / "results/raw.csv", index=False)
    summary.to_csv(args.outdir / "results/summary.csv", index=False)
    write_config(args.outdir / "results/config.json", vars(args),
                 "Linear Gaussian SCM, d=5, p=10; population rank <=2. "
                 "Sample covariance pseudoinverse uses rcond=1e-10, no ridge. "
                 "Model parameters and samples are redrawn per repetition. "
                 "Uncertainty is summarized across repetitions after averaging targets.")
    plot(summary, args.outdir / "figures")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
