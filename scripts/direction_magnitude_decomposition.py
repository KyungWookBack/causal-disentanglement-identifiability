#!/usr/bin/env python3
"""Decompose additive perturbation prediction into direction and magnitude.

This script produces the effect-space diagnostic in the report. It reads
context means from a CSV or aggregates them from a processed .h5ad file; the
AnnData file itself is not bundled with the repository.

For each held-out double perturbation, define

    delta     = mu_double - mu_control
    delta_hat = (mu_single_1 - mu_control) + (mu_single_2 - mu_control).

The decomposition reports

    a = ||delta_hat|| / ||delta||                       (magnitude ratio)
    c = <delta, delta_hat> / (||delta|| ||delta_hat||) (direction cosine)

and verifies

    R2_mean = 1 - ||delta - delta_hat||^2 / ||delta||^2 = 2*a*c - a^2.

It also computes the optimally rescaled prediction alpha* delta_hat and the
corresponding R2_scaled = c^2 (up to numerical tolerance).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class PairMetrics:
    double: str
    single_1: str
    single_2: str
    n_genes: int
    true_effect_norm: float
    predicted_effect_norm: float
    magnitude_ratio: float
    direction_cosine: float
    r2_mean: float
    r2_from_decomposition: float
    optimal_scale: float
    r2_scaled: float
    pearson_mean_response: float
    pearson_effect: float


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Pearson inputs have different shapes: {x.shape} vs {y.shape}")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= EPS:
        return math.nan
    return float(np.dot(x_centered, y_centered) / denom)


def compute_pair_metrics(
    mu_control: np.ndarray,
    mu_single_1: np.ndarray,
    mu_single_2: np.ndarray,
    mu_double: np.ndarray,
    *,
    double_label: str,
    single_1_label: str,
    single_2_label: str,
) -> PairMetrics:
    """Compute direction-magnitude diagnostics for one held-out pair."""
    arrays = [
        np.asarray(mu_control, dtype=float).ravel(),
        np.asarray(mu_single_1, dtype=float).ravel(),
        np.asarray(mu_single_2, dtype=float).ravel(),
        np.asarray(mu_double, dtype=float).ravel(),
    ]
    if len({arr.shape for arr in arrays}) != 1:
        raise ValueError(f"All mean vectors must have the same shape; got {[a.shape for a in arrays]}")
    if not all(np.isfinite(arr).all() for arr in arrays):
        raise ValueError(f"Non-finite values found for pair {double_label!r}")

    mu0, mu1, mu2, mu12 = arrays
    true_effect = mu12 - mu0
    predicted_mean = mu1 + mu2 - mu0
    predicted_effect = predicted_mean - mu0

    true_norm = float(np.linalg.norm(true_effect))
    pred_norm = float(np.linalg.norm(predicted_effect))
    if true_norm <= EPS:
        raise ValueError(
            f"True double-perturbation effect is numerically zero for {double_label!r}; "
            "R2_mean and the decomposition are undefined."
        )

    if pred_norm <= EPS:
        cosine = math.nan
        magnitude_ratio = 0.0
        optimal_scale = math.nan
        r2_scaled = 0.0
    else:
        cosine = float(np.dot(true_effect, predicted_effect) / (true_norm * pred_norm))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        magnitude_ratio = pred_norm / true_norm
        optimal_scale = float(np.dot(true_effect, predicted_effect) / (pred_norm**2))
        scaled_residual = true_effect - optimal_scale * predicted_effect
        r2_scaled = 1.0 - float(np.dot(scaled_residual, scaled_residual)) / (true_norm**2)

    residual = true_effect - predicted_effect
    r2_mean = 1.0 - float(np.dot(residual, residual)) / (true_norm**2)
    r2_identity = (
        2.0 * magnitude_ratio * cosine - magnitude_ratio**2
        if math.isfinite(cosine)
        else 0.0
    )

    if not np.isclose(r2_mean, r2_identity, atol=1e-9, rtol=1e-8):
        raise AssertionError(
            f"R2 decomposition failed for {double_label}: direct={r2_mean}, identity={r2_identity}"
        )
    if math.isfinite(cosine) and not np.isclose(r2_scaled, cosine**2, atol=1e-9, rtol=1e-8):
        raise AssertionError(
            f"Scaled R2 identity failed for {double_label}: scaled={r2_scaled}, cosine^2={cosine**2}"
        )

    return PairMetrics(
        double=double_label,
        single_1=single_1_label,
        single_2=single_2_label,
        n_genes=int(mu0.size),
        true_effect_norm=true_norm,
        predicted_effect_norm=pred_norm,
        magnitude_ratio=float(magnitude_ratio),
        direction_cosine=float(cosine),
        r2_mean=float(r2_mean),
        r2_from_decomposition=float(r2_identity),
        optimal_scale=float(optimal_scale),
        r2_scaled=float(r2_scaled),
        pearson_mean_response=_safe_pearson(mu12, predicted_mean),
        pearson_effect=_safe_pearson(true_effect, predicted_effect),
    )


def load_means_csv(path: Path, context_column: str) -> pd.DataFrame:
    """Load context-by-gene means from CSV."""
    frame = pd.read_csv(path)
    if context_column not in frame.columns:
        raise ValueError(
            f"Context column {context_column!r} not found in {path}. "
            f"Available columns: {list(frame.columns[:10])}"
        )
    if frame[context_column].duplicated().any():
        dup = frame.loc[frame[context_column].duplicated(), context_column].tolist()[:5]
        raise ValueError(f"Duplicate context labels in means CSV: {dup}")
    frame = frame.set_index(context_column)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if numeric.shape[1] < 2:
        raise ValueError("Means CSV must contain at least two gene columns")
    numeric.index = numeric.index.astype(str)
    return numeric


def _collapse_context_label(label: str, control_token: str | None) -> str:
    """Remove a control token from ``+``-separated perturbation labels.

    For example, both ``ctrl+KLF1`` and ``KLF1+ctrl`` become ``KLF1`` while
    ``CEBPE+KLF1`` is unchanged. A pure control label remains ``ctrl``.
    """
    text = str(label)
    if control_token is None:
        return text
    parts = [part.strip() for part in text.split("+")]
    kept = [part for part in parts if part and part != control_token]
    return control_token if not kept else "+".join(kept)


def load_means_h5ad(
    path: Path,
    context_key: str,
    *,
    layer: str | None,
    use_raw: bool,
    collapse_control_token: str | None,
) -> pd.DataFrame:
    """Aggregate an AnnData object into context-by-gene means.

    The file is opened in backed mode so that large unused layers do not need to
    be loaded into memory.
    """
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError(
            "Reading .h5ad requires optional dependencies 'anndata' and 'scipy'. "
            "Install them or supply --means-csv instead."
        ) from exc

    adata = ad.read_h5ad(path, backed="r")
    if context_key not in adata.obs.columns:
        raise ValueError(
            f"Context key {context_key!r} not found in adata.obs. "
            f"Available keys include: {list(adata.obs.columns[:20])}"
        )

    if use_raw:
        if adata.raw is None:
            raise ValueError("--use-raw was requested but adata.raw is None")
        matrix = adata.raw.X
        gene_names = np.asarray(adata.raw.var_names, dtype=str)
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer {layer!r} not found; available layers: {list(adata.layers.keys())}")
        matrix = adata.layers[layer]
        gene_names = np.asarray(adata.var_names, dtype=str)
    else:
        matrix = adata.X
        gene_names = np.asarray(adata.var_names, dtype=str)

    raw_labels = adata.obs[context_key].astype(str).to_numpy()
    labels = np.asarray(
        [_collapse_context_label(label, collapse_control_token) for label in raw_labels],
        dtype=str,
    )
    rows: list[np.ndarray] = []
    contexts: list[str] = []
    for context in pd.unique(labels):
        indices = np.flatnonzero(labels == context)
        block = matrix[indices, :]
        if sparse.issparse(block):
            mean = np.asarray(block.mean(axis=0)).ravel()
        else:
            mean = np.asarray(block, dtype=float).mean(axis=0)
        contexts.append(str(context))
        rows.append(mean)

    result = pd.DataFrame(np.vstack(rows), index=contexts, columns=gene_names)
    result.index.name = "context"
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return result


def load_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["double", "single_1", "single_2"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Pairs CSV is missing columns {missing}; required columns are {required}")
    frame = frame[required].dropna().astype(str)
    if frame.empty:
        raise ValueError("Pairs CSV contains no evaluable rows")
    return frame


def validate_contexts(means: pd.DataFrame, control: str, pairs: pd.DataFrame) -> None:
    required: set[str] = {control}
    for col in ("double", "single_1", "single_2"):
        required.update(pairs[col].tolist())
    missing = sorted(required.difference(set(means.index.astype(str))))
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise ValueError(f"The following required contexts are absent from the means table: {preview}{suffix}")


def evaluate_pairs(means: pd.DataFrame, control: str, pairs: pd.DataFrame) -> pd.DataFrame:
    validate_contexts(means, control, pairs)
    mu0 = means.loc[control].to_numpy(dtype=float)
    metrics: list[PairMetrics] = []
    for row in pairs.itertuples(index=False):
        metrics.append(
            compute_pair_metrics(
                mu0,
                means.loc[row.single_1].to_numpy(dtype=float),
                means.loc[row.single_2].to_numpy(dtype=float),
                means.loc[row.double].to_numpy(dtype=float),
                double_label=row.double,
                single_1_label=row.single_1,
                single_2_label=row.single_2,
            )
        )
    return pd.DataFrame([asdict(item) for item in metrics])


def make_plot(results: pd.DataFrame, output_pdf: Path, output_png: Path, label_all: bool) -> None:
    import matplotlib.pyplot as plt

    x = results["direction_cosine"].to_numpy(dtype=float)
    y = results["magnitude_ratio"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("No finite direction-magnitude pairs are available for plotting")

    x_min = min(0.8, float(np.nanmin(x[finite])) - 0.08)
    x_max = 1.04
    y_min = max(0.0, min(1.0, float(np.nanmin(y[finite]))) - 0.10)
    y_max = max(1.30, float(np.nanmax(y[finite])) * 1.10)

    grid_x = np.linspace(max(-1.0, x_min), x_max, 300)
    grid_y = np.linspace(y_min, y_max, 300)
    C, A = np.meshgrid(grid_x, grid_y)
    R2 = 2.0 * A * C - A**2

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    levels = [-1.0, 0.0, 0.5, 0.75]
    contours = ax.contour(C, A, R2, levels=levels, linewidths=0.9)
    label_positions = [(1.01, 2.43), (1.01, 2.02), (1.01, 1.73), (0.97, 0.53)]
    ax.clabel(contours, inline=True, fontsize=8, manual=label_positions,
              fmt=lambda value: rf"$R^2={value:g}$")
    ax.scatter(x[finite], y[finite], s=24, alpha=0.85, zorder=3)

    finite_indices = np.flatnonzero(finite)
    if label_all:
        to_label: Iterable[int] = finite_indices
    else:
        finite_r2 = results.iloc[finite_indices]["r2_mean"].to_numpy(dtype=float)
        worst = int(finite_indices[int(np.argmin(finite_r2))])
        best = int(finite_indices[int(np.argmax(finite_r2))])
        to_label = sorted({worst, best})

    for idx in to_label:
        row = results.iloc[idx]
        right = float(row["direction_cosine"]) > (x_min + x_max) / 2
        offset = (-20, 12) if right else (10, 12)
        if not label_all and idx == best and len(results) > 30:
            offset = (-45, -42)
        ax.annotate(
            str(row["double"]),
            (float(row["direction_cosine"]), float(row["magnitude_ratio"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="right" if right else "left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            arrowprops={"arrowstyle": "-", "linewidth": 0.6},
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0, label=r"$a=1$ (matched norm)")
    ax.axvline(1.0, linestyle=":", linewidth=1.0, label=r"$c=1$ (aligned effects)")
    ax.set_xlim(max(-1.0, x_min), x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Direction cosine $c$")
    ax.set_ylabel(r"Magnitude ratio $a=\|\widehat\delta\|_2/\|\delta\|_2$")
    ax.set_title(r"Additive prediction: $R^2_{\mathrm{mean}}=2ac-a^2$")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(results: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n_pairs": int(len(results)),
        "median_r2_mean": float(results["r2_mean"].median()),
        "mean_r2_mean": float(results["r2_mean"].mean()),
        "median_direction_cosine": float(results["direction_cosine"].median()),
        "median_magnitude_ratio": float(results["magnitude_ratio"].median()),
        "median_r2_scaled": float(results["r2_scaled"].median()),
        "n_negative_r2": int((results["r2_mean"] < 0).sum()),
    }


def run_self_test() -> None:
    mu0 = np.zeros(4)
    mu1 = np.array([1.0, 0.0, 0.0, 0.0])
    mu2 = np.array([0.0, 1.0, 0.0, 0.0])
    mu12 = np.array([0.5, 0.5, 0.0, 0.0])
    result = compute_pair_metrics(
        mu0,
        mu1,
        mu2,
        mu12,
        double_label="A+B",
        single_1_label="A",
        single_2_label="B",
    )
    assert np.isclose(result.direction_cosine, 1.0)
    assert np.isclose(result.magnitude_ratio, 2.0)
    assert np.isclose(result.r2_mean, 0.0)
    assert np.isclose(result.r2_scaled, 1.0)
    print("Self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--means-csv", type=Path, help="CSV with one context per row and one gene per column")
    source.add_argument("--h5ad", type=Path, help="Processed AnnData file")
    parser.add_argument("--context-column", default="context", help="Context-label column in --means-csv")
    parser.add_argument("--context-key", help="adata.obs column containing context labels (required with --h5ad)")
    parser.add_argument(
        "--collapse-control-token",
        help=(
            "Remove this token from '+'-separated labels before aggregation; "
            "for Norman guide_merged use 'ctrl'"
        ),
    )
    parser.add_argument("--layer", help="AnnData layer to aggregate; default uses adata.X")
    parser.add_argument("--use-raw", action="store_true", help="Aggregate adata.raw.X instead of adata.X")
    parser.add_argument("--pairs-csv", type=Path, help="CSV columns: double,single_1,single_2")
    parser.add_argument("--control-label", help="Context label for control cells")
    parser.add_argument("--out-dir", type=Path, default=Path("direction_magnitude_results"))
    parser.add_argument("--label-all", action="store_true", help="Annotate every pair in the plot")
    parser.add_argument("--self-test", action="store_true", help="check the R2 identities on a small example")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    if args.means_csv is None and args.h5ad is None:
        raise SystemExit("Supply either --means-csv or --h5ad (or use --self-test).")
    if args.pairs_csv is None or args.control_label is None:
        raise SystemExit("--pairs-csv and --control-label are required for analysis.")
    if args.h5ad is not None and not args.context_key:
        raise SystemExit("--context-key is required with --h5ad.")

    if args.means_csv is not None:
        means = load_means_csv(args.means_csv, args.context_column)
    else:
        means = load_means_h5ad(
            args.h5ad,
            args.context_key,
            layer=args.layer,
            use_raw=args.use_raw,
            collapse_control_token=args.collapse_control_token,
        )

    pairs = load_pairs(args.pairs_csv)
    results = evaluate_pairs(means, args.control_label, pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "direction_magnitude_metrics.csv"
    summary_path = args.out_dir / "direction_magnitude_summary.json"
    pdf_path = args.out_dir / "direction_magnitude_decomposition.pdf"
    png_path = args.out_dir / "direction_magnitude_decomposition.png"

    results.to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(summarize(results), indent=2) + "\n")
    make_plot(results, pdf_path, png_path, args.label_all)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
