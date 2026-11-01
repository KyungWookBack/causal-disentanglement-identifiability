#!/usr/bin/env python3
"""Prepare collapsed Norman Perturb-seq context means and eligible double pairs.

The processed AnnData used in the report stores labels such as ``ctrl+KLF1`` and
``KLF1+ctrl`` for single perturbations.  This script removes the control token,
collapses equivalent labels, aggregates context means in row chunks so the full
matrix is never held in memory, and applies the cell-count thresholds used in
the report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    min_control: int = 200
    min_single: int = 50
    min_double: int = 30


def collapse_label(raw: str, control_token: str = "ctrl", separator: str = "+") -> str:
    """Remove control tokens while preserving the order of perturbation targets."""
    tokens = [token.strip() for token in str(raw).split(separator) if token.strip()]
    targets = [token for token in tokens if token != control_token]
    return control_token if not targets else separator.join(targets)


def target_tokens(label: str, control_token: str = "ctrl", separator: str = "+") -> list[str]:
    if label == control_token:
        return []
    return [token for token in str(label).split(separator) if token and token != control_token]


def discover_eligible_pairs(
    counts: pd.Series,
    thresholds: Thresholds,
    control_token: str = "ctrl",
    separator: str = "+",
) -> tuple[list[str], pd.DataFrame]:
    """Return singles with at least min_single cells and doubles with at least
    min_double cells whose two targets are both eligible singles."""
    if int(counts.get(control_token, 0)) < thresholds.min_control:
        raise ValueError(
            f"Control count {int(counts.get(control_token, 0))} is below "
            f"the required {thresholds.min_control}."
        )

    singles = sorted(
        label
        for label, count in counts.items()
        if len(target_tokens(label, control_token, separator)) == 1
        and int(count) >= thresholds.min_single
    )
    single_set = set(singles)

    pairs: list[dict[str, str]] = []
    for label, count in counts.items():
        tokens = target_tokens(label, control_token, separator)
        if len(tokens) != 2 or int(count) < thresholds.min_double:
            continue
        if tokens[0] in single_set and tokens[1] in single_set:
            pairs.append({"double": label, "single_1": tokens[0], "single_2": tokens[1]})

    pair_frame = pd.DataFrame(pairs, columns=["double", "single_1", "single_2"])
    if not pair_frame.empty:
        pair_frame = pair_frame.sort_values("double", kind="stable").reset_index(drop=True)
    return singles, pair_frame


def _dense_rows(matrix, start: int, stop: int) -> np.ndarray:
    block = matrix[start:stop]
    try:
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError("scipy is required to read sparse AnnData matrices") from exc
    if sparse.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=np.float64)


def aggregate_h5ad(
    h5ad: Path,
    context_key: str,
    control_token: str,
    separator: str,
    chunk_size: int,
    layer: str | None,
    use_raw: bool,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Install anndata to prepare inputs from .h5ad") from exc

    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        if context_key not in adata.obs.columns:
            raise KeyError(
                f"Context key {context_key!r} is absent. Available columns include: "
                f"{list(adata.obs.columns[:20])}"
            )
        if use_raw and layer is not None:
            raise ValueError("Use only one of --use-raw and --layer")

        if use_raw:
            if adata.raw is None:
                raise ValueError("--use-raw was requested but adata.raw is absent")
            matrix = adata.raw.X
            gene_names = np.asarray(adata.raw.var_names.astype(str))
            matrix_source = "raw.X"
        elif layer is not None:
            if layer not in adata.layers:
                raise KeyError(f"Layer {layer!r} is absent; available layers: {list(adata.layers.keys())}")
            matrix = adata.layers[layer]
            gene_names = np.asarray(adata.var_names.astype(str))
            matrix_source = f"layers[{layer!r}]"
        else:
            matrix = adata.X
            gene_names = np.asarray(adata.var_names.astype(str))
            matrix_source = "X"

        raw_labels = adata.obs[context_key].astype(str).to_numpy()
        collapsed = np.asarray(
            [collapse_label(label, control_token, separator) for label in raw_labels],
            dtype=object,
        )

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        n_obs = int(adata.n_obs)
        for start in range(0, n_obs, chunk_size):
            stop = min(start + chunk_size, n_obs)
            block = _dense_rows(matrix, start, stop)
            block_labels = collapsed[start:stop]
            for label in pd.unique(block_labels):
                mask = block_labels == label
                subtotal = np.asarray(block[mask].sum(axis=0), dtype=np.float64).ravel()
                if label in sums:
                    sums[label] += subtotal
                else:
                    sums[label] = subtotal
                counts[label] = counts.get(label, 0) + int(mask.sum())

        ordered = sorted(sums)
        mean_array = np.vstack([sums[label] / counts[label] for label in ordered])
        means = pd.DataFrame(mean_array, index=ordered, columns=gene_names)
        means.index.name = "context"
        count_series = pd.Series({label: counts[label] for label in ordered}, name="count")
        count_series.index.name = "context"
        metadata = {
            "h5ad": str(h5ad),
            "n_cells": n_obs,
            "n_genes": int(len(gene_names)),
            "context_key": context_key,
            "control_token": control_token,
            "separator": separator,
            "matrix_source": matrix_source,
            "chunk_size": int(chunk_size),
        }
        return means, count_series, metadata
    finally:
        file_obj = getattr(adata, "file", None)
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass


def run_self_test() -> None:
    labels = ["ctrl", "ctrl+A", "A+ctrl", "A+B", "ctrl+B"]
    expected = ["ctrl", "A", "A", "A+B", "B"]
    got = [collapse_label(label) for label in labels]
    if got != expected:
        raise AssertionError((got, expected))

    counts = pd.Series({"ctrl": 300, "A": 60, "B": 70, "A+B": 40, "A+C": 50})
    singles, pairs = discover_eligible_pairs(counts, Thresholds())
    if singles != ["A", "B"] or pairs.to_dict("records") != [
        {"double": "A+B", "single_1": "A", "single_2": "B"}
    ]:
        raise AssertionError((singles, pairs.to_dict("records")))
    print("prepare_norman_direction_inputs.py self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, help="processed AnnData file")
    parser.add_argument("--context-key", default="guide_merged", help="adata.obs column with perturbation labels")
    parser.add_argument("--control-token", default="ctrl", help="label token that marks control")
    parser.add_argument("--separator", default="+", help="separator between targets in a label")
    parser.add_argument("--out-dir", type=Path, default=Path("generated_direction_inputs"), help="output directory")
    parser.add_argument("--chunk-size", type=int, default=512, help="rows read per chunk while aggregating")
    parser.add_argument("--layer", help="AnnData layer to aggregate; default uses adata.X")
    parser.add_argument("--use-raw", action="store_true", help="aggregate adata.raw.X instead of adata.X")
    parser.add_argument("--min-control", type=int, default=200, help="minimum control cells")
    parser.add_argument("--min-single", type=int, default=50, help="minimum cells per single perturbation")
    parser.add_argument("--min-double", type=int, default=30, help="minimum cells per double perturbation")
    parser.add_argument("--expect-cells", type=int, help="fail unless the file has this many cells")
    parser.add_argument("--expect-genes", type=int, help="fail unless the file has this many genes")
    parser.add_argument("--expect-singles", type=int, help="fail unless this many singles are eligible")
    parser.add_argument("--expect-doubles", type=int, help="fail unless this many doubles are evaluable")
    parser.add_argument("--self-test", action="store_true", help="check label collapsing and pair discovery on a small example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.h5ad is None:
        raise SystemExit("--h5ad is required unless --self-test is used")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    thresholds = Thresholds(args.min_control, args.min_single, args.min_double)
    means, counts, metadata = aggregate_h5ad(
        h5ad=args.h5ad,
        context_key=args.context_key,
        control_token=args.control_token,
        separator=args.separator,
        chunk_size=args.chunk_size,
        layer=args.layer,
        use_raw=args.use_raw,
    )
    singles, pairs = discover_eligible_pairs(
        counts, thresholds, args.control_token, args.separator
    )

    metadata.update(
        {
            "min_control": thresholds.min_control,
            "min_single": thresholds.min_single,
            "min_double": thresholds.min_double,
            "n_collapsed_contexts": int(len(counts)),
            "n_eligible_singles": int(len(singles)),
            "n_evaluable_doubles": int(len(pairs)),
        }
    )

    checks = {
        "cells": (args.expect_cells, metadata["n_cells"]),
        "genes": (args.expect_genes, metadata["n_genes"]),
        "singles": (args.expect_singles, metadata["n_eligible_singles"]),
        "doubles": (args.expect_doubles, metadata["n_evaluable_doubles"]),
    }
    mismatches = [
        f"{name}: expected {expected}, got {actual}"
        for name, (expected, actual) in checks.items()
        if expected is not None and int(expected) != int(actual)
    ]
    if mismatches:
        raise RuntimeError("Validation failed: " + "; ".join(mismatches))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    means.reset_index().to_csv(out / "context_means.csv", index=False)
    count_frame = counts.reset_index()
    count_frame["arity"] = count_frame["context"].map(
        lambda label: len(target_tokens(label, args.control_token, args.separator))
    )
    count_frame.to_csv(out / "context_counts_collapsed.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))
    print(f"Wrote {out / 'context_means.csv'}")
    print(f"Wrote {out / 'pairs.csv'}")


if __name__ == "__main__":
    main()
