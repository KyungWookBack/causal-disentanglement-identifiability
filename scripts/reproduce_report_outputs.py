#!/usr/bin/env python3
"""Rebuild the report figures from the included numerical results.

Synthetic summaries are recomputed from per-repetition tables. Norman pair
metrics are recomputed from the saved context means.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from direction_magnitude_decomposition import evaluate_pairs, make_plot, summarize
from experiment_utils import file_sha256
import run_observational_oracle_probe as observational
import run_soft_mmd_probe as soft
import run_perfect_precision_probe as perfect

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10})


def load_and_verify(root: Path, strict: bool = False) -> tuple[dict[str, pd.DataFrame], bool]:
    """Load each synthetic summary and check it matches the summary recomputed from raw.csv."""
    result = {}
    source_issues = []
    for name, module in [('observational', observational), ('soft', soft), ('perfect', perfect)]:
        folder = root / name / 'results'
        raw = pd.read_csv(folder / 'raw.csv')
        stored = pd.read_csv(folder / 'summary.csv')
        computed = module.summarize(raw)
        pd.testing.assert_frame_equal(stored, computed, check_exact=False, atol=1e-10, rtol=1e-10)
        config = json.loads((folder / 'config.json').read_text())
        hashes = config.get('source_sha256', {})
        expected_sources = {Path(module.__file__).name, 'experiment_utils.py'}
        if set(hashes) != expected_sources:
            source_issues.append(f'{name}: incomplete or unexpected source-hash entries')
        for filename in sorted(expected_sources):
            expected = hashes.get(filename)
            if file_sha256(ROOT / 'scripts' / filename) != expected:
                source_issues.append(f'{name}: {filename} differs from the recorded source')
        result[name] = stored
    if source_issues:
        message = 'Source provenance differs: ' + '; '.join(source_issues)
        if strict:
            raise ValueError(message)
        warnings.warn(message + '. Plotting the saved results; rerun the experiments '
                      'to generate results for the current source.', UserWarning, stacklevel=2)
    return result, not source_issues


def synthetic_figure(summaries: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), layout='constrained')
    obs, sm, perf = (summaries[k] for k in ('observational', 'soft', 'perfect'))
    ax = axes[0, 0]
    ax.plot(obs.tau, obs.leaf_recall_mean, 'o-', label='Conditional leaf recall')
    ax.plot(obs.tau, obs.leaf_auc_mean, 's-', label='Conditional leaf AUC')
    ax.set(xscale='symlog', ylim=(-.02, 1.05), xlabel='Hessian-noise strength τ',
           ylabel='Mean score', title='(a) Oracle Hessian; true stages given')
    ax.set_xscale('symlog', linthresh=.1)
    ax.set_xlim(0, max(obs.tau) * 1.2)
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    for ax, col, title in [(axes[0, 1], 'ancestor_auc_mean', '(b) Oracle MMD: ancestor AUC'),
                            (axes[1, 0], 'tc_f1_mean', '(c) Oracle MMD: heuristic F1')]:
        tab = sm.pivot(index='delta', columns='n', values=col).sort_index().sort_index(axis=1)
        im = ax.imshow(tab, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(tab.columns)), tab.columns)
        ax.set_yticks(range(len(tab.index)), tab.index)
        ax.set(xlabel='Samples used per distribution', ylabel='Intervention shift δ', title=title)
        for i in range(len(tab.index)):
            for j in range(len(tab.columns)):
                v = tab.iloc[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white' if v < .45 else 'black')
        fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    ax = axes[1, 1]
    ax.plot(perf.n, perf.rank_leakage_mean, 'o-', color='#176b9b')
    ax.set(xscale='log', xlabel='Samples per context', ylabel='Rank leakage 1 − ρ₂',
           title='(d) Perfect-intervention simulation')
    ax.tick_params(axis='y', colors='#176b9b')
    ax.grid(alpha=.2)
    other = ax.twinx()
    other.plot(perf.n, perf.null_norm_mean, 's--', color='#ae5127')
    other.set_ylabel('Control–control precision difference norm', color='#ae5127')
    other.tick_params(axis='y', colors='#ae5127')
    for ext in ['png', 'pdf']:
        fig.savefig(outdir / f'synthetic_probes.{ext}', dpi=200)
    plt.close(fig)


def verify_norman(data: Path) -> dict:
    """Recompute the pair metrics and their summary from context means."""
    metrics = pd.read_csv(data / 'direction_magnitude_metrics.csv')
    means = pd.read_csv(data / 'context_means.csv', index_col='context')
    pairs = pd.read_csv(data / 'pairs.csv')
    computed = evaluate_pairs(means, 'control', pairs)
    pd.testing.assert_frame_equal(metrics, computed, check_exact=False, atol=1e-10, rtol=1e-10)
    a, c = metrics.magnitude_ratio, metrics.direction_cosine
    err = float(np.max(np.abs(metrics.r2_mean - (2 * a * c - a**2))))
    scaled_err = float(np.max(np.abs(metrics.r2_scaled - c**2)))
    if max(err, scaled_err) > 1e-10:
        raise ValueError('Direction/magnitude identities do not agree')
    saved = json.loads((data / 'direction_magnitude_summary.json').read_text())
    for key, val in summarize(computed).items():
        if not np.isclose(saved[key], val, atol=1e-10, rtol=0):
            raise ValueError(f'Norman summary mismatch: {key}')
    return {'n_pairs': len(metrics), 'max_identity_error': err,
            'max_scaled_identity_error': scaled_err}


def describe_norman(data: Path) -> dict:
    """Cohort-level summaries used in the report's Norman results and discussion."""
    metrics = pd.read_csv(data / 'direction_magnitude_metrics.csv')
    means = pd.read_csv(data / 'context_means.csv', index_col='context')
    norms = pd.Series(np.linalg.norm(means.subtract(means.loc['control']), axis=1), index=means.index)
    strength = metrics.single_1.map(norms) + metrics.single_2.map(norms)
    failed = metrics[metrics.r2_mean < 0]
    return {
        'n_negative_r2': len(failed),
        'min_magnitude_ratio_negative_r2': float(failed.magnitude_ratio.min()),
        'n_negative_r2_with_c_squared_above_0_7': int((failed.r2_scaled > 0.7).sum()),
        'raw_pearson_min': float(metrics.pearson_mean_response.min()),
        'raw_pearson_max': float(metrics.pearson_mean_response.max()),
        'spearman_raw_pearson_r2': float(spearmanr(metrics.pearson_mean_response, metrics.r2_mean).statistic),
        'spearman_magnitude_ratio_sum_single_norms': float(spearmanr(metrics.magnitude_ratio, strength).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-root', type=Path, default=ROOT / 'results/synthetic',
                        help='directory holding the observational/, soft/, and perfect/ result folders')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs',
                        help='where figures/ and verification.json are written')
    parser.add_argument('--norman-root', type=Path, default=ROOT / 'data/norman',
                        help='directory containing Norman context means and result tables')
    parser.add_argument('--strict', action='store_true',
                        help='fail on missing or mismatched source hashes')
    args = parser.parse_args()
    figures = args.outdir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    summaries, source_hashes_match = load_and_verify(args.synthetic_root, strict=args.strict)
    synthetic_figure(summaries, figures)
    norman = args.norman_root
    checks = verify_norman(norman)
    metrics = pd.read_csv(norman / 'direction_magnitude_metrics.csv')
    (args.outdir / 'norman_descriptive_summary.json').write_text(
        json.dumps(describe_norman(norman), indent=2) + '\n')
    make_plot(metrics, figures / 'norman_direction_magnitude.pdf',
              figures / 'norman_direction_magnitude.png', False)
    checks['synthetic_summary_checks'] = 'All three summaries recomputed from per-repetition data.'
    checks['synthetic_source_hashes_match'] = source_hashes_match
    (args.outdir / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
