#!/usr/bin/env python3
"""Rebuild figures; optionally rerun synthetic experiments and numerical checks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(arguments: list[str]) -> None:
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    print('+', ' '.join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs',
                        help='output directory for figures, verification.json, and any reruns')
    parser.add_argument('--rerun-synthetic', action='store_true',
                        help='rerun the three synthetic probes with default settings before plotting (slow)')
    parser.add_argument('--smoke-tests', action='store_true',
                        help='run the unit tests and the self-tests of the Norman scripts first')
    parser.add_argument('--strict', action='store_true',
                        help='fail on missing or mismatched source hashes')
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    source = ROOT / 'results/synthetic'
    if args.rerun_synthetic:
        source = outdir / 'synthetic'
        for name, script in [('observational', 'run_observational_oracle_probe.py'),
                             ('soft', 'run_soft_mmd_probe.py'), ('perfect', 'run_perfect_precision_probe.py')]:
            run([sys.executable, str(ROOT / 'scripts' / script), '--outdir', str(source / name)])
    if args.smoke_tests:
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        run([sys.executable, 'scripts/prepare_norman_direction_inputs.py', '--self-test'])
        run([sys.executable, 'scripts/direction_magnitude_decomposition.py', '--self-test'])
        run([sys.executable, 'scripts/run_norman_realdata_experiment.py', '--self-test',
             '--outdir', str(outdir / 'norman_self_test')])
    run([sys.executable, 'scripts/reproduce_report_outputs.py',
         '--synthetic-root', str(source), '--outdir', str(outdir)]
        + (['--strict'] if args.strict else []))


if __name__ == '__main__':
    main()
