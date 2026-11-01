"""Numerical checks for the synthetic probes and the Norman pipeline."""
from pathlib import Path
import importlib.util
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_perfect_precision_probe as perfect
import run_observational_oracle_probe as observational
import run_soft_mmd_probe as soft
import run_norman_realdata_experiment as norman
import reproduce_report_outputs as report_outputs


class NumericalChecks(unittest.TestCase):
    def test_scperturb_target_labels(self):
        names, kinds = norman.context_arrays(
            np.array(['control', 'KLF1', 'CEBPE_KLF1']), target_separator='_')
        self.assertEqual(names.tolist(), ['control', 'KLF1', 'CEBPE+KLF1'])
        self.assertEqual(kinds.tolist(), ['control', 'single', 'double'])

    @unittest.skipUnless(importlib.util.find_spec('anndata'), 'optional AnnData dependency')
    def test_block_h5ad_preprocessing_matches_in_memory(self):
        import anndata as ad
        rng = np.random.default_rng(10)
        counts = rng.poisson(np.linspace(1, 25, 24), size=(200, 24)).astype(np.float32)
        counts[140:, -1] += 1000
        labels = np.array(['control'] * 80 + ['A'] * 60 + ['A_B'] * 60)
        genes = np.array([f'G{i}' for i in range(24)])
        for storage in (np.asarray, sparse.csr_matrix, sparse.csc_matrix):
            for normalize in (True, False):
                with self.subTest(storage=storage.__name__, normalize=normalize), tempfile.TemporaryDirectory() as tmp:
                    X = storage(counts)
                    reference = norman.normalize_log1p(X) if normalize else X
                    selected = norman.select_variable_genes(reference[:140], 8, 50000, 0)
                    expected = reference[:, selected]
                    expected = expected.toarray() if sparse.issparse(expected) else expected
                    data = ad.AnnData(X, obs=pd.DataFrame({'perturbation': labels},
                                     index=[f'cell{i}' for i in range(len(labels))]),
                                     var=pd.DataFrame(index=genes))
                    path = Path(tmp) / 'counts.h5ad'
                    data.write_h5ad(path)
                    loaded = norman.load_h5ad(path, 'perturbation', None, n_genes=8,
                                             normalize=normalize, target_separator='_', block_size=5)
                    np.testing.assert_array_equal(loaded.genes, genes[selected])
                    actual = loaded.X.toarray() if sparse.issparse(loaded.X) else loaded.X
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_precision_support_and_reconstruction(self):
        rng = np.random.default_rng(41)
        a, w, g = perfect.parameters(rng)
        x = perfect.sample(1000, a, w, g, rng)
        gp = np.linalg.pinv(g)
        z = x @ gp.T
        reference = gp.T @ np.linalg.inv(np.cov(z, rowvar=False)) @ gp
        result = perfect.precision(x)
        np.testing.assert_allclose(result, reference, atol=1e-11, rtol=1e-11)
        null_projection = np.eye(perfect.P) - g @ gp
        self.assertLess(np.linalg.norm(result @ null_projection), 1e-10)

    def test_population_intervention_rank(self):
        rng = np.random.default_rng(4)
        a, w, g = perfect.parameters(rng)
        gp = np.linalg.pinv(g)
        def theta(a, w):
            b = np.diag(1 / np.sqrt(w)) @ (np.eye(len(w)) - a)
            return gp.T @ b.T @ b @ gp
        base = theta(a, w)
        for target in range(perfect.D):
            a1, w1 = perfect.intervene(a, w, target, 4.)
            difference = theta(a1, w1) - base
            self.assertLessEqual(np.linalg.matrix_rank(difference, tol=1e-10), 2)
            self.assertAlmostEqual(perfect.rho2(difference), 1., places=12)

    def test_oracle_hessian_matches_finite_differences(self):
        z = observational.sample_latents(10, 82)
        h = 1e-4
        for active in [[0, 1, 2, 3, 4], [0, 1, 2, 3]]:
            def logp(x):
                residual = observational.residuals(x)[:, active]
                return -.5 * np.sum((residual / observational.SIGMA[active])**2, axis=1)
            actual = observational.oracle_diag_hessian(z, active)
            for col, node in enumerate(active):
                plus, minus = z.copy(), z.copy()
                plus[:, node] += h
                minus[:, node] -= h
                numeric = (logp(plus) - 2 * logp(z) + logp(minus)) / h**2
                np.testing.assert_allclose(actual[:, col], numeric, atol=2e-5, rtol=2e-5)

    def test_conditional_leaf_signal_without_noise(self):
        raw = observational.run_probe(80, 3, [0.], 12)
        self.assertTrue(np.all(raw.leaf_recall == 1.))
        self.assertTrue(np.all(raw.leaf_auc == 1.))
        self.assertEqual(set(raw.stage), {0, 1, 2})

    def test_block_kernel_uses_all_rows(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=517), rng.normal(size=411)
        reference = np.exp(-.7 * (x[:, None] - y[None, :])**2).mean()
        self.assertAlmostEqual(soft.rbf_mean(x, y, .7, block_size=64), reference, places=14)
        np.testing.assert_array_equal(soft.subsample(x, 0, rng), x)
        self.assertEqual(len(soft.subsample(x, 250, rng)), 250)

    def test_mmd_identical_empirical_distribution(self):
        x = np.random.default_rng(3).normal(size=301)
        self.assertLess(soft.mmd2_rbf(x, x, 0, np.random.default_rng(4)), 1e-12)

    def test_sparse_dense_pca_agree(self):
        data = norman.self_test_data(5)
        names, kinds = norman.context_arrays(data.labels)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'results').mkdir(); (folder / 'figures').mkdir()
            dense = norman.lowrank_analysis(data.X, names, kinds, folder, 5, 30, 0)
            sp = norman.lowrank_analysis(sparse.csr_matrix(data.X), names, kinds, folder, 5, 30, 0)
            pd.testing.assert_frame_equal(dense, sp, atol=1e-6, rtol=1e-6)

    def test_main_uses_training_genes_and_shared_prediction_features(self):
        from unittest.mock import patch
        data = norman.self_test_data(0)
        names, kinds = norman.context_arrays(data.labels)
        expected = data.X[(kinds == 'control') | (kinds == 'single')]
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['run_norman_realdata_experiment.py', '--self-test', '--skip-lowrank',
                    '--n-hvgs', '10', '--outdir', tmp]
            with patch.object(sys, 'argv', argv), patch.object(
                    norman, 'select_variable_genes', wraps=norman.select_variable_genes) as select:
                norman.main()
            np.testing.assert_array_equal(select.call_args.args[0], expected)
            results = Path(tmp) / 'results'
            a = pd.read_csv(results / 'double_prediction_metrics.csv').set_index('pair')
            b = pd.read_csv(results / 'direction_magnitude_metrics.csv').set_index('double')
            self.assertTrue(np.all(b.n_genes == 10))
            np.testing.assert_allclose(a.r2_mean_additive_vs_control,
                                       b.loc[a.index, 'r2_mean'], atol=2e-6, rtol=2e-6)
            means = pd.read_csv(results / 'context_means.csv', index_col='context')
            pairs = pd.read_csv(results / 'pairs.csv')
            recomputed = norman.evaluate_pairs(means, 'control', pairs).set_index('double')
            pd.testing.assert_frame_equal(b, recomputed, atol=1e-10, rtol=1e-10)


class ReproductionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        project = Path(__file__).resolve().parents[1]
        shutil.copytree(project / 'scripts', self.root / 'scripts')
        shutil.copytree(project / 'results/synthetic', self.root / 'results/synthetic')
        self.source = self.root / 'results/synthetic'

    def test_comment_change_warns_by_default_and_fails_in_strict_mode(self):
        script = self.root / 'scripts/run_observational_oracle_probe.py'
        script.write_text(script.read_text() + '\n# Documentation-only change.\n')
        with patch.object(report_outputs, 'ROOT', self.root):
            with self.assertWarnsRegex(UserWarning, 'run_observational_oracle_probe.py'):
                summaries, matched = report_outputs.load_and_verify(self.source)
            self.assertEqual(set(summaries), {'observational', 'soft', 'perfect'})
            self.assertFalse(matched)
            with self.assertRaisesRegex(ValueError, 'run_observational_oracle_probe.py'):
                report_outputs.load_and_verify(self.source, strict=True)

    def test_incorrect_summary_fails_in_both_modes(self):
        path = self.source / 'observational/results/summary.csv'
        summary = pd.read_csv(path)
        summary.loc[0, 'leaf_recall_mean'] = -1.
        summary.to_csv(path, index=False)
        with patch.object(report_outputs, 'ROOT', self.root):
            for strict in (False, True):
                with self.subTest(strict=strict), self.assertRaises(AssertionError):
                    report_outputs.load_and_verify(self.source, strict=strict)


if __name__ == '__main__':
    unittest.main()
