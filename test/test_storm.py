import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from storm.storm import (
    StormGraph,
    gpu_enabled,
    power2trt,
    power_storm,
    powertrt,
    prepare_storm_graph,
    storm,
    storm2trt,
    stormtrt,
)


class TestStorm(unittest.TestCase):
    def test_basic_storm(self):
        """Test storm with simple test data."""
        np.random.seed(42)
        M = 100  # spots
        N = 10   # genes

        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        exp_mat = np.random.rand(M, N)

        result = storm(coords, exp_mat, k_nn=10, approx=True)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('gene_names', result.columns)
        self.assertIn('p_values', result.columns)
        self.assertIn('effect_size', result.columns)
        self.assertEqual(len(result), N)
        self.assertTrue(all(0 <= p <= 1 for p in result['p_values']))

    def test_storm_with_dataframe(self):
        """Test storm with pandas DataFrame input."""
        np.random.seed(42)
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)

        exp_df = pd.DataFrame(
            np.random.rand(100, 5),
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
        )

        result = storm(coords, exp_df, k_nn=10)
        self.assertEqual(set(result['gene_names']), set(exp_df.columns))

    def test_storm_with_sparse_matrix(self):
        """Test that sparse and dense storm paths agree."""
        np.random.seed(42)
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        exp_dense = np.random.rand(100, 5).astype(np.float32)
        exp_sparse = csr_matrix(exp_dense)

        result_dense = storm(coords, exp_dense, k_nn=10)
        result_sparse = storm(coords, exp_sparse, k_nn=10)

        np.testing.assert_allclose(
            result_sparse["p_values"].to_numpy(),
            result_dense["p_values"].to_numpy(),
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            result_sparse["effect_size"].to_numpy(),
            result_dense["effect_size"].to_numpy(),
            rtol=5e-5,
            atol=5e-5,
        )

    @unittest.skipUnless(gpu_enabled, "CUDA is not available")
    def test_storm_sparse_gpu_equivalence(self):
        rng = np.random.default_rng(42)
        coords = rng.normal(size=(100, 3))
        exp_sparse = csr_matrix(rng.poisson(1.0, size=(100, 8)).astype(np.float32))

        result_cpu = storm(coords, exp_sparse, k_nn=10, use_gpu=False)
        result_gpu = storm(coords, exp_sparse, k_nn=10, use_gpu=True)

        np.testing.assert_allclose(
            result_gpu["p_values"], result_cpu["p_values"], rtol=1e-4, atol=1e-3
        )
        np.testing.assert_allclose(
            result_gpu["effect_size"],
            result_cpu["effect_size"],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_storm_excludes_self_from_knn(self):
        """Test that storm uses true neighbors rather than self-matches."""
        coords = np.array([[0.0], [1.0], [2.0]], dtype=float)
        exp_mat = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)

        result = storm(coords, exp_mat, k_nn=1, approx=True)

        self.assertAlmostEqual(result["effect_size"].iloc[0], 0.25, places=6)
        self.assertTrue(0 <= result["p_values"].iloc[0] <= 1)

    def test_storm_handles_duplicate_coordinates(self):
        """Duplicate locations must not break self-neighbor removal."""
        coords = np.array([[0.0], [0.0], [0.0], [1.0]])
        exp_mat = np.arange(4, dtype=np.float32)[:, None]

        result = storm(coords, exp_mat, k_nn=1)

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result.loc[0, "p_values"]))

    def test_storm_rejects_invalid_neighbor_count(self):
        coords = np.arange(4, dtype=float)[:, None]
        exp_mat = coords.copy()
        for invalid in (0, -1, 1.5, True):
            with self.subTest(k_nn=invalid), self.assertRaises(ValueError):
                storm(coords, exp_mat, k_nn=invalid)

    def test_storm_constant_gene_is_neutral(self):
        coords = np.arange(10, dtype=float)[:, None]
        result = storm(coords, np.ones((10, 1)), k_nn=2)
        self.assertEqual(result.loc[0, "p_values"], 1.0)
        self.assertEqual(result.loc[0, "effect_size"], 0.0)

    def test_storm_approx_vs_exact(self):
        """Test that approx and exact modes both work."""
        np.random.seed(42)
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        exp_mat = np.random.rand(100, 5)

        result_approx = storm(coords, exp_mat, k_nn=10, approx=True)
        result_exact = storm(coords, exp_mat, k_nn=10, approx=False)

        self.assertEqual(len(result_approx), 5)
        self.assertEqual(len(result_exact), 5)
        self.assertTrue(all(0 <= p <= 1 for p in result_approx['p_values']))
        self.assertTrue(all(0 <= p <= 1 for p in result_exact['p_values']))

    @unittest.skipUnless(gpu_enabled, "CUDA is not available")
    def test_storm_exact_gpu_equivalence(self):
        rng = np.random.default_rng(7)
        coords = rng.normal(size=(80, 2))
        exp_mat = rng.lognormal(size=(80, 6)).astype(np.float32)

        result_cpu = storm(coords, exp_mat, k_nn=8, approx=False, use_gpu=False)
        result_gpu = storm(coords, exp_mat, k_nn=8, approx=False, use_gpu=True)

        np.testing.assert_allclose(
            result_gpu["p_values"], result_cpu["p_values"], rtol=1e-4, atol=1e-3
        )
        np.testing.assert_allclose(
            result_gpu["effect_size"],
            result_cpu["effect_size"],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_storm_default_knn(self):
        """Test that k_nn defaults to 50 (approx) or 100 (exact)."""
        np.random.seed(42)
        coords = np.random.rand(150, 2)
        exp_mat = np.random.rand(150, 3)

        result_approx = storm(coords, exp_mat, approx=True)
        self.assertEqual(len(result_approx), 3)

        result_exact = storm(coords, exp_mat, approx=False)
        self.assertEqual(len(result_exact), 3)

        self.assertTrue(all(0 <= p <= 1 for p in result_approx['p_values']))
        self.assertTrue(all(0 <= p <= 1 for p in result_exact['p_values']))

    def test_storm_real_data(self):
        """Test storm with real test data."""
        input_file = Path(__file__).parent / "test_data" / "scenario1_RW1_3-5_1"
        input_data = pd.read_csv(input_file)

        coords = input_data[["x", "y", "z"]].to_numpy()
        exp_mat = input_data.iloc[:, 3:].to_numpy()

        result = storm(coords, exp_mat, k_nn=50)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), exp_mat.shape[1])
        sig_count = (result['p_values'] < 0.05).sum()
        self.assertGreater(sig_count, 0)


class TestPreparedGraph(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.coords = rng.normal(size=(100, 3))
        self.expression = rng.poisson(1.2, size=(100, 8)).astype(np.float32)
        self.graph = prepare_storm_graph(self.coords, k_nn=10)

    def test_graph_is_immutable_and_describes_knn(self):
        self.assertIsInstance(self.graph, StormGraph)
        self.assertEqual(self.graph.n_spots, 100)
        self.assertEqual(self.graph.k_nn, 10)
        self.assertEqual(self.graph.patches_raw.nnz, 1000)
        with self.assertRaises(ValueError):
            self.graph.coords[0, 0] = 0
        with self.assertRaises(ValueError):
            self.graph.patches_raw.data[0] = 0

    def test_cached_dense_and_sparse_results_match(self):
        for expression in (self.expression, csr_matrix(self.expression)):
            with self.subTest(sparse=issparse(expression)):
                expected = storm(self.coords, expression, k_nn=10)
                result = storm(self.coords, expression, graph=self.graph)
                pd.testing.assert_frame_equal(result, expected, check_exact=True)

    def test_cached_exact_results_match_and_constant_is_reused(self):
        expected = storm(
            self.coords, self.expression, k_nn=10, approx=False
        )
        result = storm(
            self.coords, self.expression, approx=False, graph=self.graph
        )
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        self.assertIsNotNone(self.graph._test_constant)
        self.assertEqual(
            self.graph.exact_test_constant(), self.graph._test_constant
        )

    @unittest.skipUnless(gpu_enabled, "CUDA is not available")
    def test_cached_gpu_results_match(self):
        for approx in (True, False):
            with self.subTest(approx=approx):
                expected = storm(
                    self.coords,
                    self.expression,
                    k_nn=10,
                    approx=approx,
                    use_gpu=True,
                )
                result = storm(
                    self.coords,
                    self.expression,
                    approx=approx,
                    use_gpu=True,
                    graph=self.graph,
                )
                pd.testing.assert_frame_equal(result, expected, check_exact=True)

    def test_cache_mismatch_is_rejected(self):
        changed_coords = self.coords.copy()
        changed_coords[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "coordinates do not match"):
            storm(changed_coords, self.expression, graph=self.graph)
        with self.assertRaisesRegex(ValueError, "k_nn does not match"):
            storm(self.coords, self.expression, k_nn=9, graph=self.graph)
        with self.assertRaises(TypeError):
            storm(self.coords, self.expression, graph=object())


class TestStormTrt(unittest.TestCase):
    def test_basic_stormtrt(self):
        """Test stormtrt with simple data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'Group': ['Control'] * 50 + ['Treatment'] * 50,
            'Pvalue': np.concatenate([
                np.random.uniform(0, 1, 50),
                np.random.uniform(0, 0.2, 50)
            ])
        })

        result = stormtrt(data, control='Control', sig_level=0.05)

        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('conf_int', result)
        self.assertIn('estimate', result)
        self.assertGreater(result['estimate']['prop_treatment'], result['estimate']['prop_control'])
        self.assertLess(result['p_value'], 0.05)

    def test_stormtrt_missing_columns(self):
        """Test that stormtrt raises error for missing columns."""
        data = pd.DataFrame({'Group': ['A', 'B'], 'Value': [0.1, 0.2]})

        with self.assertRaises(ValueError):
            stormtrt(data, control='A')

    def test_stormtrt_matches_upstream_pooled_test(self):
        data = pd.DataFrame({
            "Group": ["Control"] * 10 + ["Treatment"] * 10,
            "Pvalue": [0.01] * 2 + [0.5] * 8 + [0.01] * 6 + [0.5] * 4,
        })

        result = stormtrt(data, control="Control")
        pooled = 8 / 20
        expected_z = (0.6 - 0.2) / np.sqrt(pooled * (1 - pooled) * 0.2)
        self.assertAlmostEqual(result["statistic"], expected_z)
        self.assertTrue(np.isposinf(result["conf_int"][1]))

    def test_stormtrt_conf_level_compatibility_alias(self):
        data = pd.DataFrame({
            "Group": ["Control"] * 4 + ["Treatment"] * 4,
            "Pvalue": [0.01, 0.5, 0.5, 0.5, 0.01, 0.02, 0.5, 0.5],
        })
        expected = stormtrt(data, control="Control", sig_level=0.05)
        with self.assertWarns(DeprecationWarning):
            result = stormtrt(data, control="Control", conf_level=0.95)
        self.assertAlmostEqual(result["p_value"], expected["p_value"])


class TestStorm2Trt(unittest.TestCase):
    def test_basic_storm2trt(self):
        """Test storm2trt with simple data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'Group': ['A'] * 10 + ['B'] * 10,
            'EffectSize': np.concatenate([
                np.random.normal(0.06, 0.02, 10),
                np.random.normal(0.02, 0.03, 10)
            ]),
            'M': np.random.randint(3000, 5000, 20),
            'K': [50] * 20
        })

        result = storm2trt(data, alternative='two.sided', conf_level=0.95)

        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('conf_int', result)
        self.assertIn('estimate', result)
        self.assertTrue(0 <= result['p_value'] <= 1)

    def test_storm2trt_alternatives(self):
        """Test all alternative hypotheses."""
        data = pd.DataFrame({
            'Group': ['A'] * 10 + ['B'] * 10,
            'EffectSize': np.concatenate([
                np.random.normal(0.05, 0.01, 10),
                np.random.normal(0.05, 0.01, 10)
            ]),
            'M': [4000] * 20,
            'K': [50] * 20
        })

        for alt in ['two.sided', 'less', 'greater']:
            result = storm2trt(data, alternative=alt)
            self.assertEqual(result['alternative'], alt)
            self.assertTrue(0 <= result['p_value'] <= 1)

    def test_one_sided_confidence_intervals_contain_estimate(self):
        data = pd.DataFrame({
            "Group": ["A"] * 3 + ["B"] * 3,
            "EffectSize": [0.10, 0.11, 0.09, 0.02, 0.03, 0.01],
            "M": [100] * 6,
            "K": [5] * 6,
        })
        estimate = 0.08
        greater = storm2trt(data, alternative="greater")["conf_int"]
        less = storm2trt(data, alternative="less")["conf_int"]
        self.assertLess(greater[0], estimate)
        self.assertGreater(less[1], estimate)


class TestPowerStorm(unittest.TestCase):
    def test_solve_for_n(self):
        """Test solving for sample size."""
        result = power_storm(es=0.06, n=None, power=0.80, sig_level=0.05)

        self.assertIn('n', result)
        self.assertIn('power', result)
        self.assertGreater(result['n'], 0)
        self.assertAlmostEqual(result['power'], 0.80, places=4)

    def test_solve_for_power(self):
        """Test solving for power."""
        result = power_storm(es=0.06, n=424, power=None, sig_level=0.05)

        self.assertIn('power', result)
        self.assertTrue(0 <= result['power'] <= 1)

    def test_solve_for_effect_size(self):
        """Test solving for effect size."""
        result = power_storm(es=None, n=424, power=0.80, sig_level=0.05)

        self.assertIn('effect_size', result)
        self.assertGreater(result['effect_size'], 0)

    def test_solve_for_sig_level(self):
        """Test solving for significance level."""
        result = power_storm(es=0.06, n=424, power=0.8, sig_level=None)

        self.assertIn('sig_level', result)
        self.assertTrue(0 <= result['sig_level'] <= 1)

    def test_error_on_multiple_none(self):
        """Test that error is raised when multiple parameters are None."""
        with self.assertRaises(ValueError):
            power_storm(es=None, n=None, power=0.8, sig_level=0.05)

    def test_effect_size_matches_upstream_k_rule(self):
        low_k = power_storm(es=None, n=424, power=0.8, sig_level=0.05, k=5)
        high_k = power_storm(es=None, n=424, power=0.8, sig_level=0.05, k=20)
        self.assertAlmostEqual(
            low_k["effect_size"], high_k["effect_size"], places=12
        )


class TestPowerTrt(unittest.TestCase):
    def test_solve_for_sample_size_and_power(self):
        sample = powertrt(nsample=None, power=0.9, sig_level=0.05)
        achieved = powertrt(nsample=7, power=None, sig_level=0.05, ratio=1.5)
        self.assertGreater(sample["nsample"], 0)
        self.assertTrue(0 <= achieved["power"] <= 1)

    def test_exact_power(self):
        result = powertrt(
            nsample=7, power=None, sig_level=0.05, ratio=1.5, exact=True
        )
        self.assertTrue(0 <= result["power"] <= 1)


class TestPower2Trt(unittest.TestCase):
    def test_solve_for_power_and_sample_size(self):
        args = dict(
            delta=0.1,
            phi=0.01,
            psi1=0.02,
            psi2=0.02,
            ratio=1,
            sig_level=0.05,
        )
        achieved = power2trt(**args, nsample=50, power=None)
        required = power2trt(**args, nsample=None, power=0.8)
        self.assertTrue(0 <= achieved["power"] <= 1)
        self.assertGreater(required["nsample"], 1)


if __name__ == "__main__":
    unittest.main()
