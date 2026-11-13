"""Executable numerical regressions on controlled fixtures, not SHM outcomes.

Run with: python -m unittest discover -s work/aicset_revision_v4 -p test_models_v4.py -v
The tests require no raw dataset, HPC account, plotting, or archived results.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

spec = importlib.util.spec_from_file_location("aicset_v3_models", Path(__file__).with_name("models.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SpectralExtractionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.signal = rng.normal(size=(1024, 3))

    def test_synchronous_cpsd_channel_permutation_and_amplitude(self):
        f, s = m.cpsd_singular_spectra(self.signal, 100, nperseg=128)
        fp, sp = m.cpsd_singular_spectra(self.signal[:, [2, 0, 1]], 100, nperseg=128)
        fa, sa = m.cpsd_singular_spectra(7 * self.signal, 100, nperseg=128)
        assert_array_equal(f, fp); assert_array_equal(f, fa)
        assert_allclose(s, sp, rtol=1e-12, atol=1e-13)
        assert_allclose(sa, 49 * s, rtol=1e-12, atol=1e-12)
        self.assertEqual(s.shape, (64, 2))
        self.assertTrue(np.all(s[:, 0] >= s[:, 1]))

    def test_rank_one_and_single_welch_segment_rejected(self):
        with self.assertRaisesRegex(ValueError, "second singular"):
            m.cpsd_singular_spectra(np.tile(self.signal[:, :1], (1, 2)), 100, nperseg=128)
        with self.assertRaisesRegex(ValueError, "Welch segments"):
            m.cpsd_singular_spectra(self.signal[:128], 100, nperseg=128)
        # A small, genuinely identifiable second channel is retained.
        weak = np.column_stack([self.signal[:, 0], self.signal[:, 0] + 1e-5 * self.signal[:, 1]])
        _, s = m.cpsd_singular_spectra(weak, 100, nperseg=128)
        self.assertGreater(s[:, 1].max(), 0)

    def test_fractional_integer_settings_rejected(self):
        for value in (128.4, True, np.inf):
            with self.assertRaises(ValueError):
                m.cpsd_singular_spectra(self.signal, 100, nperseg=value)
        with self.assertRaises(ValueError): m.SpectralNormalizer(rank=2.1)
        with self.assertRaises(ValueError): m.ConventionalPCA(rank=True)
        with self.assertRaises(ValueError): m.FrequencyRatioBaseline(peaks_per_curve=2.2)

    def test_grid_floor_and_normalization_scale_invariance(self):
        f, s = m.cpsd_singular_spectra(self.signal, 100, nperseg=128)
        a = m.grid_log_curves(f, s, 4)
        b = m.grid_log_curves(f, 13 * s, 4)
        assert_allclose(m.normalize_curves(a), m.normalize_curves(b), atol=1e-13)
        with self.assertRaisesRegex(ValueError, "support"):
            m.grid_log_curves(f, s, .01)
        with self.assertRaises(ValueError): m.grid_log_curves(f, s, 4, bins=128.1)
        with self.assertRaises(ValueError): m.normalize_curves(np.ones((2, 128)))

    def test_f1_uses_first_prominent_training_peak(self):
        f = np.linspace(.1, 20, 1000)
        s1 = .001 + np.exp(-((f - 4) / .1) ** 2) + 3 * np.exp(-((f - 10) / .2) ** 2)
        spectra = [(f, a * np.column_stack([s1, .3 * s1])) for a in (1, 2, 10)]
        f1, audit = m.estimate_reference_f1(spectra, band=(1, 15))
        self.assertAlmostEqual(f1, 4, delta=.03)
        self.assertEqual(audit["selected_f1_hz"], f1)
        with self.assertRaises(ValueError): m.estimate_reference_f1([(f, np.ones((len(f), 2)))])


class ProtectedModelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(92)
        self.X = self.rng.normal(size=(80, 40))
        self.groups = np.repeat(np.arange(10), 8).astype(str)
        self.U = np.linalg.qr(self.rng.normal(size=(40, 5)))[0]
        self.model = m.SpectralNormalizer(rank=3).fit(self.X, self.groups, protected_basis=self.U)

    def test_nnls_constraints_and_protected_coordinate_invariance(self):
        query = self.rng.normal(size=(25, 40))
        corrected = self.model.transform(query)
        assert_allclose(self.U.T @ self.model.B_, 0, atol=1e-13)
        assert_allclose((corrected - query) @ self.U, 0, atol=1e-13)
        coefficients = self.model.coefficients(query)
        self.assertTrue(np.all(coefficients >= 0))
        self.assertTrue(np.all(self.model.score(query) <= self.model.score(query, "raw") + 1e-12))
        # An orthonormal cone has this independent closed-form projection.
        expected = np.maximum((query - self.model.mean_) @ self.model.B_, 0)
        assert_allclose(coefficients, expected, atol=2e-14)

    def test_protection_rank_and_orthonormality_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            m.SpectralNormalizer(rank=3).fit(np.ones((12, 40)), np.repeat(["a", "b", "c"], 4))
        damaged_u = self.U.copy(); damaged_u[:, 0] *= 1 + 1e-6
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            m.SpectralNormalizer().fit(self.X, self.groups, protected_basis=damaged_u)
        with self.assertRaises(ValueError): m.SpectralNormalizer().fit(self.X, [None] * len(self.X))

    def test_explicit_source_vectors_and_mean_do_not_use_evaluation(self):
        vectors = self.rng.normal(size=(9, 40))
        model = m.SpectralNormalizer().fit(self.X, self.groups, protected_basis=self.U, nuisance_vectors=vectors)
        assert_allclose(model.mean_, m.group_centroids(self.X, self.groups)[0].mean(axis=0))
        saved = model.B_.copy(); model.score(self.rng.normal(size=(10, 40)) * 1e6)
        assert_array_equal(model.B_, saved)
        self.assertEqual(model.metadata_["nuisance_vectors"], "EXPLICIT_TRAINING_VECTORS")

    def test_finite_secants_are_protected_and_shift_edges_explicit(self):
        rho = np.linspace(.25, 2.5, 128)
        L = np.stack([np.sin(7 * rho) + .2 * rho, np.cos(5 * rho) + rho])
        U = m.build_protected_basis(L, grid=rho)
        assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-12)
        shifted = L.copy(); shifted[0] = m.shift_log_curves(L, .01, grid=rho)[0]
        delta = m.normalize_curves(shifted) - m.normalize_curves(L)
        assert_allclose(delta - U @ (U.T @ delta), 0, atol=1e-12)
        shifted_all = m.shift_log_curves(L, .02, grid=rho)
        assert_array_equal(shifted_all[:, 0], L[:, 0])
        with self.assertRaises(ValueError): m.shift_log_curves(L, .01, grid=rho * np.nan)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(72)
        self.X = self.rng.normal(size=(40, 8))
        self.groups = np.repeat(["a", "b", "c", "d", "e"], 8)

    def test_conventional_pca_matches_weighted_covariance(self):
        X = self.X[:28]; g = self.groups[:28]
        model = m.ConventionalPCA(3).fit(X, g)
        w = m.group_weights(g); w /= w.sum()
        centered = X - np.average(X, axis=0, weights=w)
        covariance = (centered * w[:, None]).T @ centered
        eig, basis = np.linalg.eigh(covariance); basis = basis[:, -3:]
        assert_allclose(model.basis_ @ model.basis_.T, basis @ basis.T, atol=1e-12)
        assert_allclose(model.explained_variance_, eig[::-1][:3], atol=1e-12)
        # Repeating every acquisition from one group changes no group's mass.
        mask = g == "a"
        doubled = m.ConventionalPCA(3).fit(np.vstack([X, X[mask], X[mask]]), np.r_[g, g[mask], g[mask]])
        assert_allclose(model.mean_, doubled.mean_, atol=1e-14)
        assert_allclose(model.basis_ @ model.basis_.T, doubled.basis_ @ doubled.basis_.T, atol=1e-12)

    def test_ridge_fitting_only_preprocessors_and_no_row_broadcast(self):
        C = np.linspace(-10, 20, 40)[:, None]; C[3] = np.nan
        Y = np.column_stack([np.nan_to_num(C[:, 0], nan=5), np.zeros(40)]) + self.X[:, :2] * .02
        for piecewise in (False, True):
            model = m.TemperatureRidge(piecewise=piecewise, alpha_grid=(.001, .1), cv_splits=3).fit(Y, C, self.groups)
            self.assertEqual(len(model.cv_scores_), 2)
            before = {k: v.copy() for k, v in model.params_.items()}
            model.score(np.ones((3, 2)), np.array([[1e8], [np.nan], [-1e8]]))
            for key, value in before.items(): assert_array_equal(value, model.params_[key])
            with self.assertRaisesRegex(ValueError, "same rows"):
                model.transform(np.ones((3, 2)), np.ones((1, 1)))

    def ratio_fixture(self):
        grid = np.linspace(.25, 2.5, 128)
        curves = []
        for shift in np.linspace(-.003, .003, 10):
            # Exact concave quadratics near each distinct sigma1 maximum.
            first = -100 * (grid - (.73 + shift)) ** 2
            second = -.2 - 70 * (grid - (1.54 + shift)) ** 2
            sigma1 = np.maximum(first, second)
            sigma2 = np.cos(grid * 9)  # ignored by the default ratio baseline
            curves.append(np.stack([sigma1, sigma2]))
        return grid, np.asarray(curves)

    def test_peak_ratios_use_distinct_sigma1_peaks_and_subbin_estimates(self):
        grid, L = self.ratio_fixture()
        model = m.FrequencyRatioBaseline(grid=grid).fit(L, np.repeat(["a", "b"], 5))
        self.assertEqual([band[0] for band in model.bands_], [0, 0])
        values, diagnostics = model.tracked_frequencies(L)
        shifts = np.linspace(-.003, .003, 10)
        assert_allclose(values[:, 0], .73 + shifts, atol=1e-13)
        assert_allclose(values[:, 1], 1.54 + shifts, atol=1e-13)
        self.assertTrue(diagnostics["quadratic_interpolation_used"].all())
        self.assertFalse(diagnostics["band_edge_maximum"].any())
        assert_allclose(model.features(L)[:, 0], (1.54 + shifts) / (.73 + shifts), atol=1e-13)
        altered_sigma2 = L.copy(); altered_sigma2[:, 1] *= 1e5
        assert_array_equal(model.features(L), model.features(altered_sigma2))
        edge = np.tile(grid, (2, 1))
        _, flags = model.tracked_frequencies(edge)
        self.assertTrue(flags["band_edge_maximum"].all())
        with self.assertRaises(ValueError): model.fit(np.ones_like(L), np.repeat(["a", "b"], 5))


class SerializationTests(unittest.TestCase):
    setUp = BaselineTests.setUp
    ratio_fixture = BaselineTests.ratio_fixture

    def test_all_fitted_model_roundtrips_without_refit(self):
        U = np.linalg.qr(self.rng.normal(size=(8, 2)))[0]
        spectrum = m.SpectralNormalizer(3).fit(self.X, self.groups, protected_basis=U)
        pca = m.ConventionalPCA(3).fit(self.X, self.groups)
        C = self.rng.normal(size=(40, 2)); C[0, 0] = np.nan
        ridge = m.TemperatureRidge(fixed_alpha=.1).fit(self.X, C, self.groups)
        piecewise = m.TemperatureRidge(piecewise=True, fixed_alpha=.1).fit(self.X, C[:, :1], self.groups)
        grid, L = self.ratio_fixture(); ratios = m.FrequencyRatioBaseline(grid).fit(L)
        with tempfile.TemporaryDirectory() as folder:
            for name, model, query, cov in [("spectral", spectrum, self.X, None), ("pca", pca, self.X, None),
                    ("ridge", ridge, self.X, C), ("piecewise", piecewise, self.X, C[:, :1]), ("ratios", ratios, L, None)]:
                prefix = Path(folder) / (name + ".version.3")
                files = model.save(prefix, extra={"f1_hz": np.float64(4), "grid_for_audit": np.array([1., 2.])})
                restored = type(model).load(files["metadata"])
                if cov is not None:
                    assert_array_equal(model.transform(query, cov), restored.transform(query, cov))
                    assert_array_equal(model.score(query, cov), restored.score(query, cov))
                elif name == "spectral":
                    for method in m.METHODS:
                        assert_array_equal(model.transform(query, method), restored.transform(query, method))
                        assert_array_equal(model.score(query, method), restored.score(query, method))
                else:
                    assert_array_equal(model.transform(query), restored.transform(query))
                    assert_array_equal(model.score(query), restored.score(query))
                with np.load(files["arrays"], allow_pickle=False) as z:
                    self.assertTrue(all(z[k].dtype.kind in "biuf" for k in z.files))

    def test_mismatched_archives_and_metadata_overrides_rejected(self):
        model = m.ConventionalPCA(3).fit(self.X, self.groups)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            with self.assertRaisesRegex(ValueError, "override"):
                model.save(path, extra={"rank": 2})
            self.assertFalse(path.with_suffix(".npz").exists())
            paths = model.save(path)
            original = Path(paths["arrays"]).read_bytes()
            Path(paths["arrays"]).write_bytes(original + b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                m.ConventionalPCA.load(path)
            Path(paths["arrays"]).write_bytes(original)
            meta = json.loads(Path(paths["metadata"]).read_text())
            meta["model_type"] = "TemperatureRidge"
            Path(paths["metadata"]).write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Expected"):
                m.ConventionalPCA.load(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
