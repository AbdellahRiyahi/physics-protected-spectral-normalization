"""Mathematical and protocol controls; these are not bridge observations."""
from pathlib import Path
from decimal import Decimal, ROUND_CEILING
import importlib.util
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("aicset240_statistics_v3", HERE / "statistics.py")
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


class StatisticalProtocolControls(unittest.TestCase):
    def test_conformal_requested_rank_avoids_binary_roundoff(self):
        q = STATS.conformal_threshold(np.arange(9), alpha=.7)
        self.assertEqual(q["k"], 3)
        self.assertEqual(q["threshold"], 2.)
        for alpha in [.05, .1, .2, .3, .58, .7]:
            for m in range(1, 80):
                expected = int((Decimal(m + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
                result = STATS.conformal_threshold(np.arange(m), alpha)
                self.assertEqual(result["k"], expected)
                self.assertEqual(result["finite_supported"], expected <= m)

    def test_nominal95_support_boundary_and_abstention(self):
        q18 = STATS.conformal_threshold(np.arange(18))
        q19 = STATS.conformal_threshold(np.arange(19))
        q24 = STATS.conformal_threshold(np.arange(24))
        self.assertFalse(q18["finite_supported"])
        self.assertEqual(q19["k"], 19)
        self.assertEqual(q19["threshold"], 18.)
        self.assertEqual(q24["threshold"], 23.)
        self.assertFalse(q24["assumption_established"])
        result = STATS.coverage_summary(pd.Series([0., 1.]), q18)
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["covered_groups"])

    def test_empty_calibration_is_explicitly_unsupported(self):
        q = STATS.conformal_threshold([])
        self.assertFalse(q["finite_supported"])
        self.assertEqual(q["reason"], "NO_CALIBRATION_GROUPS")
        self.assertTrue(np.isinf(q["threshold"]))

    def test_group_max_uses_physical_ids_and_supports_tuple_ids(self):
        result = STATS.grouped_max([1., 3., 2.], ["date_a", "date_a", "date_b"])
        self.assertEqual(result.to_dict(), {"date_a": 3., "date_b": 2.})
        tuples = [("bridge_a", "day_1"), ("bridge_a", "day_1"), ("bridge_a", "day_2")]
        self.assertEqual(STATS.grouped_max([1., 3., 2.], tuples).to_list(), [3., 2.])

    def test_missing_or_nonfinite_groups_and_scores_fail_closed(self):
        for groups in [["date_a", ""], ["date_a", "  "], ["date_a", None], [("a", "b"), ("a", None)]]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                STATS.grouped_max([1., 2.], groups)
        for values in [[1., np.nan], [1., np.inf], [1., -1.]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                STATS.conformal_threshold(values)
        with self.assertRaises(ValueError):
            STATS.conformal_threshold(pd.Series([1., 2.], index=["same", "same"]))

    def test_wilson_boundary_and_known_small_sample(self):
        low, high = STATS.wilson_interval(10, 10)
        self.assertAlmostEqual(low, .7224672001371107)
        self.assertEqual(high, 1.)
        self.assertEqual(STATS.wilson_interval(0, 10)[0], 0.)
        with self.assertRaises(ValueError):
            STATS.wilson_interval(1, 0)

    def test_threshold_equality_is_covered_not_alerted(self):
        result = STATS.coverage_summary(pd.Series([0., 1., 2.]), 1.)
        self.assertEqual(result["covered_groups"], 2)
        self.assertEqual(result["alert_groups"], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)

    def test_exact_mcnemar_one_improvement_is_not_significant(self):
        raw = pd.Series([1.] * 9 + [3.])
        corrected = pd.Series([1.] * 10)
        result = STATS.paired_decision_comparison(raw, corrected, 2., 2.)
        self.assertEqual(result["raw_alert_only_groups"], 1)
        self.assertEqual(result["corrected_alert_only_groups"], 0)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.)
        self.assertAlmostEqual(result["paired_coverage_difference"], .1)
        self.assertIn("independent paired groups", result["mcnemar_note"])

    def test_exact_mcnemar_known_all_discordant_case(self):
        result = STATS.paired_decision_comparison(pd.Series([2.] * 10), pd.Series([0.] * 10), 1., 1.)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 2 / 2 ** 10)

    def test_pairing_aligns_ids_and_does_not_mutate_inputs(self):
        raw = pd.Series([2., 0.], index=["a", "b"])
        corrected = pd.Series([0., 2.], index=["b", "a"])
        original_raw, original_corrected = raw.copy(), corrected.copy()
        result = STATS.paired_decision_comparison(raw, corrected, 1., 1.)
        self.assertEqual(result["discordant_groups"], 0)
        pd.testing.assert_series_equal(raw, original_raw)
        pd.testing.assert_series_equal(corrected, original_corrected)
        with self.assertRaises(ValueError):
            STATS.paired_decision_comparison(raw, pd.Series([0., 2.], index=["a", "c"]), 1., 1.)

    def test_decisions_do_not_compare_incompatible_score_units(self):
        raw = pd.Series([0., 1e-320], index=["a", "b"])
        other_family = pd.Series([1e308, 0.], index=["a", "b"])
        result = STATS.paired_decision_comparison(raw, other_family, 0., 1e307)
        self.assertEqual(result["raw_alert_only_groups"], 1)
        self.assertEqual(result["corrected_alert_only_groups"], 1)
        self.assertFalse(any("reduction" in key or "compression" in key for key in result))
        with self.assertRaises(ValueError):
            STATS.paired_group_comparison(pd.Series([1e-320]), pd.Series([1e308]), bootstrap_reps=0)

    def test_median_paired_reduction_is_not_ratio_of_medians(self):
        a, b = pd.Series([1., 10., 10.]), pd.Series([.9, 1., 9.])
        result = STATS.paired_group_comparison(a, b, bootstrap_reps=1000, seed=4)
        repeat = STATS.paired_group_comparison(a, b, bootstrap_reps=1000, seed=4)
        self.assertAlmostEqual(result["median_paired_relative_reduction"], .1)
        self.assertAlmostEqual(result["ratio_of_medians_reduction"], .9)
        self.assertEqual(result["bootstrap_replicates_performed"], 1000)
        self.assertEqual(result["median_paired_relative_reduction_ci95_low"], repeat["median_paired_relative_reduction_ci95_low"])
        self.assertEqual(result["median_paired_relative_reduction_ci95_high"], repeat["median_paired_relative_reduction_ci95_high"])

    def test_zero_baselines_do_not_create_relative_improvements(self):
        result = STATS.paired_group_comparison(pd.Series([0., 0.]), pd.Series([0., 0.]), 0., 0., bootstrap_reps=50)
        self.assertIsNone(result["median_paired_relative_reduction"])
        self.assertEqual(result["n_nonpositive_raw_groups_excluded_from_relative_reductions"], 2)
        self.assertEqual(result["bootstrap_replicates_performed"], 0)
        self.assertEqual(result["bootstrap_status"], "NO_POSITIVE_RAW_PAIRS")
        self.assertIsNone(result["median_paired_relative_reduction_ci95_low"])

    def test_one_positive_pair_bootstrap_is_marked_degenerate(self):
        result = STATS.paired_group_comparison(pd.Series([2.]), pd.Series([1.]), bootstrap_reps=50)
        self.assertEqual(result["bootstrap_status"], "DEGENERATE_SINGLE_PAIR")
        self.assertEqual(result["median_paired_relative_reduction_ci95_low"], .5)

    def test_damage_auc_and_nominal_abstention_are_separate(self):
        scores = pd.Series([1., 2., 3., 4.])
        labels = pd.Series(["reference", "reference", "altered", "altered"])
        q = STATS.conformal_threshold(pd.Series([1.]))
        result, roc = STATS.damage_discrimination(scores, labels, q)
        self.assertEqual(result["auc"], 1.)
        self.assertEqual(result["decision_status"], "ABSTAIN")
        self.assertIsNone(result["sensitivity"])
        self.assertEqual(roc.iloc[0].true_positive_rate, 0.)
        self.assertEqual(roc.iloc[-1].true_positive_rate, 1.)
        descriptive, _ = STATS.damage_discrimination(scores, labels, 2.)
        self.assertEqual(descriptive["sensitivity"], 1.)
        self.assertEqual(descriptive["specificity"], 1.)

    def test_tied_auc_is_half(self):
        result, roc = STATS.damage_discrimination(pd.Series([1.] * 4), pd.Series(["reference"] * 2 + ["altered"] * 2), 1.)
        self.assertEqual(result["auc"], .5)
        self.assertEqual(result["sensitivity"], 0.)
        self.assertEqual(result["specificity"], 1.)
        self.assertEqual(len(roc), 2)

    def test_altered_only_set_retains_observed_detection_counts(self):
        scores = pd.Series([2., 0.], index=["scenario_03", "scenario_04"])
        labels = pd.Series(["altered", "altered"], index=scores.index)
        result, roc = STATS.damage_discrimination(scores, labels, 1.)
        self.assertEqual(result["status"], "NOT_ESTIMABLE")  # AUC is unavailable.
        self.assertEqual(result["decision_status"], "PARTIAL_CLASS_DECISIONS")
        self.assertEqual(result["sensitivity"], .5)
        self.assertEqual(result["true_positive_groups"], 1)
        self.assertEqual(result["false_negative_groups"], 1)
        self.assertIsNone(result["auc"])
        self.assertIsNone(result["specificity"])
        self.assertTrue(roc.empty)

    def test_reference_only_set_retains_observed_false_alert_counts(self):
        result, roc = STATS.damage_discrimination(pd.Series([2., 0.]), pd.Series(["reference", "reference"]), 1.)
        self.assertEqual(result["false_alert_rate"], .5)
        self.assertEqual(result["specificity"], .5)
        self.assertIsNone(result["sensitivity"])
        self.assertIsNone(result["auc"])
        self.assertTrue(roc.empty)

    def test_unknown_labels_are_counted_and_not_relabelled(self):
        result, _ = STATS.damage_discrimination(pd.Series([1., 2.]), pd.Series(["reference", "unknown"]), 1.)
        self.assertEqual(result["n_unknown_label_groups_excluded"], 1)
        self.assertEqual(result["n_groups_assessed"], 1)
        self.assertIsNone(result["auc"])
        with self.assertRaises(ValueError):
            STATS.damage_discrimination(pd.Series([1.], index=["a"]), pd.Series(["altered"], index=["b"]), 1.)

    def test_multiindex_group_pairing_is_supported(self):
        idx = pd.MultiIndex.from_tuples([("bridge", "a"), ("bridge", "b")])
        result = STATS.paired_decision_comparison(pd.Series([1., 2.], index=idx), pd.Series([0., 1.], index=idx), 1., 1.)
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["raw_alert_only_groups"], 1)

    def test_holm_adjustment_and_missing_entries(self):
        result = STATS.holm_adjust(pd.Series([.01, np.nan, .04, .03], index=list("abcd")))
        np.testing.assert_allclose(result.loc[["a", "c", "d"]].to_numpy(), [.03, .06, .06])
        self.assertTrue(np.isnan(result["b"]))
        self.assertEqual(result.name, "p_adjusted_holm")
        with self.assertRaises(ValueError):
            STATS.holm_adjust([1.1])

    def test_invalid_replication_counts_raise_value_error(self):
        for value in [True, .5, np.inf, np.nan, -1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                STATS.paired_group_comparison(pd.Series([1.]), pd.Series([.5]), bootstrap_reps=value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
