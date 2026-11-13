"""Physical-group statistical summaries for the AICSET 240 revision.

These helpers operate on one score per physical group. They do not turn windows,
setups, dates, or scenarios into independent replications. Confidence intervals
and paired tests are reported with explicit assumptions, not as proof that those
assumptions hold for the recorded bridge data.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal, ROUND_CEILING

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, rankdata


INFERENCE_NOTE = (
    "Descriptive or exploratory group-level inference. Group independence, "
    "exchangeability, and prospective temporal validity have not been established."
)
BOOTSTRAP_NOTE = (
    "Percentile resampling of paired physical groups assumes that resampling "
    "groups is appropriate. It conditions on the frozen fitted and calibrated "
    "rules and does not include fitting or calibration uncertainty. Serial "
    "dependence can invalidate nominal coverage."
)
MCNEMAR_NOTE = (
    "Exact two-sided McNemar test conditions on discordant paired binary "
    "decisions and assumes independent paired groups conditional on the frozen "
    "fitted and calibrated rules. Rules must not be selected using these "
    "evaluation groups. Reducing reference-state alerts does not establish "
    "retained sensitivity to structural damage."
)


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be a finite integer")
    value = int(number)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _missing_group_id(value) -> bool:
    """Recognize missing IDs, including empty strings and tuple components."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, tuple):
        return any(_missing_group_id(part) for part in value)
    marker = pd.isna(value)
    if np.ndim(marker) != 0:
        raise ValueError("Physical-group identifiers must be hashable scalars or tuples")
    if bool(marker):
        return True
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("Physical-group identifiers must be hashable scalars or tuples") from exc
    return False


def _validate_group_index(index: pd.Index, name: str):
    if not index.is_unique:
        raise ValueError(f"{name} must contain exactly one value per unique group")
    # MultiIndex.hasnans is not implemented by pandas; inspect its tuple IDs.
    if any(_missing_group_id(value) for value in index.to_list()):
        raise ValueError(f"{name} has a missing physical-group identifier")


def _group_series(values, name: str, allow_empty: bool = True) -> pd.Series:
    result = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    _validate_group_index(result.index, name)
    result = pd.to_numeric(result, errors="raise").astype(float)
    if not allow_empty and result.empty:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(result.to_numpy())):
        raise ValueError(f"{name} contains a missing or nonfinite score")
    if np.any(result.to_numpy() < 0):
        raise ValueError(f"{name} contains a negative distance score")
    result.name = name
    return result


def _threshold_details(threshold) -> tuple[float, bool, str]:
    """Read a finite scalar threshold or the conformal-threshold result dict."""
    if isinstance(threshold, Mapping):
        reason = str(threshold.get("reason", "UNSPECIFIED"))
        value = threshold.get("threshold")
        if value is None:
            return math.inf, False, reason
        q = float(value)
        supported = bool(threshold.get("finite_supported", math.isfinite(q)))
    elif threshold is None:
        return math.inf, False, "NO_THRESHOLD_PROVIDED"
    else:
        q = float(threshold)
        supported = math.isfinite(q)
        reason = "FINITE_THRESHOLD" if supported else "NO_FINITE_THRESHOLD"
    if math.isnan(q) or q == -math.inf or q < 0:
        raise ValueError("A distance threshold must be nonnegative or +infinity")
    supported = supported and math.isfinite(q)
    return q, supported, reason


def grouped_max(scores, groups) -> pd.Series:
    """Aggregate acquisition scores to maxima of explicitly supplied groups.

    Positional matching is deliberate: ``scores[i]`` belongs to ``groups[i]``.
    Do not supply time windows as if they were independent physical groups.
    """
    values = np.asarray(scores, dtype=float)
    if isinstance(groups, (str, bytes)):
        raise ValueError("groups must be a sequence of explicit physical-group IDs")
    identifiers = list(groups)
    if values.ndim != 1 or len(values) != len(identifiers):
        raise ValueError("scores and groups must be equally sized 1D sequences")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Acquisition distance scores must be finite and nonnegative")
    if any(_missing_group_id(v) for v in identifiers):
        raise ValueError("Every acquisition needs an explicit physical-group ID")
    if not identifiers:
        return pd.Series(dtype=float, name="group_max_score").rename_axis("group_id")
    frame = pd.DataFrame({"group_id": identifiers, "score": values})
    result = frame.groupby("group_id", sort=False, dropna=False)["score"].max()
    return result.rename("group_max_score")


def conformal_threshold(calibration_scores, alpha: float = 0.05) -> dict:
    """Split-conformal order statistic for physical-group scores.

    If the requested rank exceeds the observed calibration sample size, return
    +infinity with ``finite_supported=False``. Downstream decision summaries then
    abstain instead of misreporting an infinite threshold as perfect coverage.
    An empty calibration set is returned as an explicitly unsupported result.
    """
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    values = _group_series(calibration_scores, "calibration_score").to_numpy()
    m = int(values.size)
    # Decimal conversion uses the requested decimal alpha, avoiding e.g.
    # ceil(10 * (1 - .7)) == 4 from binary roundoff instead of the correct 3.
    requested_rank = Decimal(m + 1) * (Decimal(1) - Decimal(str(alpha)))
    k = int(requested_rank.to_integral_value(rounding=ROUND_CEILING))
    if m == 0:
        reason = "NO_CALIBRATION_GROUPS"
    elif k > m:
        reason = "REQUESTED_ORDER_STATISTIC_EXCEEDS_CALIBRATION_COUNT"
    else:
        reason = "FINITE_ORDER_STATISTIC_SUPPORTED"
    supported = m > 0 and 1 <= k <= m
    q = float(np.sort(values)[k - 1]) if supported else math.inf
    return {
        "m": m,
        "k": k,
        "alpha": alpha,
        "nominal_reference_coverage": 1.0 - alpha,
        "threshold": q,
        "finite_supported": supported,
        "reason": reason,
        "decision_rule": "alert iff score > threshold",
        "guarantee_condition": "Exchangeable calibration and future group scores",
        "rank_computation": "EXACT_DECIMAL_ALPHA_ORDER_STATISTIC",
        "assumption_established": False,
    }


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson binomial interval; applicability needs independent Bernoulli units."""
    k = _integer(k, "k")
    n = _integer(n, "n", minimum=1)
    confidence = float(confidence)
    if k > n:
        raise ValueError("k cannot exceed n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = k / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4 * n * n)) / denominator
    lower = 0.0 if k == 0 else max(0.0, center - half_width)
    upper = 1.0 if k == n else min(1.0, center + half_width)
    return lower, upper


def coverage_summary(eval_group_scores, threshold) -> dict:
    """Reference-state coverage, or explicit abstention without a finite rule."""
    scores = _group_series(eval_group_scores, "evaluation_score")
    q, supported, reason = _threshold_details(threshold)
    result = {
        "n_groups": int(len(scores)),
        "threshold": q,
        "finite_supported": supported,
        "status": "NOT_ESTIMABLE" if scores.empty else ("OK" if supported else "ABSTAIN"),
        "reason": "NO_EVALUATION_GROUPS" if scores.empty else reason,
        "covered_groups": None,
        "alert_groups": None,
        "coverage": None,
        "wilson_95_low": None,
        "wilson_95_high": None,
        "inference_note": INFERENCE_NOTE,
    }
    if scores.empty or not supported:
        return result
    covered = int((scores <= q).sum())
    lo, hi = wilson_interval(covered, len(scores))
    result.update({
        "covered_groups": covered,
        "alert_groups": int(len(scores) - covered),
        "coverage": covered / len(scores),
        "wilson_95_low": lo,
        "wilson_95_high": hi,
    })
    return result


def _aligned_scores(raw, corrected) -> tuple[pd.Series, pd.Series]:
    raw = _group_series(raw, "raw_score", allow_empty=False)
    corrected = _group_series(corrected, "corrected_score", allow_empty=False)
    if len(raw) != len(corrected) or not raw.index.isin(corrected.index).all():
        raise ValueError("Paired methods must be evaluated on exactly the same group IDs")
    return raw, corrected.reindex(raw.index)


def paired_decision_comparison(
    raw: pd.Series, corrected: pd.Series, raw_threshold=None, corrected_threshold=None
) -> dict:
    """Compare paired binary rules without comparing continuous score units.

    This helper is suitable for different feature families. Each method applies
    its own finite threshold directly, so a threshold of zero remains valid.
    It deliberately returns no score-compression or score-ratio quantities.
    """
    raw, corrected = _aligned_scores(raw, corrected)
    a, b = raw.to_numpy(), corrected.to_numpy()
    result = {
        "n_groups": int(len(a)),
        "mcnemar_status": "ABSTAIN",
        "mcnemar_reason": "FINITE_THRESHOLDS_REQUIRED_FOR_BOTH_METHODS",
        "raw_alert_only_groups": None,
        "corrected_alert_only_groups": None,
        "discordant_groups": None,
        "raw_reference_coverage": None,
        "corrected_reference_coverage": None,
        "paired_coverage_difference": None,
        "mcnemar_exact_two_sided_p": None,
        "mcnemar_note": MCNEMAR_NOTE,
        "inference_note": INFERENCE_NOTE,
    }
    qr, raw_supported, _ = _threshold_details(raw_threshold)
    qc, corrected_supported, _ = _threshold_details(corrected_threshold)
    if not (raw_supported and corrected_supported):
        return result
    raw_alert, corrected_alert = a > qr, b > qc
    improved = int(np.sum(raw_alert & ~corrected_alert))
    worsened = int(np.sum(~raw_alert & corrected_alert))
    discordant = improved + worsened
    pvalue = float(binomtest(improved, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    raw_coverage = float(np.mean(~raw_alert))
    corrected_coverage = float(np.mean(~corrected_alert))
    result.update({
        "mcnemar_status": "EXPLORATORY_EXACT_TEST",
        "mcnemar_reason": "PAIRED_FINITE_THRESHOLD_DECISIONS" if discordant else "NO_DISCORDANT_PAIRS",
        "raw_alert_only_groups": improved,
        "corrected_alert_only_groups": worsened,
        "discordant_groups": discordant,
        "raw_reference_coverage": raw_coverage,
        "corrected_reference_coverage": corrected_coverage,
        "paired_coverage_difference": corrected_coverage - raw_coverage,
        "mcnemar_exact_two_sided_p": pvalue,
    })
    return result


def paired_group_comparison(
    raw: pd.Series,
    corrected: pd.Series,
    raw_threshold=None,
    corrected_threshold=None,
    bootstrap_reps: int = 10_000,
    seed: int = 20261113,
) -> dict:
    """Paired descriptive reductions and an exploratory exact McNemar test.

    Both Series must have the same unique physical-group IDs. Relative changes
    use positive raw scores only, with exclusions counted explicitly. Bootstrap
    samples resample *paired groups*, never individual windows or acquisitions.
    """
    raw, corrected = _aligned_scores(raw, corrected)
    a = raw.to_numpy()
    b = corrected.to_numpy()
    positive = a > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reductions = 1.0 - b[positive] / a[positive]
    if not np.isfinite(reductions).all():
        raise ValueError("Relative score reductions overflow; inspect score scaling before reporting")
    bootstrap_reps = _integer(bootstrap_reps, "bootstrap_reps")
    result = {
        "n_groups": int(len(a)),
        "n_positive_raw_groups": int(positive.sum()),
        "n_nonpositive_raw_groups_excluded_from_relative_reductions": int((~positive).sum()),
        "median_raw_score": float(np.median(a)),
        "median_corrected_score": float(np.median(b)),
        "median_absolute_reduction": float(np.median(a - b)),
        "median_paired_relative_reduction": float(np.median(reductions)) if len(reductions) else None,
        "ratio_of_medians_reduction": float(1.0 - np.median(b) / np.median(a)) if np.median(a) > 0 else None,
        "relative_reduction_unit": "fraction",
        "bootstrap_replicates": bootstrap_reps,
        "bootstrap_replicates_performed": 0,
        "bootstrap_status": "NOT_REQUESTED" if not bootstrap_reps else "NO_POSITIVE_RAW_PAIRS",
        "bootstrap_seed": int(seed),
        "median_paired_relative_reduction_ci95_low": None,
        "median_paired_relative_reduction_ci95_high": None,
        "bootstrap_note": BOOTSTRAP_NOTE,
        "mcnemar_status": "ABSTAIN",
        "mcnemar_reason": "FINITE_THRESHOLDS_REQUIRED_FOR_BOTH_METHODS",
        "raw_alert_only_groups": None,
        "corrected_alert_only_groups": None,
        "discordant_groups": None,
        "raw_reference_coverage": None,
        "corrected_reference_coverage": None,
        "paired_coverage_difference": None,
        "mcnemar_exact_two_sided_p": None,
        "mcnemar_note": MCNEMAR_NOTE,
        "inference_note": INFERENCE_NOTE,
    }
    if len(reductions) and bootstrap_reps:
        rng = np.random.default_rng(seed)
        medians = np.empty(bootstrap_reps, dtype=float)
        batch_size = max(1, min(1024, 1_000_000 // len(reductions)))
        for start in range(0, bootstrap_reps, batch_size):
            stop = min(start + batch_size, bootstrap_reps)
            indices = rng.integers(0, len(reductions), size=(stop - start, len(reductions)))
            medians[start:stop] = np.median(reductions[indices], axis=1)
        lo, hi = np.quantile(medians, [0.025, 0.975])
        result.update({
            "median_paired_relative_reduction_ci95_low": float(lo),
            "median_paired_relative_reduction_ci95_high": float(hi),
            "bootstrap_replicates_performed": bootstrap_reps,
            "bootstrap_status": "DEGENERATE_SINGLE_PAIR" if len(reductions) == 1 else "EXPLORATORY_PAIRED_PERCENTILE",
        })
    result.update(paired_decision_comparison(raw, corrected, raw_threshold, corrected_threshold))
    return result


def damage_discrimination(
    scores: pd.Series, labels: pd.Series, thresholdinfo=None
) -> tuple[dict, pd.DataFrame]:
    """Descriptive reference-versus-altered discrimination on evaluation groups.

    The caller must exclude all fitting/calibration groups and validate that
    labels describe actual structural states. ``altered`` is not automatically a
    proven damage class. Accepted aliases are ref/healthy and damage/damaged;
    other labels are counted and excluded explicitly. AUC requires both classes;
    a finite rule can still yield sensitivity for altered-only groups, or false
    alerts for reference-only groups. No AUC confidence interval or independence
    claim is generated from sequential damage scenarios.
    """
    scores = _group_series(scores, "evaluation_score")
    if not isinstance(labels, pd.Series):
        raise ValueError("labels must be a Series with unique, nonmissing group IDs")
    _validate_group_index(labels.index, "labels")
    if not scores.index.isin(labels.index).all():
        raise ValueError("Some score groups have no structural-state label")
    labels = labels.reindex(scores.index)
    aliases = {"reference": 0, "ref": 0, "healthy": 0, "altered": 1, "damage": 1, "damaged": 1}
    encoded = labels.astype("string").str.strip().str.lower().map(aliases)
    known = encoded.notna()
    values = scores.loc[known].to_numpy()
    y = encoded.loc[known].to_numpy(dtype=int)
    n_ref = int(np.sum(y == 0))
    n_alt = int(np.sum(y == 1))
    q, supported, reason = _threshold_details(thresholdinfo)
    result = {
        "status": "DESCRIPTIVE_DISCRIMINATION" if n_ref and n_alt else "NOT_ESTIMABLE",
        "n_groups_supplied": int(len(scores)),
        "n_groups_assessed": int(len(values)),
        "n_unknown_label_groups_excluded": int((~known).sum()),
        "n_reference_groups": n_ref,
        "n_altered_groups": n_alt,
        "auc": None,
        "auc_status": "DESCRIPTIVE_AUC" if n_ref and n_alt else "NOT_ESTIMABLE",
        "auc_reason": "BOTH_CLASSES_AVAILABLE" if n_ref and n_alt else "BOTH_STRUCTURAL_STATE_CLASSES_REQUIRED",
        "small_class_note": "At least one class has fewer than two physical groups; no population discrimination precision is established." if min(n_ref, n_alt) < 2 else "",
        "higher_score_is_altered": True,
        "roc_decision_rule": "score >= roc_threshold (standard ROC convention)",
        "threshold": q,
        "finite_supported": supported,
        "decision_status": "OK" if supported and n_ref and n_alt else (
            "PARTIAL_CLASS_DECISIONS" if supported and len(values) else "ABSTAIN"),
        "decision_reason": reason if supported and n_ref and n_alt else (
            "FINITE_THRESHOLD_ONE_CLASS_ONLY" if supported and len(values) else (
                reason if not supported else "NO_KNOWN_STATE_GROUPS")),
        "true_positive_groups": None,
        "false_negative_groups": None,
        "false_positive_groups": None,
        "true_negative_groups": None,
        "sensitivity": None,
        "specificity": None,
        "false_alert_rate": None,
        "inference_note": INFERENCE_NOTE,
        "label_note": "Altered-state discrimination does not by itself establish damage localization or severity.",
    }
    columns = ["roc_threshold", "true_positive_rate", "false_positive_rate", "true_positive_groups", "false_positive_groups"]
    rows = []
    if n_ref and n_alt:
        ranks = rankdata(values, method="average")
        result["auc"] = float((ranks[y == 1].sum() - n_alt * (n_alt + 1) / 2.0) / (n_ref * n_alt))
        for threshold in np.r_[np.inf, np.sort(np.unique(values))[::-1]]:
            alert = values >= threshold
            tp = int(np.sum(alert & (y == 1)))
            fp = int(np.sum(alert & (y == 0)))
            rows.append({"roc_threshold": float(threshold), "true_positive_rate": tp / n_alt,
                         "false_positive_rate": fp / n_ref, "true_positive_groups": tp, "false_positive_groups": fp})
    if supported and len(values):
        alert = values > q
        tp = int(np.sum(alert & (y == 1)))
        fp = int(np.sum(alert & (y == 0)))
        if n_alt:
            result.update({"true_positive_groups": tp, "false_negative_groups": n_alt - tp,
                           "sensitivity": tp / n_alt})
        if n_ref:
            result.update({"false_positive_groups": fp, "true_negative_groups": n_ref - fp,
                           "specificity": (n_ref - fp) / n_ref, "false_alert_rate": fp / n_ref})
    return result, pd.DataFrame(rows, columns=columns)


def holm_adjust(pvalues):
    """Holm-adjust available p-values; preserve missing entries and Series IDs."""
    is_series = isinstance(pvalues, pd.Series)
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    valid = np.isfinite(values)
    if np.any(np.isinf(values)) or np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError("p-values must lie in [0, 1], or be missing")
    adjusted = np.full(len(values), np.nan, dtype=float)
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(values[valid], kind="stable")]
    if len(order):
        candidate = values[order] * np.arange(len(order), 0, -1)
        adjusted[order] = np.minimum(1.0, np.maximum.accumulate(candidate))
    if is_series:
        return pd.Series(adjusted, index=pvalues.index, name="p_adjusted_holm")
    return adjusted
