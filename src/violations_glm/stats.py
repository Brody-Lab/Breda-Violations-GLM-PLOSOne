"""Statistical analyses used by publication figures."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import (
    linregress,
    pearsonr,
    shapiro,
    spearmanr,
    ttest_1samp,
    ttest_rel,
    wilcoxon,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from violations_glm.features import shift_within_sessions
from violations_glm.transforms import RELATIVE_DELAY_ORDER, session_position_trials


def within_subject_test(
    frame: pd.DataFrame,
    *,
    subject_col: str,
    condition_col: str,
    value_col: str,
    condition_labels: dict[Any, str] | None = None,
) -> dict[str, Any]:
    """Paired within-subject test used by Figure 4 hit-rate and psychometric stats.

    Prints descriptives, Shapiro–Wilk on paired differences, then a paired
    t-test (if differences are normal) or Wilcoxon signed-rank otherwise.
    """
    wide = frame.pivot(index=subject_col, columns=condition_col, values=value_col)
    if condition_labels:
        wide = wide.rename(columns=condition_labels)
    conditions = list(wide.columns)
    if len(conditions) != 2:
        raise ValueError(
            f"within_subject_test expects exactly two conditions, got {conditions}"
        )
    first, second = conditions

    mean_a, sd_a = wide[first].mean(), wide[first].std()
    mean_b, sd_b = wide[second].mean(), wide[second].std()
    n_subjects = len(wide)
    print(f"{n_subjects} subjects/replicates:")
    print(f" • {first}: mean = {mean_a:.3f}, SD = {sd_a:.3f}")
    print(f" • {second}: mean = {mean_b:.3f}, SD = {sd_b:.3f}")

    differences = wide[second] - wide[first]
    w_stat, p_norm = shapiro(differences)
    print(f"\nShapiro–Wilk: W = {w_stat:.3f}, p = {p_norm:.3f}")

    if p_norm > 0.05:
        t_stat, p_value = ttest_rel(wide[second], wide[first])
        print(f"Paired t-test: t = {t_stat}, p = {p_value}")
        return {"test": "ttest", "stat": float(t_stat), "p": float(p_value)}

    w_signed, p_signed = wilcoxon(wide[second], wide[first])
    print(f"Wilcoxon signed-rank: W = {w_signed}, p = {p_signed}")
    return {"test": "wilcoxon", "stat": float(w_signed), "p": float(p_signed)}


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - values.mean()
    denominator = np.dot(centered, centered) or 1
    positive = np.array(
        [
            np.dot(centered[:-lag], centered[lag:]) / denominator
            for lag in range(1, max_lag + 1)
        ]
    )
    return np.concatenate((positive[::-1], np.array([1.0]), positive))


def violation_sessions(frame: pd.DataFrame, *, max_lag: int) -> list[np.ndarray]:
    sessions = []
    for _, group in frame.groupby("animal_id_session", sort=False):
        values = group.sort_values("trial")["violation"].to_numpy()
        if len(values) > max_lag and values.any():
            sessions.append(values)
    return sessions


def session_autocorrelations(
    sessions: Sequence[np.ndarray], *, max_lag: int
) -> np.ndarray:
    return np.stack([autocorrelation(values, max_lag) for values in sessions])


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((n_bootstrap, values.shape[1]))
    for index in range(n_bootstrap):
        selection = rng.integers(0, len(values), size=len(values))
        means[index] = values[selection].mean(axis=0)
    tail = (1 - confidence) * 50
    return (
        np.percentile(means, tail, axis=0),
        np.percentile(means, 100 - tail, axis=0),
    )


def permutation_distribution(
    sessions: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    max_lag: int,
    n_permutations: int,
) -> np.ndarray:
    """Return one reusable permutation distribution of session means."""
    output = np.empty((n_permutations, 2 * max_lag + 1))
    batch_size = 64
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        size = stop - start
        positive_total = np.zeros((size, max_lag))
        for values in sessions:
            permutations = np.broadcast_to(values, (size, len(values))).copy()
            rng.permuted(permutations, axis=1, out=permutations)
            centered = permutations - values.mean()
            denominator = np.dot(values - values.mean(), values - values.mean()) or 1
            for lag in range(1, max_lag + 1):
                positive_total[:, lag - 1] += (
                    np.einsum(
                        "ij,ij->i",
                        centered[:, :-lag],
                        centered[:, lag:],
                        optimize=True,
                    )
                    / denominator
                )
        positive = positive_total / len(sessions)
        output[start:stop, :max_lag] = positive[:, ::-1]
        output[start:stop, max_lag] = 1.0
        output[start:stop, max_lag + 1 :] = positive
    return output


def _permutation_chunk(
    sessions: Sequence[np.ndarray],
    max_lag: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    return permutation_distribution(
        sessions,
        rng=np.random.default_rng(seed),
        max_lag=max_lag,
        n_permutations=n_permutations,
    )


def parallel_permutation_distribution(
    sessions: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    max_lag: int,
    n_permutations: int,
    workers: int,
) -> np.ndarray:
    """Compute deterministic chunks and concatenate one reusable null array."""
    if workers <= 1:
        return permutation_distribution(
            sessions,
            rng=rng,
            max_lag=max_lag,
            n_permutations=n_permutations,
        )
    chunk_sizes = np.full(workers, n_permutations // workers, dtype=int)
    chunk_sizes[: n_permutations % workers] += 1
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=workers, dtype=np.uint64)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _permutation_chunk,
                    sessions,
                    max_lag,
                    int(chunk_size),
                    int(seed),
                )
                for chunk_size, seed in zip(chunk_sizes, seeds, strict=True)
                if chunk_size
            ]
            return np.concatenate([future.result() for future in futures])
    except OSError:
        return permutation_distribution(
            sessions,
            rng=rng,
            max_lag=max_lag,
            n_permutations=n_permutations,
        )


def empirical_autocorrelation_pvalues(
    observed: np.ndarray,
    null_distribution: np.ndarray,
    *,
    max_lag: int,
) -> pd.DataFrame:
    """Test unique positive lags and mirror results to negative lags."""
    rows = []
    for lag in range(1, max_lag + 1):
        position = max_lag + lag
        extreme = np.count_nonzero(
            np.abs(null_distribution[:, position]) >= abs(observed[position])
        )
        raw = (extreme + 1) / (len(null_distribution) + 1)
        corrected = min(raw * max_lag, 1.0)
        rows.append(
            {
                "lag": lag,
                "p_value": raw,
                "p_value_bonferroni": corrected,
                "significant": corrected < 0.05,
            }
        )
    positive = pd.DataFrame(rows)
    negative = positive.assign(lag=-positive["lag"])
    return pd.concat([negative, positive], ignore_index=True).sort_values("lag")


def hit_violation_regressions(
    session_percentages: pd.DataFrame,
    animal_means: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the submitted MixedLM, OLS, and animal-mean linear regressions.

    Rates are expected on a percentage scale, matching the original Figure 1
    analysis. Returns a JSON-serializable statistics dictionary and a residual
    table for the MixedLM diagnostic scatter.
    """
    mixed = smf.mixedlm(
        "viol_rate ~ hit_rate",
        data=session_percentages,
        groups=session_percentages["animal_id"],
    )
    mixed_result = mixed.fit()
    fixed_intercept = float(mixed_result.fe_params["Intercept"])
    fixed_slope = float(mixed_result.fe_params["hit_rate"])
    pred_fixed = (
        fixed_intercept + fixed_slope * session_percentages["hit_rate"].to_numpy()
    )
    residuals = pd.DataFrame(
        {
            "fitted": mixed_result.fittedvalues.to_numpy(),
            "residual": mixed_result.resid.to_numpy(),
        }
    )

    ols_x = sm.add_constant(animal_means[["hit_rate"]])
    ols_result = sm.OLS(animal_means["viol_rate"], ols_x).fit()

    sklearn_model = LinearRegression()
    sklearn_x = animal_means[["hit_rate"]]
    sklearn_y = animal_means["viol_rate"]
    sklearn_model.fit(sklearn_x, sklearn_y)
    sklearn_pred = sklearn_model.predict(sklearn_x)
    sklearn_r2 = float(r2_score(sklearn_y, sklearn_pred))

    statistics: dict[str, Any] = {
        "mixedlm_summary": str(mixed_result.summary()),
        "conditional_r2": float(
            r2_score(session_percentages["viol_rate"], mixed_result.fittedvalues)
        ),
        "marginal_r2": float(r2_score(session_percentages["viol_rate"], pred_fixed)),
        "ols_summary": str(ols_result.summary()),
        "sklearn": {
            "r2": sklearn_r2,
            "slope": float(sklearn_model.coef_[0]),
            "intercept": float(sklearn_model.intercept_),
        },
        "n_sessions": int(len(session_percentages)),
        "n_animals": int(animal_means["animal_id"].nunique()),
    }
    return statistics, residuals


def animal_level_association(frame: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    """Pearson, Spearman, and OLS summaries for one value per animal."""
    pearson = pearsonr(frame[x], frame[y])
    spearman = spearmanr(frame[x], frame[y])
    ols = sm.OLS(frame[y], sm.add_constant(frame[[x]])).fit()
    return {
        "x": x,
        "y": y,
        "n_animals": int(frame["animal_id"].nunique()),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "slope": float(ols.params[x]),
        "intercept": float(ols.params["const"]),
        "r2": float(ols.rsquared),
    }


def per_animal_slope_test(wide: pd.DataFrame) -> dict[str, Any]:
    """Test per-animal linear slopes against zero across animals.

    ``wide`` has one row per animal and numeric condition columns; missing
    conditions are skipped when fitting each animal's slope.
    """
    slopes = wide.apply(
        lambda row: linregress(row.dropna().index, row.dropna()).slope, axis=1
    )
    test = ttest_1samp(slopes, 0)
    return {
        "n_animals": len(slopes),
        "mean_slope": float(slopes.mean()),
        "n_positive": int((slopes > 0).sum()),
        "t": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def post_violation_effect(trials: pd.DataFrame) -> dict[str, Any]:
    """Per-animal increase in P(violation) after a violation vs after a non-violation.

    The first trial of each session is excluded because it has no within-session
    predecessor, matching the previous-violation regressor in Figure 3.
    """
    ordered = trials.sort_values(["animal_id", "session", "trial"])
    positioned = session_position_trials(ordered)
    core = positioned.loc[positioned["position"] > 0]
    differences: list[float] = []
    for _, group in core.groupby("animal_id", sort=False):
        previous = shift_within_sessions(
            group["violation"].astype(int), group["session"]
        )
        after_violation = group.loc[previous == 1, "violation"].mean()
        after_non_violation = group.loc[previous == 0, "violation"].mean()
        if np.isnan(after_violation) or np.isnan(after_non_violation):
            continue
        differences.append(float((after_violation - after_non_violation) * 100))
    test = ttest_1samp(differences, 0)
    return {
        "n_animals": len(differences),
        "mean_difference_pp": float(np.mean(differences)),
        "n_positive": int(sum(value > 0 for value in differences)),
        "t": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def session_progress_statistics(progress_rates: pd.DataFrame) -> dict[str, Any]:
    """Per-animal slope of violation rate over session progress (pp per 10%)."""
    wide = progress_rates.pivot(
        index="animal_id", columns="session_progress", values="viol_rate"
    )
    return {
        "slope_per_10_percent": per_animal_slope_test(
            wide.rename(columns=lambda x: x / 10)
        ),
        "first_bin_mean": float(wide.iloc[:, 0].mean()),
        "last_bin_mean": float(wide.iloc[:, -1].mean()),
    }


def delay_violation_statistics(
    delay_rates: pd.DataFrame, *, reference: float = 2.0
) -> dict[str, Any]:
    """Across-animal tests of violation rate by delay length.

    Tests per-animal linear slopes against zero, and each delay against the
    reference delay with paired t-tests over animals that used both
    (Bonferroni-corrected across comparisons).
    """
    wide = delay_rates.pivot(
        index="animal_id", columns="delay_time", values="viol_rate"
    )
    comparisons = [delay for delay in wide.columns if delay != reference]
    versus_reference = {}
    for delay in comparisons:
        paired = wide[[reference, delay]].dropna()
        test = ttest_rel(paired[delay], paired[reference])
        versus_reference[f"{delay:g}"] = {
            "n_animals": len(paired),
            "mean_difference": float((paired[delay] - paired[reference]).mean()),
            "t": float(test.statistic),
            "p_value": float(test.pvalue),
            "p_value_bonferroni": min(float(test.pvalue) * len(comparisons), 1.0),
        }
    return {
        "slope": per_animal_slope_test(wide),
        "reference_delay": reference,
        "versus_reference": versus_reference,
    }


def relative_delay_statistics(
    relative_rates: pd.DataFrame, matched_rates: pd.DataFrame
) -> dict[str, Any]:
    """Paired tests of longer-than-previous versus same-or-shorter delays.

    ``unmatched`` compares animal rates directly; ``delay_matched`` averages
    the longer minus same-or-shorter difference within each current delay
    before testing it against zero across animals.
    """
    same, longer = RELATIVE_DELAY_ORDER
    wide = relative_rates.pivot(
        index="animal_id", columns="relative_delay", values="viol_rate"
    )
    unmatched = ttest_rel(wide[longer], wide[same])
    delays = relative_rates.pivot(
        index="animal_id", columns="relative_delay", values="mean_delay_time"
    ).mean()
    matched = matched_rates.pivot_table(
        index=["animal_id", "delay_time"],
        columns="relative_delay",
        values="viol_rate",
    ).dropna()
    differences = (matched[longer] - matched[same]).groupby("animal_id").mean()
    matched_test = ttest_1samp(differences, 0)
    return {
        "unmatched": {
            "n_animals": len(wide),
            "mean_same_or_shorter": float(wide[same].mean()),
            "mean_longer": float(wide[longer].mean()),
            "mean_delay_same_or_shorter": float(delays[same]),
            "mean_delay_longer": float(delays[longer]),
            "t": float(unmatched.statistic),
            "p_value": float(unmatched.pvalue),
        },
        "delay_matched": {
            "n_animals": len(differences),
            "mean_difference": float(differences.mean()),
            "t": float(matched_test.statistic),
            "p_value": float(matched_test.pvalue),
        },
    }
