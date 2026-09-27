import numpy as np
import pandas as pd

from violations_glm.stats import (
    animal_level_association,
    autocorrelation,
    empirical_autocorrelation_pvalues,
    hit_violation_regressions,
    per_animal_slope_test,
    permutation_distribution,
    post_violation_effect,
    session_progress_statistics,
)


def test_autocorrelation_is_symmetric_and_normalized() -> None:
    values = np.array([0, 1, 0, 1, 1])
    observed = autocorrelation(values, max_lag=2)
    centered = values - values.mean()
    full = np.correlate(centered, centered, mode="full")
    legacy = full[2:7] / full[4]
    np.testing.assert_allclose(observed, legacy)
    np.testing.assert_allclose(observed, observed[::-1])
    assert observed[2] == 1


def test_empirical_pvalues_use_plus_one_and_unique_lags() -> None:
    max_lag = 2
    observed = np.array([0.2, 0.5, 1.0, 0.5, 0.2])
    null = np.zeros((9, 5))
    result = empirical_autocorrelation_pvalues(observed, null, max_lag=max_lag)
    assert result["lag"].tolist() == [-2, -1, 1, 2]
    assert set(result["p_value"]) == {0.1}
    assert set(result["p_value_bonferroni"]) == {0.2}


def test_batched_permutation_distribution_is_symmetric_and_reproducible() -> None:
    sessions = [
        np.array([0, 1, 0, 1, 1]),
        np.array([1, 0, 0, 1, 0, 1]),
    ]
    first = permutation_distribution(
        sessions,
        rng=np.random.default_rng(47),
        max_lag=2,
        n_permutations=65,
    )
    second = permutation_distribution(
        sessions,
        rng=np.random.default_rng(47),
        max_lag=2,
        n_permutations=65,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, first[:, ::-1])
    np.testing.assert_array_equal(first[:, 2], np.ones(65))


def test_hit_violation_regressions_return_summaries_and_residuals() -> None:
    sessions = pd.DataFrame(
        {
            "animal_id": (["W051"] * 3 + ["W060"] * 3 + ["W065"] * 3 + ["W066"] * 3),
            "session": [1, 2, 3] * 4,
            "hit_rate": [
                80.0,
                82.0,
                85.0,
                70.0,
                73.0,
                75.0,
                90.0,
                88.0,
                92.0,
                78.0,
                79.0,
                81.0,
            ],
            "viol_rate": [
                22.0,
                19.0,
                17.0,
                41.0,
                37.0,
                35.0,
                12.0,
                14.0,
                10.0,
                25.0,
                24.0,
                22.0,
            ],
            "n_trials": [200] * 12,
        }
    )
    animals = sessions.groupby("animal_id", as_index=False).agg(
        hit_rate=("hit_rate", "mean"),
        viol_rate=("viol_rate", "mean"),
        n_trials=("n_trials", "mean"),
    )
    statistics, residuals = hit_violation_regressions(sessions, animals)
    assert "mixedlm_summary" in statistics
    assert "ols_summary" in statistics
    assert set(statistics["sklearn"]) == {"r2", "slope", "intercept"}
    assert statistics["n_animals"] == 4
    assert len(residuals) == len(sessions)
    assert {"fitted", "residual"} <= set(residuals.columns)


def test_animal_level_association_recovers_linear_fit() -> None:
    animals = pd.DataFrame(
        {
            "animal_id": ["W051", "W060", "W065", "W066", "W068"],
            "predictor": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    animals["outcome"] = 10 - 2 * animals["predictor"]
    statistics = animal_level_association(animals, "predictor", "outcome")
    assert statistics["n_animals"] == 5
    np.testing.assert_allclose(
        [statistics["slope"], statistics["intercept"], statistics["r2"]],
        [-2.0, 10.0, 1.0],
    )
    np.testing.assert_allclose(
        [statistics["pearson_r"], statistics["spearman_r"]], [-1.0, -1.0]
    )


def test_per_animal_slope_test_skips_missing_conditions() -> None:
    wide = pd.DataFrame(
        {2.0: [10.0, 20.0, 5.0], 4.0: [12.0, np.nan, 9.0], 6.0: [14.0, 26.0, 13.0]},
        index=["W051", "W060", "W065"],
    )
    result = per_animal_slope_test(wide)
    assert result["n_animals"] == 3
    assert result["n_positive"] == 3
    np.testing.assert_allclose(result["mean_slope"], (1.0 + 1.5 + 2.0) / 3)


def test_post_violation_effect_matches_hand_computed_difference() -> None:
    rows = {
        "animal_id": ["W051"] * 6 + ["W060"] * 6,
        "session": [1] * 12,
        "trial": [1, 2, 3, 4, 5, 6] * 2,
        "violation": [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    }
    trials = pd.DataFrame(rows)
    result = post_violation_effect(trials)
    assert result["n_animals"] == 2
    np.testing.assert_allclose(result["mean_difference_pp"], -100 / 6)


def test_session_progress_statistics_report_slope_per_ten_percent() -> None:
    progress = pd.DataFrame(
        {
            "animal_id": ["W051"] * 3 + ["W060"] * 3,
            "session_progress": [5.0, 15.0, 25.0] * 2,
            "viol_rate": [10.0, 11.0, 12.0, 20.0, 22.0, 24.0],
        }
    )
    statistics = session_progress_statistics(progress)
    np.testing.assert_allclose(statistics["slope_per_10_percent"]["mean_slope"], 1.5)
    assert statistics["first_bin_mean"] == 15.0
    assert statistics["last_bin_mean"] == 18.0
