from violations_glm.config import ALL_ANIMAL_IDS, MODEL_ANIMAL_IDS
from violations_glm.io import (
    load_alignment_deltas,
    load_all_trials_df,
    load_history_robustness,
    load_rolling_performance,
    load_session_edge_rates,
    load_session_position_statistics,
    load_session_progress_violation_rates,
    load_threshold_sessions,
    load_training_predictor_statistics,
    load_training_predictors,
    load_training_summary,
    load_trials_df,
)


def test_modeling_and_reconstruction_loaders_use_distinct_cohorts() -> None:
    modeling = load_trials_df()
    reconstruction = load_all_trials_df()
    assert set(modeling["animal_id"]) == set(MODEL_ANIMAL_IDS)
    assert set(reconstruction["animal_id"]) == set(ALL_ANIMAL_IDS)


def test_named_result_loaders_return_publication_tables() -> None:
    deltas = load_alignment_deltas()
    assert set(deltas["animal_id"]) == set(ALL_ANIMAL_IDS)
    assert set(load_training_summary()["animal_id"]) == set(MODEL_ANIMAL_IDS)
    assert set(load_threshold_sessions()["animal_id"]) == set(MODEL_ANIMAL_IDS)
    assert set(load_training_predictors()["animal_id"]) == set(MODEL_ANIMAL_IDS)
    assert all(
        entry["n_animals"] == len(MODEL_ANIMAL_IDS)
        for entry in load_training_predictor_statistics().values()
    )
    assert set(load_session_progress_violation_rates()["animal_id"]) == set(
        MODEL_ANIMAL_IDS
    )
    assert set(load_session_edge_rates()["aligned_to"]) == {"Start", "End"}
    position_stats = load_session_position_statistics()
    assert position_stats["trimmed"]["slope_per_10_percent"]["n_animals"] == len(
        MODEL_ANIMAL_IDS
    )
    assert position_stats["all_trials"]["slope_per_10_percent"]["n_animals"] == len(
        MODEL_ANIMAL_IDS
    )
    assert set(
        load_session_progress_violation_rates(include_edges=True)["animal_id"]
    ) == set(MODEL_ANIMAL_IDS)
    assert load_history_robustness()["trimmed"]["n_animals"] == len(MODEL_ANIMAL_IDS)
    assert {"rolling_mean_hit", "rolling_mean_violation"} <= set(
        load_rolling_performance().columns
    )
