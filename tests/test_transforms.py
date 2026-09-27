import pandas as pd
import pytest

from violations_glm.transforms import (
    animal_session_edge_rates,
    animal_session_progress_violation_rates,
    animal_training_predictors,
    session_position_trials,
    trim_session_edges,
)


def _all_trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal_id": ["W051"] * 6 + ["W060"] * 4,
            "session": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2],
            "training_stage": [1, 1, 1, 1, 2, 2, 1, 1, 4, 4],
            "trial": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "hit": [1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            "violation": [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        }
    )


def _trained_animals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal_id": ["W051", "W060"],
            "hit_rate": [80.0, 70.0],
            "viol_rate": [15.0, 25.0],
            "n_trials": [300.0, 250.0],
        }
    )


def test_animal_training_predictors_use_fixation_stage_session_means() -> None:
    thresholds = pd.DataFrame(
        {"animal_id": ["W051", "W060"], "hit_threshold": [120.0, 240.0]}
    )
    predictors = animal_training_predictors(
        _all_trials(), thresholds, _trained_animals()
    )
    assert predictors["animal_id"].tolist() == ["W051", "W060"]
    assert predictors["fixation_stage_sessions"].tolist() == [2, 1]
    assert predictors["fixation_stage_viol_rate"].tolist() == [25.0, 100.0]
    assert predictors["sessions_to_trained"].tolist() == [120.0, 240.0]
    assert predictors["trained_viol_rate"].tolist() == [15.0, 25.0]
    assert predictors["trained_hit_rate"].tolist() == [80.0, 70.0]


def test_animal_training_predictors_require_every_trained_animal() -> None:
    thresholds = pd.DataFrame({"animal_id": ["W051"], "hit_threshold": [120.0]})
    with pytest.raises(ValueError, match="W060"):
        animal_training_predictors(_all_trials(), thresholds, _trained_animals())


def _edge_violation_session(n_trials: int = 30) -> pd.DataFrame:
    violation = [1] + [0] * (n_trials - 2) + [1]
    return pd.DataFrame(
        {
            "animal_id": "W051",
            "session": 1,
            "trial": range(1, n_trials + 1),
            "violation": violation,
            "hit": [None if value else 1.0 for value in violation],
        }
    ).iloc[::-1]


def test_session_position_trials_orders_by_trial() -> None:
    positioned = session_position_trials(_edge_violation_session())
    assert positioned["position"].tolist() == list(range(30))
    assert positioned["trials_from_end"].tolist() == list(range(29, -1, -1))
    assert positioned["session_progress"].iloc[[0, -1]].tolist() == [0.0, 1.0]


def test_session_progress_rates_exclude_edge_trials() -> None:
    rates = animal_session_progress_violation_rates(
        _edge_violation_session(), edge_trials=10
    )
    assert rates["n_trials"].sum() == 10
    assert (rates["viol_rate"] == 0).all()


def test_session_edge_rates_align_to_start_and_end() -> None:
    rates = animal_session_edge_rates(_edge_violation_session(), n_edge=2).set_index(
        ["aligned_to", "trial_offset"]
    )
    assert rates.loc[("Start", 1), "viol_rate"] == 100
    assert rates.loc[("Start", 2), "viol_rate"] == 0
    assert rates.loc[("End", -1), "viol_rate"] == 100
    assert rates.loc[("End", -2), "hit_rate"] == 100


def test_trim_session_edges_drops_first_and_last_trials() -> None:
    frame = _edge_violation_session(n_trials=30)
    trimmed = trim_session_edges(frame, n_edge=10)
    assert len(trimmed) == 10
    assert trimmed["trial"].min() == 11
    assert trimmed["trial"].max() == 20
