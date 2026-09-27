import json

import pandas as pd

from violations_glm.artifacts import compare_fit_tables


def test_comparison_can_target_figure_relevant_outputs() -> None:
    base = {
        "animal_id": "R001",
        "model_name": "base",
        "model_type": "binary",
        "sigma": 1.0,
        "features": "[]",
        "n_train_trials": 10,
        "n_test_trials": 5,
        "nll": 2.0,
        "train_nll": 3.0,
        "weights": json.dumps([1.0]),
    }
    history = {
        **base,
        "model_name": "prev_viol_filtered_itrx",
        "nll": 1.5,
        "train_nll": 2.5,
        "weights": json.dumps([0.5]),
    }
    expected = pd.DataFrame([base, history])
    observed = expected.copy()
    observed.loc[0, "nll"] = 100.0
    observed.loc[0, "train_nll"] = 100.0
    observed.loc[0, "weights"] = json.dumps([100.0])

    differences = compare_fit_tables(
        observed,
        expected,
        ignore_weight_models=frozenset({"base"}),
        ignore_numeric_models=frozenset({"base"}),
        numeric_columns=("nll",),
    )

    assert differences["nll"] == 0
    assert differences["ignored_weight_models"] == ["base"]
    assert differences["ignored_numeric_models"] == ["base"]


def test_animal_model_deltas_are_relative_to_the_base_model() -> None:
    from violations_glm.summaries import animal_model_deltas

    fits = pd.DataFrame(
        [
            {
                "animal_id": "W051",
                "model_name": "base",
                "model_type": "binary",
                "sigma": 1.0,
                "nll": 80.0,
                "n_test_trials": 10,
            },
            {
                "animal_id": "W051",
                "model_name": "prev_viol_filtered_itrx",
                "model_type": "binary",
                "sigma": 1.0,
                "nll": 70.0,
                "n_test_trials": 10,
            },
        ]
    )
    nulls = pd.DataFrame(
        [
            {
                "animal_id": "W051",
                "model_name": "null",
                "model_type": "binary",
                "nll": 100.0,
                "n_test_trials": 10,
            }
        ]
    )

    deltas = animal_model_deltas(fits, nulls, model_name="prev_viol_filtered_itrx")
    assert len(deltas) == 1
    assert deltas.loc[0, "bits_per_trial_relative_to_base"] > 0
