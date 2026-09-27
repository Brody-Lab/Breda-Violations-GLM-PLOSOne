"""Psychometric simulations derived from fitted multinomial GLM weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from violations_glm.summaries import best_fits, unpack_weights

FEATURES = ("bias", "s_a", "sa_x_prev_viol", "s_b", "sb_x_prev_viol")


def right_minus_left_weights(fits: pd.DataFrame) -> pd.DataFrame:
    weights = unpack_weights(best_fits(fits))
    weights = weights.loc[
        weights["feature"].isin(FEATURES) & weights["weight_class"].isin(["L", "R"])
    ]
    pivot = weights.pivot_table(
        index=["animal_id", "feature"],
        columns="weight_class",
        values="weight",
    ).reset_index()
    pivot["weight"] = pivot["R"] - pivot["L"]
    return pivot[["animal_id", "feature", "weight"]]


def simulate_psychometrics(
    fits: pd.DataFrame,
    *,
    delta_range: tuple[float, float] = (-2, 2),
    num_points: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_weights = right_minus_left_weights(fits)
    delta = np.linspace(*delta_range, num_points)
    curves = []
    metrics = []
    for animal_id, animal in delta_weights.groupby("animal_id"):
        weights = animal.set_index("feature")["weight"]
        for previous_violation in (0, 1):
            slope_coefficient = 0.5 * (
                (weights["s_a"] - weights["s_b"])
                + previous_violation
                * (weights["sa_x_prev_viol"] - weights["sb_x_prev_viol"])
            )
            logits = slope_coefficient * delta + weights["bias"]
            probability = 1 / (1 + np.exp(-logits))
            curves.append(
                pd.DataFrame(
                    {
                        "animal_id": animal_id,
                        "previous_violation": previous_violation,
                        "delta": delta,
                        "p_right": probability,
                    }
                )
            )
            p_zero = 1 / (1 + np.exp(-weights["bias"]))
            metrics.append(
                {
                    "animal_id": animal_id,
                    "previous_violation": previous_violation,
                    "slope_at_zero": (slope_coefficient * p_zero * (1 - p_zero)),
                    "combined_lapse_rate": probability[0] + (1 - probability[-1]),
                }
            )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(metrics)
