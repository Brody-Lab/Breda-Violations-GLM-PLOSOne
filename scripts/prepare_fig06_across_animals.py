"""Figure 6 across-animal summaries: why does the Fig 5 binary-comparison gain vary?

Builds one per-animal table of candidate explanations (model decomposition,
model-free post-violation effects, animal-level violation measures) and
correlates each with the trained violation rate and with the binary-comparison
gain (``binary_prev_viol_filtered_itrx``). Violation timing within the trial is
not recorded, so nothing here separates violation subtypes. Writes
``animal_table.csv`` and ``correlations.json`` to ``results/fig06/across_animals/``.
"""

from __future__ import annotations

import json

import pandas as pd
from scipy import stats

from violations_glm.features import previous_violation
from violations_glm.io import (
    load_delay_violation_rates,
    load_hit_violation_animals,
    load_model_summary,
    load_training_predictors,
    load_trials_df,
)
from violations_glm.paths import results_dir
from violations_glm.psychometrics import simulate_psychometrics
from violations_glm.stats import partial_correlation
from violations_glm.summaries import best_fits, bits_per_trial

MULTI = "2025_07_22_fig5_compare_multi_intrxs_rs47"
BINARY = "2025_07_22_fig5_compare_to_binary_rs47"
FIG04_BINARY_ITRX = "2025_07_22_fig4_fit_binary_prev_viol_stim_intrx_n_rs47"
TAU_SWEEP = "2025_07_19_fig3_sweep_prev_viol_taus_rs47"
OUTLIERS = ("W065", "W083")


def model_deltas(artifact: str, prefix: str) -> pd.DataFrame:
    comparison = bits_per_trial(
        load_model_summary(artifact), load_model_summary(artifact, kind="null_models")
    )
    wide = comparison.pivot(
        index="animal_id", columns="model_name", values="bits_per_trial_relative_to_base"
    ).drop(columns=["base", "null"])
    wide["itrx_gain"] = wide["prev_viol_filtered_itrx"] - wide["prev_viol_filtered"]
    return wide.add_prefix(f"{prefix}_")


def behavioral_measures(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()
    trials["prev_violation"] = previous_violation(trials)
    completed = trials.loc[trials["violation"] == 0]

    session_hits = completed.groupby(
        ["animal_id", "session", "prev_violation"], as_index=False
    )["hit"].mean()
    hits = session_hits.pivot_table(
        index="animal_id", columns="prev_violation", values="hit"
    )
    post_violation_hit_drop = (hits[0] - hits[1]) * 100

    clustering = trials.groupby(["animal_id", "prev_violation"])["violation"].mean().unstack()
    violation_lift = clustering[1] / clustering[0]

    session_rates = trials.groupby(["animal_id", "session"])["violation"].mean()
    session_viol_sd = session_rates.groupby("animal_id").std() * 100

    return pd.DataFrame(
        {
            "post_violation_hit_drop_pp": post_violation_hit_drop,
            "violation_lift_after_violation": violation_lift,
            "session_viol_rate_sd_pp": session_viol_sd,
        }
    )


def psychometric_measures() -> pd.DataFrame:
    _, metrics = simulate_psychometrics(load_model_summary(FIG04_BINARY_ITRX))
    wide = metrics.pivot(
        index="animal_id",
        columns="previous_violation",
        values=["slope_at_zero", "combined_lapse_rate"],
    )
    return pd.DataFrame(
        {
            "baseline_slope": wide[("slope_at_zero", 0)],
            "post_violation_slope_drop": wide[("slope_at_zero", 0)] - wide[("slope_at_zero", 1)],
            "post_violation_lapse_increase": wide[("combined_lapse_rate", 1)]
            - wide[("combined_lapse_rate", 0)],
        }
    )


def delay_slopes() -> pd.Series:
    rates = load_delay_violation_rates()
    return rates.groupby("animal_id").apply(
        lambda animal: stats.linregress(animal["delay_time"], animal["viol_rate"]).slope,
        include_groups=False,
    ).rename("viol_rate_per_s_delay")


def selected_taus() -> pd.Series:
    sweep = best_fits(load_model_summary(TAU_SWEEP))
    if "tau" in sweep:
        return sweep.set_index("animal_id")["tau"].rename("tau")
    taus = pd.read_csv(results_dir() / "model_summaries" / "current" / TAU_SWEEP / "selected_taus.csv")
    return taus.set_index("animal_id")["prev_violation_tau"].rename("tau")


def correlate(x: pd.Series, y: pd.Series) -> dict[str, float]:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    keep = ~x.index.isin(OUTLIERS)
    without = stats.pearsonr(x[keep], y[keep])
    loo = [stats.pearsonr(x.drop(animal), y.drop(animal))[0] for animal in x.index]
    return {
        "n": int(mask.sum()),
        "pearson_r": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman_rho": float(spearman[0]),
        "spearman_p": float(spearman[1]),
        "pearson_r_without_W065_W083": float(without[0]),
        "pearson_p_without_W065_W083": float(without[1]),
        "loo_r_min": float(min(loo)),
        "loo_r_max": float(max(loo)),
    }


def partial(frame: pd.DataFrame, x: str, y: str, covariate: str) -> dict[str, float | str]:
    return {
        **partial_correlation(frame[x], frame[y], frame[covariate]),
        "covariate": covariate,
    }


def main() -> None:
    animals = load_hit_violation_animals().set_index("animal_id")[["viol_rate", "hit_rate"]]
    predictors = load_training_predictors().set_index("animal_id")[
        ["fixation_stage_viol_rate", "sessions_to_trained"]
    ]
    table = (
        animals.join(model_deltas(BINARY, "binary"))
        .join(model_deltas(MULTI, "multi"))
        .join(behavioral_measures(load_trials_df()))
        .join(psychometric_measures())
        .join(delay_slopes())
        .join(selected_taus())
        .join(predictors)
    )

    target = "binary_prev_viol_filtered_itrx"
    against_viol_rate = {
        column: correlate(table["viol_rate"], table[column])
        for column in table.columns
        if column != "viol_rate"
    }
    against_delta = {
        column: correlate(table[column], table[target])
        for column in table.columns
        if column != target
    }
    partials = {
        "delta_vs_viol_rate_given_hit_rate": partial(
            table, "viol_rate", target, "hit_rate"
        ),
        "delta_vs_viol_rate_given_baseline_slope": partial(
            table, "viol_rate", target, "baseline_slope"
        ),
        "delta_vs_hit_rate_given_viol_rate": partial(
            table, "hit_rate", target, "viol_rate"
        ),
        "delta_vs_viol_rate_given_post_violation_hit_drop": partial(
            table, "viol_rate", target, "post_violation_hit_drop_pp"
        ),
        "delta_vs_post_violation_hit_drop_given_viol_rate": partial(
            table, "post_violation_hit_drop_pp", target, "viol_rate"
        ),
    }

    destination = results_dir() / "fig06" / "across_animals"
    destination.mkdir(parents=True, exist_ok=True)
    table.sort_values("viol_rate").to_csv(destination / "animal_table.csv")
    (destination / "correlations.json").write_text(
        json.dumps(
            {
                "against_viol_rate": against_viol_rate,
                "against_binary_filtered_itrx_delta": against_delta,
                "partial": partials,
            },
            indent=2,
        )
        + "\n"
    )

    def show(title: str, results: dict[str, dict[str, float]]) -> None:
        print(f"\n## {title}")
        print(
            pd.DataFrame(results)
            .T[["pearson_r", "pearson_p", "spearman_rho", "pearson_r_without_W065_W083", "loo_r_min", "loo_r_max"]]
            .round(3)
            .to_string()
        )

    pd.set_option("display.width", 200)
    print(table.sort_values("viol_rate").round(4).to_string())
    show("Correlation with trained violation rate", against_viol_rate)
    show("Correlation with binary filtered+itrx delta", against_delta)
    print("\n## Partial correlations")
    print(pd.DataFrame(partials).T.to_string())


if __name__ == "__main__":
    main()
