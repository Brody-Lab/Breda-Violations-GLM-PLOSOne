"""Figure 6 within-animal summaries: do violations disrupt more when they are rare?

Two analyses on the trained-session trial table (post hoc, hypothesis-generating):

1. Within-animal session split: each animal's sessions are split at its median
   session violation rate, and the post-violation accuracy drop is compared
   between the low- and high-violation halves (paired across animals).
2. Repeat-violation control: violation trials are excluded from the accuracy
   calculation, so a disengaged post-violation trial that becomes another
   violation drops out of the drop. Relates the drop to the probability of
   violating again right after a violation, and partials it out of the
   violation rate vs drop correlation.

Writes ``session_split.csv`` and ``statistics.json`` to
``results/fig06/within_animal/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from violations_glm.features import previous_violation
from violations_glm.io import load_hit_violation_animals, load_trials_df
from violations_glm.paths import results_dir
from violations_glm.stats import within_subject_test

HALF_LABELS = {0: "Low-violation sessions", 1: "High-violation sessions"}


def prepare_trials() -> pd.DataFrame:
    trials = load_trials_df()[
        ["animal_id", "session", "violation", "hit", "choice"]
    ].copy()
    trials["prev_violation"] = previous_violation(trials)
    session_rate = trials.groupby(["animal_id", "session"])["violation"].transform("mean")
    trials["session_viol_rate"] = session_rate * 100
    return trials


def assign_split(trials: pd.DataFrame, *, quantile: float | None) -> pd.Series:
    """0 = low, 1 = high, NaN = excluded. ``quantile=None`` is the median split."""
    sessions = (
        trials.groupby(["animal_id", "session"], as_index=False)["session_viol_rate"]
        .first()
        .set_index(["animal_id", "session"])["session_viol_rate"]
    )
    if quantile is None:
        median = sessions.groupby("animal_id").transform("median")
        label = (sessions > median).astype(float)  # ties go to the low half
    else:
        low = sessions.groupby("animal_id").transform(lambda s: s.quantile(quantile))
        high = sessions.groupby("animal_id").transform(lambda s: s.quantile(1 - quantile))
        label = pd.Series(np.nan, index=sessions.index)
        label[sessions <= low] = 0.0
        label[sessions >= high] = 1.0
    keys = pd.MultiIndex.from_frame(trials[["animal_id", "session"]])
    return pd.Series(label.reindex(keys).to_numpy(), index=trials.index)


def pooled_drop(trials: pd.DataFrame) -> pd.DataFrame:
    """Per animal and half: hit rate after non-violation/violation, drop in pp."""
    completed = trials.loc[(trials["violation"] == 0) & trials["half"].notna()]
    grouped = completed.groupby(["animal_id", "half", "prev_violation"])["hit"]
    rates = grouped.mean().unstack("prev_violation")
    counts = grouped.size().unstack("prev_violation")
    session_rate = (
        trials.loc[trials["half"].notna()]
        .groupby(["animal_id", "half"])["violation"]
        .mean()
        * 100
    )
    out = pd.DataFrame(
        {
            "mean_viol_rate": session_rate,
            "baseline_hit_rate": rates[0] * 100,
            "post_violation_hit_rate": rates[1] * 100,
            "n_post_violation_trials": counts[1],
            "pooled_drop_pp": (rates[0] - rates[1]) * 100,
        }
    )
    return out.reset_index()


def half_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Per animal and half: Fig 4A-style ``drop_pp`` plus the pooled-trial version."""
    return pooled_drop(trials).merge(
        session_averaged_drop(trials), on=["animal_id", "half"], how="left"
    )


def session_averaged_drop(trials: pd.DataFrame) -> pd.DataFrame:
    """Fig 4A-style drop: per-session hit rates first, then averaged per animal and half."""
    completed = trials.loc[(trials["violation"] == 0) & trials["half"].notna()]
    session_hits = completed.groupby(
        ["animal_id", "half", "session", "prev_violation"], as_index=False
    )["hit"].mean()
    wide = session_hits.pivot_table(
        index=["animal_id", "half", "session"], columns="prev_violation", values="hit"
    ).dropna()
    drops = ((wide[0] - wide[1]) * 100).groupby(["animal_id", "half"]).mean()
    return drops.rename("drop_pp").reset_index()


def paired_test(
    drops: pd.DataFrame, label: str, *, value_col: str = "drop_pp"
) -> dict[str, float | str]:
    print(f"\n### {label}")
    result = within_subject_test(
        drops,
        subject_col="animal_id",
        condition_col="half",
        value_col=value_col,
        condition_labels=HALF_LABELS,
    )
    wide = drops.pivot(index="animal_id", columns="half", values=value_col)
    difference = wide[0] - wide[1]  # positive = larger drop in low-violation sessions
    result.update(
        {
            "n_animals": int(len(wide)),
            "mean_drop_low_pp": float(wide[0].mean()),
            "mean_drop_high_pp": float(wide[1].mean()),
            "mean_low_minus_high_pp": float(difference.mean()),
            "n_animals_larger_drop_in_low": int((difference > 0).sum()),
        }
    )
    return result


def repeat_violation_probability(trials: pd.DataFrame) -> pd.Series:
    after = trials.loc[trials["prev_violation"] == 1]
    return (after.groupby("animal_id")["violation"].mean() * 100).rename(
        "p_violate_after_violation"
    )


def overall_drop(trials: pd.DataFrame) -> pd.Series:
    """Fig 4A-style drop per animal: session hit rates first, then averaged."""
    completed = trials.loc[trials["violation"] == 0]
    session_hits = completed.groupby(
        ["animal_id", "session", "prev_violation"], as_index=False
    )["hit"].mean()
    wide = session_hits.pivot_table(
        index=["animal_id", "session"], columns="prev_violation", values="hit"
    ).dropna()
    return ((wide[0] - wide[1]) * 100).groupby("animal_id").mean().rename("drop_pp")


def residual(y: pd.Series, covariate: pd.Series) -> pd.Series:
    fit = stats.linregress(covariate, y)
    return y - (fit.intercept + fit.slope * covariate)


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman_rho": float(spearman[0]),
        "spearman_p": float(spearman[1]),
    }


def partial_correlation(x: pd.Series, y: pd.Series, covariate: pd.Series) -> dict[str, float]:
    r, p = stats.pearsonr(residual(x, covariate), residual(y, covariate))
    return {"partial_r": float(r), "partial_p": float(p)}


def main() -> None:
    trials = prepare_trials()
    animals = load_hit_violation_animals().set_index("animal_id")["viol_rate"]

    # Analysis 1: within-animal session split.
    trials["half"] = assign_split(trials, quantile=None)
    split = half_table(trials)
    results: dict[str, object] = {
        "median_split": paired_test(
            split, "Median split, session-averaged drop (main)"
        ),
        "median_split_pooled": paired_test(
            split, "Median split, pooled trials", value_col="pooled_drop_pp"
        ),
    }
    halves = split.pivot(index="animal_id", columns="half")
    results["split_descriptives"] = {
        "mean_viol_rate_low_pct": float(halves[("mean_viol_rate", 0.0)].mean()),
        "mean_viol_rate_high_pct": float(halves[("mean_viol_rate", 1.0)].mean()),
        "mean_baseline_hit_low_pct": float(halves[("baseline_hit_rate", 0.0)].mean()),
        "mean_baseline_hit_high_pct": float(halves[("baseline_hit_rate", 1.0)].mean()),
        "min_post_violation_trials": int(split["n_post_violation_trials"].min()),
    }
    trials["half"] = assign_split(trials, quantile=1 / 3)
    tertiles = half_table(trials)
    results["tertile_split"] = paired_test(
        tertiles, "Bottom vs top third of sessions, session-averaged drop"
    )
    results["tertile_split_pooled"] = paired_test(
        tertiles,
        "Bottom vs top third of sessions, pooled trials",
        value_col="pooled_drop_pp",
    )

    # Analysis 2: repeat-violation control.
    repeat = repeat_violation_probability(trials)
    drop = overall_drop(trials)
    table = pd.concat([animals.rename("viol_rate"), drop, repeat], axis=1)
    results["violation_rate_vs_drop"] = correlation(table["viol_rate"], table["drop_pp"])
    results["violation_rate_vs_repeat_probability"] = correlation(
        table["viol_rate"], table["p_violate_after_violation"]
    )
    results["drop_vs_repeat_probability"] = correlation(
        table["drop_pp"], table["p_violate_after_violation"]
    )
    results["partial_violation_rate_vs_drop_given_repeat_probability"] = partial_correlation(
        table["viol_rate"], table["drop_pp"], table["p_violate_after_violation"]
    )

    destination = results_dir() / "fig06" / "within_animal"
    destination.mkdir(parents=True, exist_ok=True)
    split.to_csv(destination / "session_split.csv", index=False)
    (destination / "statistics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print("\n" + json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
