"""Compute the corrected Figure 3 violation autocorrelation statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from violations_glm.config import RANDOM_STATE
from violations_glm.io import load_trials_df
from violations_glm.paths import processed_data_dir, results_dir
from violations_glm.preprocessing import file_sha256
from violations_glm.stats import (
    bootstrap_mean_ci,
    empirical_autocorrelation_pvalues,
    parallel_permutation_distribution,
    session_autocorrelations,
    violation_sessions,
)
from violations_glm.transforms import trim_session_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir() / "fig03_autocorrelation",
    )
    parser.add_argument(
        "--edge-trials",
        type=int,
        default=0,
        help="Drop this many trials from the start and end of each session.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_trials_df(args.data)
    source = args.data or (processed_data_dir() / "all_animals_trained_cleaned.parquet")
    default_output = results_dir() / "fig03_autocorrelation"
    output = args.output
    if args.edge_trials and output == default_output:
        output = results_dir() / "fig03_autocorrelation_edges_trimmed"
    n_trials_before = len(frame)
    if args.edge_trials:
        frame = trim_session_edges(frame, args.edge_trials)
    n_trials_excluded = n_trials_before - len(frame)
    sessions = violation_sessions(frame, max_lag=args.max_lag)
    observed_sessions = session_autocorrelations(sessions, max_lag=args.max_lag)
    observed = observed_sessions.mean(axis=0)

    rng = np.random.default_rng(RANDOM_STATE)
    ci_lower, ci_upper = bootstrap_mean_ci(
        observed_sessions,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
    )
    null = parallel_permutation_distribution(
        sessions,
        rng=rng,
        max_lag=args.max_lag,
        n_permutations=args.n_permutations,
        workers=args.workers,
    )
    null_lower, null_upper = np.percentile(null, [2.5, 97.5], axis=0)
    null_mean = null.mean(axis=0)

    lags = np.arange(-args.max_lag, args.max_lag + 1)
    summary = pd.DataFrame(
        {
            "lag": lags,
            "observed_mean": observed,
            "observed_ci_lower": ci_lower,
            "observed_ci_upper": ci_upper,
            "null_mean": null_mean,
            "null_ci_lower": null_lower,
            "null_ci_upper": null_upper,
        }
    )
    pvalues = empirical_autocorrelation_pvalues(observed, null, max_lag=args.max_lag)
    summary = summary.merge(pvalues, on="lag", how="left")

    output.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output / "summary.parquet", index=False)
    pd.DataFrame(null, columns=lags.astype(str)).to_parquet(
        output / "null_distribution.parquet", index=False
    )
    metadata = {
        "random_state": RANDOM_STATE,
        "max_lag": args.max_lag,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "workers": args.workers,
        "analysis_unit": "session",
        "n_sessions": len(sessions),
        "tested_lags": [1, args.max_lag],
        "bonferroni_comparisons": args.max_lag,
        "data_sha256": file_sha256(source.resolve()),
        "edge_trials": args.edge_trials,
        "n_trials_excluded": n_trials_excluded,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
