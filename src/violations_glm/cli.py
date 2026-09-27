"""Command-line entry points for publication fitting and validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from violations_glm.artifacts import compare_fit_tables, write_fit_artifact
from violations_glm.config import RANDOM_STATE
from violations_glm.fitting import best_fits, run_experiment, split_sessions
from violations_glm.io import load_trials_df
from violations_glm.paths import model_summaries_dir, processed_data_dir
from violations_glm.publication_specs import PUBLICATION_SPECS


def _taus(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return frame.set_index("animal_id")["prev_violation_tau"].to_dict()


def fit_command(args: argparse.Namespace) -> None:
    spec = PUBLICATION_SPECS[args.artifact]
    if args.sigma:
        spec = replace(spec, sigmas=tuple(args.sigma))
    trials = load_trials_df(args.data)
    if args.animal:
        trials = trials.loc[trials["animal_id"].isin(args.animal)].copy()
        missing = sorted(set(args.animal) - set(trials["animal_id"].unique()))
        if missing:
            raise ValueError(f"Requested animals not found: {missing}")

    fits, nulls = run_experiment(
        trials,
        spec,
        taus_by_animal=_taus(args.taus),
        random_state=RANDOM_STATE,
    )
    session_splits = {}
    for animal_id, animal_trials in trials.groupby("animal_id", sort=True):
        train_sessions, test_sessions = split_sessions(
            animal_trials, random_state=RANDOM_STATE
        )
        session_splits[animal_id] = {
            "train": train_sessions.tolist(),
            "test": test_sessions.tolist(),
        }
    destination = model_summaries_dir() / args.collection / spec.name
    write_fit_artifact(
        destination,
        fits=fits,
        nulls=nulls,
        spec=spec,
        data_path=args.data,
        random_state=RANDOM_STATE,
        session_splits=session_splits,
    )

    if args.validate_legacy:
        expected_path = (
            model_summaries_dir() / "legacy" / spec.name / "fit_models.parquet"
        )
        expected = pd.read_parquet(expected_path)
        binary_comparison = spec.name == "2025_07_22_fig5_compare_to_binary_rs47"
        rtol = 1e-6 if binary_comparison and args.rtol is None else args.rtol
        atol = 1e-2 if binary_comparison and args.atol is None else args.atol
        differences = compare_fit_tables(
            fits,
            expected,
            rtol=rtol if rtol is not None else 1e-7,
            atol=atol if atol is not None else 1e-5,
            ignore_weight_models=(
                frozenset({"base"}) if binary_comparison else frozenset()
            ),
            ignore_numeric_models=(
                frozenset({"base"}) if binary_comparison else frozenset()
            ),
            numeric_columns=("nll",) if binary_comparison else ("nll", "train_nll"),
        )
        print(json.dumps(differences, indent=2, sort_keys=True))

    if spec.taus:
        selected = best_fits(fits)[["animal_id", "tau"]].rename(
            columns={"tau": "prev_violation_tau"}
        )
        selected.to_csv(destination / "selected_taus.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="violations-glm")
    subparsers = parser.add_subparsers(required=True)
    fit = subparsers.add_parser("fit", help="Run one submitted fit experiment")
    fit.add_argument("artifact", choices=sorted(PUBLICATION_SPECS))
    fit.add_argument(
        "--data",
        type=Path,
        default=processed_data_dir() / "all_animals_trained_cleaned.parquet",
    )
    fit.add_argument(
        "--taus",
        type=Path,
        default=Path("configs/submitted_taus.csv"),
    )
    fit.add_argument("--animal", action="append")
    fit.add_argument("--sigma", action="append", type=float)
    fit.add_argument("--collection", default="current")
    fit.add_argument("--validate-legacy", action="store_true")
    fit.add_argument("--rtol", type=float)
    fit.add_argument("--atol", type=float)
    fit.set_defaults(handler=fit_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
