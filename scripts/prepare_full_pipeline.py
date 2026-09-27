"""Run all-stage trials through training thresholds and session QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from violations_glm.config import MODEL_ANIMAL_IDS
from violations_glm.io import load_all_trials_df
from violations_glm.paths import processed_data_dir, results_dir
from violations_glm.preprocessing import (
    dataframe_digest,
    dataset_counts,
    file_sha256,
    prepare_modeling_dataset,
)
from violations_glm.thresholds import (
    apply_training_thresholds,
    rolling_performance,
    threshold_sessions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=processed_data_dir())
    args = parser.parse_args()

    all_trials = load_all_trials_df(args.data)
    all_trials_path = args.data or (
        processed_data_dir() / "all_animals_cleaned.parquet"
    )
    model_trials = all_trials.loc[all_trials["animal_id"].isin(MODEL_ANIMAL_IDS)].copy()
    rolling = rolling_performance(model_trials)
    thresholds = threshold_sessions(rolling)
    trained = apply_training_thresholds(model_trials, thresholds)
    modeling, stages = prepare_modeling_dataset(trained)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trained.to_parquet(
        args.output_dir / "all_animals_trained_threshold.parquet",
        index=False,
    )
    modeling_path = args.output_dir / "all_animals_trained_cleaned.parquet"
    manifest_path = modeling_path.with_suffix(".manifest.json")
    prior_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    )
    modeling.to_parquet(modeling_path, index=False)
    logical_sha256 = dataframe_digest(modeling)
    reference = prior_manifest.get("reference")
    if reference and logical_sha256 != reference["logical_sha256"]:
        raise ValueError("Full-pipeline data do not match the submitted reference")
    manifest = {
        "producer": "scripts/prepare_full_pipeline.py",
        "input": {
            "path": all_trials_path.name,
            "sha256": file_sha256(all_trials_path),
        },
        "stages": stages,
        "output": {
            "path": modeling_path.name,
            "sha256": file_sha256(modeling_path),
            "logical_sha256": logical_sha256,
            **dataset_counts(modeling).__dict__,
        },
        "filters": {
            "animals": "16-animal submitted modeling cohort",
            "minimum_trials_per_session": 50,
            "maximum_violation_rate": 0.99,
            "require_nonmissing_hit_and_violation_rates": True,
        },
    }
    if reference:
        manifest["reference"] = reference
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    preparation = results_dir() / "data_preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    rolling.to_parquet(preparation / "rolling_performance.parquet", index=False)
    thresholds.to_csv(preparation / "threshold_sessions.csv", index=False)
    report = {
        "all_trials": dataset_counts(all_trials).__dict__,
        "model_cohort_trials": dataset_counts(model_trials).__dict__,
        "trained_threshold": dataset_counts(trained).__dict__,
        "session_quality_stages": stages,
        "modeling": dataset_counts(modeling).__dict__,
    }
    (preparation / "counts.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
