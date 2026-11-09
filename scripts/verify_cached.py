"""Execute every publication notebook in memory and verify panel outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from violations_glm.paths import REPO_ROOT, figures_dir

NOTEBOOKS = (
    "fig01.ipynb",
    "fig02.ipynb",
    "fig03.ipynb",
    "fig04.ipynb",
    "fig05.ipynb",
    "fig06.ipynb",
    "suppS1.ipynb",
    "suppS2.ipynb",
    "suppS3.ipynb",
    "suppS4.ipynb",
    "suppS5.ipynb",
    "suppS6.ipynb",
)

OUTPUTS = {
    "fig01.ipynb": {
        "fig01a_task_timing.png",
        "fig01b_example_sequences.png",
        "fig01c_trials_per_session.png",
        "fig01d_session_rates.png",
        "fig01e_choice_percentages.png",
        "fig01f_violation_by_hit_rate.png",
    },
    "fig02.ipynb": {
        "fig02a_binary_weights.png",
        "fig02b_multinomial_weights.png",
    },
    "fig03.ipynb": {
        "fig03a_violation_autocorrelation.png",
        "fig03b_prev_violation_intuition.png",
        "fig03b_prev_violation_weights.png",
        "fig03c_filtered_intuition.png",
        "fig03c_filtered_weights.png",
        "fig03d_tau_heatmap.png",
        "fig03e_example_tau.png",
        "fig03f_model_comparison.png",
    },
    "fig04.ipynb": {
        "fig04a_post_violation_performance.png",
        "fig04b_prev_violation_intuition.png",
        "fig04b_stimulus_a_intuition.png",
        "fig04b_interaction_intuition.png",
        "fig04b_binary_interaction_weights.png",
        "fig04b_filtered_interaction_weights.png",
        "fig04c_simulated_psychometrics.png",
        "fig04d_stimulus_sensitivity.png",
        "fig04e_combined_lapse_rate.png",
    },
    "fig05.ipynb": {
        "fig05a_multinomial_comparison.png",
        "fig05b_binary_comparison.png",
    },
    "fig06.ipynb": {
        "fig06a_animal_improvement.png",
        "fig06b_drop_vs_improvement.png",
        "fig06c_violation_rate_vs_drop.png",
        "fig06d_within_animal_split.png",
    },
    "suppS1.ipynb": {
        "suppS1a_example_animals.png",
        "suppS1c_hit_alignment.png",
        "suppS1d_trial_alignment.png",
        "suppS1e_delta_sessions.png",
        "suppS1f_delta_trials.png",
        "suppS1g_delta_violations.png",
    },
    "suppS2.ipynb": {
        "suppS2_stimulus_pair_counts.png",
    },
    "suppS3.ipynb": {"suppS3_trained_thresholding.png"},
    "suppS4.ipynb": {
        "suppS4a_exponential_kernel.png",
        "suppS4b_filtered_sequence.png",
    },
    "suppS5.ipynb": {
        "suppS5a_fixation_stage_violations.png",
        "suppS5b_time_in_fixation_stage.png",
        "suppS5c_time_to_trained.png",
    },
    "suppS6.ipynb": {
        "suppS6a_session_progress.png",
        "suppS6b_session_start.png",
        "suppS6c_session_end.png",
        "suppS6d_autocorrelation_robustness.png",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", action="append", choices=NOTEBOOKS)
    args = parser.parse_args()
    selected = tuple(args.notebook) if args.notebook else NOTEBOOKS

    notebook_dir = REPO_ROOT / "notebooks"
    previous = Path.cwd()
    os.chdir(notebook_dir)
    try:
        for name in selected:
            path = notebook_dir / name
            notebook = nbformat.read(path, as_version=4)
            NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(notebook_dir)}},
            ).execute()
            print(f"executed {name}")
    finally:
        os.chdir(previous)

    expected = set().union(*(OUTPUTS[name] for name in selected))
    missing = sorted(
        filename for filename in expected if not (figures_dir() / filename).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Missing expected figure panels: {missing}")


if __name__ == "__main__":
    main()
