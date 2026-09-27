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
    "suppS1.ipynb",
    "suppS2.ipynb",
    "suppS3.ipynb",
    "suppS4.ipynb",
    "suppS5.ipynb",
)

OUTPUTS = {
    "fig01.ipynb": {
        "fig01a_task_timing.png",
        "fig01b_example_sequences.png",
        "fig01c_trials_per_session.png",
        "fig01d_hit_rate.png",
        "fig01e_choice_percentages.png",
        "fig01f_violation_rate.png",
    },
    "fig02.ipynb": {
        "fig02a_binary_weights.png",
        "fig02b_multinomial_weights.png",
    },
    "fig03.ipynb": {
        "fig03a_violation_autocorrelation.png",
        "fig03bc_history_weights.png",
        "fig03de_tau_summary.png",
        "fig03f_model_comparison.png",
    },
    "fig04.ipynb": {
        "fig04a_post_violation_performance.png",
        "fig04b_interaction_weights.png",
        "fig04c_simulated_psychometrics.png",
        "fig04d_stimulus_sensitivity.png",
        "fig04e_combined_lapse_rate.png",
    },
    "fig05.ipynb": {
        "fig05a_multinomial_comparison.png",
        "fig05b_binary_comparison.png",
        "fig05c_animal_deltas.png",
    },
    "suppS1.ipynb": {
        "suppS1ab_example_animals.png",
        "suppS1c_delta_sessions.png",
        "suppS1d_delta_trials.png",
        "suppS1e_delta_violations.png",
        "suppS1f_delta_stage4_violations.png",
    },
    "suppS2.ipynb": {
        "suppS2a_stimulus_pair_counts.png",
        "suppS2b_stimulus_pair_hit_rate.png",
        "suppS2c_stimulus_pair_violation_rate.png",
    },
    "suppS3.ipynb": {"suppS3_trained_thresholding.png"},
    "suppS4.ipynb": {
        "suppS4a_exponential_kernel.png",
        "suppS4b_filtered_sequence.png",
    },
    "suppS5.ipynb": set(),
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
