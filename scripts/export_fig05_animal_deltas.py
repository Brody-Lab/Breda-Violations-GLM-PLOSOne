"""Export the Figure 5 animal-level binary-model comparison used in Revision 1."""

from __future__ import annotations

import json

from violations_glm.io import load_model_summary
from violations_glm.paths import results_dir
from violations_glm.summaries import animal_model_deltas

ARTIFACT = "2025_07_22_fig5_compare_to_binary_rs47"
MODEL_NAME = "prev_viol_filtered_itrx"


def main() -> None:
    deltas = animal_model_deltas(
        load_model_summary(ARTIFACT),
        load_model_summary(ARTIFACT, kind="null_models"),
        model_name=MODEL_NAME,
    )
    values = deltas["bits_per_trial_relative_to_base"]
    destination = results_dir() / "fig05_animal_deltas"
    destination.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(destination / "binary_filtered_interaction.csv", index=False)
    summary = {
        "artifact": ARTIFACT,
        "model_name": MODEL_NAME,
        "n_animals": int(len(deltas)),
        "n_improved": int((values > 0).sum()),
        "mean_delta": float(values.mean()),
        "min_delta": float(values.min()),
        "max_delta": float(values.max()),
        "min_animal": str(deltas.loc[values.idxmin(), "animal_id"]),
        "max_animal": str(deltas.loc[values.idxmax(), "animal_id"]),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
