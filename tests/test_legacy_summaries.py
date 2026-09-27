import json

import pandas as pd

from violations_glm.paths import model_summaries_dir

EXPECTED_ROWS = {
    "2025_07_19_fig3_compare_prev_viol_rs47": 432,
    "2025_07_19_fig3_fit_filt_prev_violation_rs47": 144,
    "2025_07_19_fig3_fit_prev_violation_rs47": 144,
    "2025_07_19_fig3_sweep_prev_viol_taus_rs47": 2736,
    "2025_07_22_fig4_fit_binary_prev_viol_stim_intrx_n_rs47": 144,
    "2025_07_22_fig4_fit_filtered_prev_viol_stim_intrx_n_rs47": 144,
    "2025_07_22_fig5_compare_multi_intrxs_rs47": 720,
    "2025_07_22_fig5_compare_to_binary_rs47": 720,
}


def test_all_legacy_fit_summaries_are_stable_and_complete() -> None:
    root = model_summaries_dir() / "legacy"
    for artifact, expected_rows in EXPECTED_ROWS.items():
        directory = root / artifact
        metadata = json.loads((directory / "metadata.json").read_text())
        fits = pd.read_parquet(directory / "fit_models.parquet")
        assert metadata["source_commit"] == "8358b9f"
        assert metadata["fit_rows"] == expected_rows
        assert len(fits) == expected_rows
        assert not fits["features"].str.contains(r"0x[0-9a-f]+").any()
        if "compare" in artifact:
            nulls = pd.read_parquet(directory / "null_models.parquet")
            assert len(nulls) == 16
