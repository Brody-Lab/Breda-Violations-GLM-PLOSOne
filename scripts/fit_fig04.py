"""Run both model fits used by Figure 4."""

import subprocess
import sys

ARTIFACTS = (
    "2025_07_22_fig4_fit_binary_prev_viol_stim_intrx_n_rs47",
    "2025_07_22_fig4_fit_filtered_prev_viol_stim_intrx_n_rs47",
)

for artifact in ARTIFACTS:
    print(f"fitting {artifact}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "violations_glm.cli", "fit", artifact, *sys.argv[1:]],
        check=True,
    )
