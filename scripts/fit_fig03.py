"""Run all model fits used by Figure 3 in dependency order."""

import subprocess
import sys

ARTIFACTS = (
    "2025_07_19_fig3_fit_prev_violation_rs47",
    "2025_07_19_fig3_sweep_prev_viol_taus_rs47",
    "2025_07_19_fig3_fit_filt_prev_violation_rs47",
    "2025_07_19_fig3_compare_prev_viol_rs47",
)

for artifact in ARTIFACTS:
    print(f"fitting {artifact}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "violations_glm.cli", "fit", artifact, *sys.argv[1:]],
        check=True,
    )
