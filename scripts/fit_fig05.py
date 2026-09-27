"""Run both model comparisons used by Figure 5."""

import subprocess
import sys

ARTIFACTS = (
    "2025_07_22_fig5_compare_multi_intrxs_rs47",
    "2025_07_22_fig5_compare_to_binary_rs47",
)

for artifact in ARTIFACTS:
    print(f"fitting {artifact}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "violations_glm.cli", "fit", artifact, *sys.argv[1:]],
        check=True,
    )
