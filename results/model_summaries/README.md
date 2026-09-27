# Model summaries

Stable Parquet and JSON outputs in this directory are the plotting and
regression-test interface for publication model results.

`legacy/` is generated from the eight trusted submitted fit objects with:

```bash
uv run python scripts/export_legacy_baselines.py \
  --legacy-repo /path/to/violations-multinomial
```

The source files are checksum-verified against
`configs/legacy_fit_manifest.yml`. Figure notebooks must not deserialize the
legacy pickle files directly.

Notebooks use `legacy` by default. Set
`VIOLATIONS_GLM_MODEL_COLLECTION=current` to plot a newly generated full
refit; `scripts/reproduce_all.py` sets this automatically in full mode.

The Figure 5 binary-base fit preserves the submitted optimizer, including its
historical regularization-sign inconsistency. That base model is not a
figure-5 history comparison; its selected sigma can move and its unused
weights are unstable. The regression check therefore validates held-out NLL
and weights for the previous-violation models only (absolute tolerance
`0.01`). All other fits validate held-out/training likelihoods and weights.
