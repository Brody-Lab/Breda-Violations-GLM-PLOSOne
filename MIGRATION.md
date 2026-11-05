# Publication migration

This checklist records the reproducibility state of each deliverable. A figure
is complete only when its inputs are documented, its notebook runs top to
bottom, and its panel values have been compared with the submitted composite.

## Foundation

- [x] Establish the uv/src-layout project structure.
- [x] Define repository-relative data, results, and figure paths.
- [x] Lock and verify the Python environment.
- [x] Build the publication-data manifest with checksums and filtering counts.
- [x] Export stable golden summaries from all eight submitted model-fit objects.
- [x] Add fast tests and the full model-regression command.

## Main figures

Fidelity: **complete** = plots and printed stats match the publication source;
**skeleton** = notebook runs but has not been checked against source. See
[MIGRATION_FIDELITY.md](MIGRATION_FIDELITY.md).

| Figure | Fidelity | Notes |
|--------|----------|--------|
| 1 | partial | Panels B–F; Panel A timing in Designer. Hit–violation in S5. |
| 2 | complete | Schematics (no historical `@` composite notebook). |
| 3 | complete | Submitted styling; autocorrelation via script; GLM from summaries. |
| 4 | skeleton | Needs parity pass vs `@fig4_viol_stim_intrxs_multi.ipynb`. |
| 5 | skeleton | Needs parity pass vs `@fig5_model_comparisons.ipynb`. |

## Supporting figures

| Supplement | Fidelity | Notes |
|------------|----------|--------|
| S1 | complete | vs `figS1_new_dataset_alignment.ipynb` + W089 mega-fig C/D; loaders + `prepare_alignment_summary.py`. **TODO:** spot-check panel PNGs on a large screen and confirm all printed stats match the source notebook. |
| S2 | complete | Submitted scatter grid; stimulus-pair summary via script |
| S3 | skeleton | vs `figS3_create_trained_threshold_df.ipynb` (mostly pipeline viz) |
| S4 | review | Schematic; confirm against filter intuition if needed |
| S5 | complete | Training history (Fig 1); see revision split from old S5 draft |
| S6 | complete | Within-session structure (Fig 3); edge-trim autocorrelation |
- [ ] S5 — superseded note: original single-notebook draft held stage 1
      violation rate, time in stage 1, and time to trained versus trained
      violation rate. Panel letters and saved outputs not yet assigned.

## Model-fit baselines

The submitted fit objects are tracked in historical commit `8358b9f` under
`notebooks/publication/data/model_fits/`. The canonical set contains four
Figure 3 objects, two Figure 4 objects, and two Figure 5 objects. The unused
`2025_07_22_fig4_compare_intrxs_rs47.pkl` reference is not a migration input.

Each baseline export must record:

- source commit and SHA-256 checksum;
- data checksum and animal IDs;
- feature order and model parameters;
- train/test session identities and seed;
- tau/sigma grids and selected values;
- weights, NLLs, trial counts, and figure-ready summary values.
