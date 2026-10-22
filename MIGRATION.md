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

- [x] Figure 1 — notebook panels B–F done (sequences B; C–F histograms/choice;
      choice E, violation F); hit–violation block in S5. Panel A task timing
      still to be added in Affinity Designer before Overleaf upload.
- [x] Figure 2 — binary and multinomial GLM schematics
- [x] Figure 3 — violation history, corrected autocorrelation, and model fits
- [x] Figure 4 — post-violation performance and psychometric simulations
- [x] Figure 5 — model comparisons

## Supporting figures

- [x] S1 — dataset alignment
- [x] S2 — stimulus pairs by animal
- [x] S3 — trained-session threshold
- [x] S4 — exponential-filter schematic
- [x] S5 / S6 — `suppS5.ipynb` (training history, cited with Fig 1) and
      `suppS6.ipynb` (within-session structure, cited with Fig 3). The
      hit–violation regression moved into Figure 1.
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
