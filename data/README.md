# Publication data

- `processed/all_animals_cleaned.parquet`: all reconstructed training stages for
  the 17-animal reconstruction cohort (including W082); used by Supporting
  Figures S1–S3 and S5.
- `processed/all_animals_trained_cleaned.parquet`: final 1,866,003-trial model
  dataset for the 16-animal modeling cohort (excluding W082), used by Figures
  1 and 3–5.
- `raw/`: immutable source artifacts distributed through Zenodo.
- `model_fits/`: optional full fit archives distributed through Zenodo.

Parquet files are available locally during development but are ignored by Git.
Their source checksums and generated manifests are versioned. The final Zenodo
DOI will be added before release.
