# EcoOOD

EcoOOD is a research-code release for reliability assessment in aquatic
ecotoxicity screening. It evaluates whether an aquatic toxicity prediction is
reliable enough for screening use by combining chemical, species, context,
bioactivity-proxy, and model-uncertainty evidence.
The reported screening actions are `screen now`, `lower priority`,
`withhold/review`, and `prioritize testing`.
Prediction-level diagnostic outputs may also use `predict`, `warn`, and
`abstain`; these are internal reliability states used before aggregation into
screening-action summaries.

Repository: https://github.com/bubaizhanshen/EcoOOD

This repository contains code, a small demo table, and reproducibility entry
points. Large source downloads and generated analysis artifacts are kept out
of version control.

## Why EcoOOD

Most ecotoxicity models are still reported under random splits and
structure-centered applicability-domain checks. EcoOOD focuses instead on the
deployment question:

**Is this toxicity prediction reliable enough to be propagated into downstream
environmental screening?**

## Main Features

- `ECOTOX` ingestion and acute aquatic endpoint curation
- `DSSTox`/`CompTox` structure and physicochemical enrichment
- `invitrodb`/`ToxCast` summary-level bioactivity proxy features
- deployment-relevant benchmark splits:
  - `random`
  - `scaffold`
  - `temporal`
  - `species`
  - `chemical_class`
  - `hard_ood`
- predictive baselines:
  - `random_forest`
  - `lightgbm`
  - optional `xgboost`
- classical and generic reliability baselines:
  - similarity AD
  - leverage AD
  - descriptor-range AD
  - input-space k-nearest-neighbor distance
  - interval width
  - descriptor-space Mahalanobis distance
  - Isolation Forest
  - Local Outlier Factor
- `EcoOOD` joint prediction-level scoring and split conformal uncertainty summaries
- screening-action validation under fixed review workloads
- 119-chemical class-focused screening workflow and downstream translation analyses

## Repository Layout

```text
app/                    Streamlit dashboard
configs/                Static configuration assets
data/processed/         Small demo data; full benchmark tables are rebuilt locally
scripts/                Data download, build, benchmark, and figure-generation scripts
src/ecoood/             Python package
tests/                  Regression tests
```

## What Is Included

- Reproducible Python code for data curation, benchmark splits, baseline
  models, EcoOOD scoring, and screening-action summaries.
- A small synthetic demo table for examples and tests.
- A local Streamlit dashboard for exploring prediction reliability and
  screening queues. The dashboard falls back to the demo table when full
  analysis tables are absent.

## Source Data

Raw ECOTOX, DSSTox/CompTox, invitrodb/ToxCast, ECHA, and PMRA downloads are
obtained from the original providers with the source-specific build steps
below. Generated benchmark tables remain local build outputs.

## Quick Start

Create the reference environment:

```bash
conda create -y -n ecoood python=3.11 pip
conda install -y -n ecoood -c conda-forge pandas scikit-learn scipy pyarrow pyyaml requests joblib pytest matplotlib seaborn networkx lightgbm xgboost openpyxl rdkit
conda run -n ecoood python -m pip install -e . --no-deps
```

Optional dashboard dependencies:

```bash
conda run -n ecoood python -m pip install ".[ui]" --no-deps
```

Run the test suite:

```bash
conda run -n ecoood pytest -q
```

## Interactive Dashboard

Launch the local dashboard:

```bash
conda run -n ecoood python scripts/run_dashboard.py --headless --port 8501
```

Then open `http://127.0.0.1:8501`.

Current views:

- benchmark overview across deployment splits
- toxicity-vs-EcoOOD decision maps
- class-focused screening queue exploration
- retrospective screening-action validation
- upload of user-scored CSVs with at least `y_pred` and `ecoood_score`

## Data and Reproducibility

This repository keeps code, a small synthetic demo table, and analysis scripts
under version control. Full processed benchmark tables and large third-party
downloads are rebuilt locally from the original public providers. Generated
analysis tables are written to ignored output directories.

### Raw data fetch

Download official EPA assets:

```bash
conda run -n ecoood python scripts/download_epa_data.py
```

Optional large-file helper for public Clowder blobs:

```bash
conda run -n ecoood python scripts/download_clowder_file.py \
  --url 'https://clowder.edap-cluster.com/files/69529775e4b0731a616efc4b/blob' \
  --output data/raw/DSSTox_CCD_dump_12092025_CSVs.zip \
  --chunk-size-mb 1
```

### Build the main processed dataset

```bash
conda run -n ecoood python scripts/build_ecotox_dataset.py \
  --structure-cache data/raw/dsstox_priority_cache_1000.csv \
  --dsstox-source data/raw/DSSTox_CCD_dump_12092025_CSVs.zip \
  --mechanism-cache data/processed/invitrodb_mechanism_features_v43.csv \
  --invitrodb-summary data/raw/INVITRODB_SUMMARY.zip \
  --max-chemicals 1000 \
  --output data/processed/ecotox_acute_ecoood_1000chem_dsstox_mech.csv \
  --structured-output data/processed/ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv
```

The `--mechanism-cache` file is a derived ToxCast/invitrodb bioactivity-proxy
feature table. If it is not present locally, generate it from the public
invitrodb summary inputs with the same build script or provide an equivalent
locally generated cache.

### Run a benchmark

```bash
conda run -n ecoood python scripts/run_benchmark.py \
  --data data/processed/ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv \
  --splits random scaffold temporal species chemical_class \
  --models random_forest lightgbm \
  --members 5 \
  --output-dir outputs/demo_benchmark
```

For a quick smoke test that does not require third-party data, use the included
synthetic demo table:

```bash
conda run -n ecoood python scripts/build_demo_dataset.py --rows 400
conda run -n ecoood python scripts/run_benchmark.py \
  --data data/processed/demo_ecoood.csv \
  --splits random scaffold temporal species chemical_class \
  --models random_forest \
  --members 2 \
  --output-dir outputs/demo_benchmark
```

### Reproduce analysis outputs

Analysis tables are generated from the curated processed tables and should be
written to a local ignored output directory:

```bash
conda run -n ecoood python scripts/generate_policy_screening_panel.py
conda run -n ecoood python scripts/generate_screening_gate_validation.py
conda run -n ecoood python scripts/run_lgbm_config_sensitivity.py
```

### External screening panel

The ECHA/PMRA external panel workflow is split into source retrieval, structure
resolution, panel cleaning, and case-level validation:

```bash
conda run -n ecoood python scripts/build_pmra_regulatory_candidate_set.py
conda run -n ecoood python scripts/build_echa_pmra_external_rows.py
conda run -n ecoood python scripts/prepare_echa_pmra_validation_panel.py
conda run -n ecoood python scripts/build_echa_pmra_clean_panel.py
conda run -n ecoood python scripts/run_echa_pmra_external_validation.py
```

The scripts write retrieved regulatory-source files and intermediate tables to
local ignored output directories.

### Generate local figures

```bash
conda run -n ecoood python scripts/generate_analysis_figures.py
conda run -n ecoood python scripts/generate_workflow_schematics.py
```

Generated PDF/PNG/SVG/TIF files are local build artifacts and are ignored by
Git.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, data policy, and
pull-request guidance.

## License

This repository is released under the [MIT License](LICENSE).
