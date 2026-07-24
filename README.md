# EcoOOD

[![Tests](https://github.com/bubaizhanshen/EcoOOD/actions/workflows/tests.yml/badge.svg)](https://github.com/bubaizhanshen/EcoOOD/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

EcoOOD is a Python workflow for reliability assessment and review
prioritization in aquatic ecotoxicity screening. It evaluates
chemical-species-context predictions after a toxicity model has produced an
estimate and measures how well reliability signals intercept high-error,
apparently low-concern predictions under a fixed review budget.

EcoOOD is the full assessment framework. The **EcoOOD score** is one component
of that framework: a logistic high-error risk score fitted to calibration-fold
residual labels. The benchmark compares it with structural similarity,
input-space distance, ensemble uncertainty, and matched-supervision risk
models.

## Workflow

1. Route unresolved identities, mixtures, and unsupported molecular
   representations to deterministic `withhold/review`.
2. Fit the toxicity predictor and reliability models using separate training
   and calibration partitions.
3. Quantify chemical, biological, contextual, bioactivity-proxy, and ensemble
   uncertainty signals for scoreable cases.
4. Compare review strategies at matched workload using endpoint-relative,
   chemical-level outcomes.
5. Assign predictions to `screen now`, `lower priority`, `withhold/review`, or
   `prioritize testing`.

## Repository Layout

```text
configs/                Fixed benchmark configurations
data/                   Synthetic benchmark input and predictor field manifest
scripts/                Data preparation, benchmark, and audit entry points
src/ecoood/             Python package
tests/                  Unit and regression tests
```

Generated benchmark outputs are stored locally under `outputs/`. The frozen
analysis tables and compact prediction outputs are distributed through the
[v0.2.0 analysis release](https://github.com/bubaizhanshen/EcoOOD/releases/tag/v0.2.0).

## Installation

Create the reference environment:

```bash
conda create -y -n ecoood python=3.11 pip
conda install -y -n ecoood -c conda-forge \
  pandas scikit-learn scipy pyarrow pyyaml requests joblib beautifulsoup4 \
  matplotlib seaborn networkx lightgbm xgboost openpyxl rdkit
conda run -n ecoood python -m pip install -e . --no-deps
```

For development:

```bash
conda run -n ecoood python -m pip install -e ".[dev]"
```

Run the tests:

```bash
conda run -n ecoood pytest -q
```

## Quick Start

The included synthetic table supports a short smoke test:

```bash
conda run -n ecoood python scripts/run_benchmark.py \
  --data data/processed/demo_ecoood.csv \
  --splits random scaffold temporal species chemical_class \
  --models random_forest \
  --members 2 \
  --output-dir outputs/demo_benchmark
```

## Reproduce the Benchmark

The frozen benchmark passed the input-feature and leakage audit and contains 4,942
records from 841 chemicals. Its SHA-256 checksum is:

```text
1e83dcc8f6b8086dcb1609a6f9f7e1a98634aff85692d910774f4f603a17461e
```

Download and unpack the release archive, then run the fixed seed panel:

```bash
conda run -n ecoood python scripts/run_integrity_benchmark.py \
  --data path/to/EcoOOD_benchmark_snapshot_structured.csv \
  --output-root outputs/integrity_benchmark
```

Summarize fixed-workload outcomes:

```bash
conda run -n ecoood python scripts/summarize_multiseed_screening.py \
  --input-root outputs/integrity_benchmark \
  --output-dir outputs/integrity_benchmark/aggregate
```

Run input-feature and leakage controls:

```bash
conda run -n ecoood python scripts/run_integrity_sensitivity.py \
  --data path/to/EcoOOD_benchmark_snapshot_structured.csv \
  --output-dir outputs/integrity_sensitivity
```

Additional audit entry points cover exact duplicates, reference holdout,
fixed temporal transfer, species-chemical overlap, configuration sensitivity,
local interval recalibration, and external dossier transfer. Run any script
with `--help` to inspect its inputs and output layout.

The named-class benchmark uses five fixed leave-one-class-out folds:
PFAS, conazoles, neonicotinoids, PPCPs, and strobins. A chemical carrying
multiple class labels enters the test fold whenever any label matches the
held-out class; the same atomic class is therefore absent from training and
calibration.

## Data Sources

The benchmark is derived from public ECOTOX, DSSTox/CompTox, and
invitrodb/ToxCast resources. The external transfer audit uses ECHA and PMRA
dossiers. Provider downloads are transformed locally with the scripts in this
repository; the release archive contains the frozen derived benchmark,
split assignments, compact predictions, selected analysis tables, and file
checksums.

The predictor field names, roles, cardinalities, and missingness rates are
listed in [`data/feature_manifest.csv`](data/feature_manifest.csv). Target
fields, identifiers, source-document metadata, and chemical-class labels are
excluded from the predictor matrix. Bioactivity fields are treated as partial
in vitro proxies.

To rebuild the structured table from local provider files:

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

The historical cache filename contains `mechanism`; the current analysis uses
these fields as bioactivity proxies.

## External Transfer Audit

```bash
conda run -n ecoood python scripts/build_pmra_regulatory_candidate_set.py
conda run -n ecoood python scripts/build_echa_pmra_external_rows.py
conda run -n ecoood python scripts/prepare_echa_pmra_validation_panel.py
conda run -n ecoood python scripts/build_echa_pmra_clean_panel.py
conda run -n ecoood python scripts/run_echa_pmra_external_validation.py
```

The external panel is chemical-identity-disjoint from the model table and is
reported with chemical-cluster uncertainty.

## License

Released under the [MIT License](LICENSE).
