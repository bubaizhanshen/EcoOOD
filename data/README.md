# Data

This directory contains the small public demo input used by EcoOOD examples and
tests, together with a field-level predictor manifest. The frozen benchmark,
split assignments, and compact prediction outputs are available in the
[v0.2.0 analysis release](https://github.com/bubaizhanshen/EcoOOD/releases/tag/v0.2.0).

## Layout

- `processed/demo_ecoood.csv`: synthetic benchmark example
- `feature_manifest.csv`: field names, roles, cardinalities, and missingness
  rates for the frozen integrity-audited benchmark snapshot (4,942 records;
  841 chemicals)
- `raw/`: provider downloads created by the source-specific fetch scripts
- `processed/`: locally built benchmark tables and feature caches

The full processed tables can also be rebuilt from local ECOTOX,
DSSTox/CompTox, invitrodb/ToxCast, ECHA, and PMRA source files with the commands
in the main [README](../README.md).

## Local Files

Typical local paths are:

- `data/raw/`
- full `data/processed/ecotox_acute_ecoood_*` benchmark tables
- `data/processed/invitrodb_mechanism_features*.csv` (historical file name;
  used as a bioactivity-proxy cache)
- local PubChem/CompTox cache expansions
- partial downloads such as `*.part`
