# Data

This directory contains the small public demo input used by EcoOOD examples and
tests. Full benchmark tables are generated locally from ECOTOX,
DSSTox/CompTox, invitrodb/ToxCast, ECHA, and PMRA source data.

## Layout

- `processed/demo_ecoood.csv`: synthetic dashboard and benchmark example
- `raw/`: provider downloads created by the source-specific fetch scripts
- `processed/`: locally built benchmark tables and feature caches

Only the demo table is versioned. The full processed files are rebuilt with the
commands in the main [README](../README.md).

## Local Files

Typical local paths are:

- `data/raw/`
- full `data/processed/ecotox_acute_ecoood_*` benchmark tables
- `data/processed/invitrodb_mechanism_features*.csv`
- local PubChem/CompTox cache expansions
- partial downloads such as `*.part`
