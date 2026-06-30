# Data Policy

This directory contains the small public demo input used by EcoOOD examples and
tests. Full processed benchmark tables are generated locally from public
third-party sources and are intentionally not tracked in the GitHub repository.

## Included

- `processed/demo_ecoood.csv`: small dashboard/example table.

## Not Included

The repository does not redistribute raw third-party downloads, licensed
documents, reference PDFs, full processed benchmark tables, derived
invitrodb/ToxCast feature caches, or local cache files. Raw resources should be
obtained from their original providers with the download/build scripts in
`scripts/`, then rebuilt locally under `data/processed/`.

Ignored local locations include:

- `data/raw/`
- full `data/processed/ecotox_acute_ecoood_*` benchmark tables
- `data/processed/invitrodb_mechanism_features*.csv`
- local PubChem/CompTox cache expansions
- partial downloads such as `*.part`
