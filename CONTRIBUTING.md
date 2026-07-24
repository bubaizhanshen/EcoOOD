# Contributing to EcoOOD

Thanks for contributing.

## Scope

EcoOOD is a research codebase for reliability assessment in aquatic
ecotoxicity screening. Useful contributions include:

- bug fixes and reproducibility improvements
- documentation and command-line usability improvements
- new benchmark analyses that match the project scope
- better data validation, curation, and quality-control utilities
- carefully justified model or OOD baselines

## Before You Start

- Open an issue for substantial changes before writing a large patch.
- Keep contributions aligned with acute aquatic ecotoxicity screening
  reliability.
- Avoid committing raw third-party downloads or large intermediate artifacts.

## Development Setup

Create the reference environment:

```bash
conda create -y -n ecoood python=3.11 pip
conda install -y -n ecoood -c conda-forge \
  pandas scikit-learn scipy pyarrow pyyaml requests joblib beautifulsoup4 \
  pytest matplotlib seaborn networkx lightgbm xgboost openpyxl rdkit
conda run -n ecoood python -m pip install -e . --no-deps
```

## Workflow

1. Create a topic branch from `main`.
2. Make focused changes with clear commit messages.
3. Add or update tests when behavior changes.
4. Run the validation commands below.
5. Open a pull request with a concise summary and rationale.

## Validation

Minimum checks:

```bash
conda run -n ecoood pytest -q
conda run -n ecoood python -m compileall -q src scripts tests
```

If you change benchmark logic, also note:

- which dataset file was used
- which splits/models were rerun
- where updated outputs were written

## Style

- Use Python 3.11-compatible syntax.
- Keep edits ASCII unless a file already requires Unicode.
- Prefer small, composable utilities over large monolithic scripts.
- Treat screening terminology carefully:
  - use `prediction reliability` or `screening reliability` for model-side risk
  - do not relabel the project as a full ecological risk-assessment engine

## Data and Artifact Policy

Do not commit:

- `data/raw/`
- full processed benchmark tables under `data/processed/`; keep only
  `data/processed/demo_ecoood.csv` in GitHub
- generated figures, result tables, and temporary benchmark outputs
- credentials or secrets

Generated tables and PDF/PNG/SVG/TIF files should remain local build artifacts
unless they are intentionally attached to a separate release or data archive.

## Pull Request Checklist

- [ ] The change is scoped and documented.
- [ ] Tests pass locally.
- [ ] New files follow repository naming and layout conventions.
- [ ] Large generated files were excluded unless they are intentional release assets.
- [ ] README, data notes, or figure manifests were updated if needed.
