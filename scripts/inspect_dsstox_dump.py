from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ecoood.dsstox import expand_source_paths, resolve_dsstox_columns


def inspect_columns(columns: list[str], label: str) -> dict[str, object]:
    resolved = resolve_dsstox_columns(columns)
    return {
        "file": label,
        "n_columns": len(columns),
        "matched_fields": ",".join(sorted(resolved)),
        "matched_columns": ",".join(f"{k}:{v}" for k, v in sorted(resolved.items())),
    }


def inspect_csv(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".zip":
        rows: list[dict[str, object]] = []
        with ZipFile(path) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with zf.open(member) as fh:
                    header = pd.read_csv(fh, nrows=0)
                rows.append(inspect_columns(header.columns.tolist(), member))
        return rows
    header = pd.read_csv(path, nrows=0)
    return [inspect_columns(header.columns.tolist(), path.name)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted DSSTox CSV dump files.")
    parser.add_argument("sources", nargs="+", help="CSV files, directories, or globs")
    args = parser.parse_args()

    rows = [row for path in expand_source_paths(args.sources) for row in inspect_csv(path)]
    if not rows:
        raise SystemExit("No CSV files found")
    frame = pd.DataFrame(rows).sort_values("file")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
