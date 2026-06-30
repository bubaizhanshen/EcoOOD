from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import pandas as pd


ZIP_MEMBER_PATTERN = re.compile(r"(DSSTox_CCD_dump_[^<\"'\s]+)")

CANONICAL_COLUMNS = [
    "dtxsid",
    "casrn",
    "chemical_name",
    "smiles",
    "inchikey",
    "molecular_weight",
    "logp",
]

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dtxsid": (
        "DTXSID",
        "dtxsid",
        "dsstox_substance_id",
        "DSSTOX_SUBSTANCE_ID",
        "DSSTox_GSID",
        "DSSTox_Generic_SID",
        "DSSTox_RID",
    ),
    "casrn": (
        "CASRN",
        "casrn",
        "TS_CASRN",
        "TestSubstance_CASRN",
    ),
    "chemical_name": (
        "PREFERRED_NAME",
        "preferred_name",
        "CHEMICAL_NAME",
        "chemical_name",
        "ChemName",
        "TS_ChemName",
        "TestSubstance_ChemicalName",
    ),
    "smiles": (
        "SMILES",
        "smiles",
        "STRUCTURE_SMILES",
        "STRUCTURE_SMILES_Desalt",
        "Canonical_QSARr",
        "canonical_qsarr",
        "QSAR_READY_SMILES",
        "qsar_ready_smiles",
        "CANONICAL_SMILES",
    ),
    "inchikey": (
        "INCHIKEY",
        "InChIKey",
        "inchikey",
        "STRUCTURE_InChIKey_v0",
        "STRUCTURE_INCHIKEY",
    ),
    "molecular_weight": (
        "AVERAGE_MASS",
        "average_mass",
        "MOL_WEIGHT",
        "mol_weight",
        "MOLECULAR_WEIGHT",
        "molecular_weight",
        "STRUCTURE_MW",
        "MolecularWeight",
    ),
    "logp": (
        "XLOGP",
        "XLogP",
        "xlogp",
        "logp",
        "LOGP",
        "KOWWIN_LOGP",
        "OPERA_LogP",
        "physchem_logp",
    ),
}


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def normalize_casrn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^0-9]+", "", str(value))


def resolve_dsstox_columns(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_normalize_column(column): column for column in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            hit = normalized.get(_normalize_column(alias))
            if hit is not None:
                resolved[canonical] = hit
                break
    return resolved


def expand_source_paths(sources: Sequence[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_source in sources:
        source = Path(raw_source)
        if any(token in str(raw_source) for token in ("*", "?", "[")):
            expanded.extend(sorted(Path().glob(str(raw_source))))
            continue
        if source.is_dir():
            expanded.extend(sorted(source.glob("*.csv")))
            continue
        if source.exists():
            expanded.append(source)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def parse_clowder_zip_listing(html_text: str) -> pd.DataFrame:
    members: list[str] = []
    seen: set[str] = set()
    for raw_member in ZIP_MEMBER_PATTERN.findall(html_text):
        member = raw_member.replace("&amp;", "&").strip()
        if member not in seen:
            members.append(member)
            seen.add(member)
    return pd.DataFrame(
        {
            "path": members,
            "basename": [Path(member).name for member in members],
            "is_directory": [member.endswith("/") for member in members],
        }
    )


def read_dsstox_catalog(paths: Sequence[str | Path]) -> pd.DataFrame:
    resolved_paths = expand_source_paths(paths)
    frames: list[pd.DataFrame] = []
    for path in resolved_paths:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as zf:
                members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                for member in members:
                    with zf.open(member) as fh:
                        columns = pd.read_csv(fh, nrows=0).columns.tolist()
                    mapping = resolve_dsstox_columns(columns)
                    if not {"smiles"} & set(mapping):
                        continue
                    usecols = sorted(set(mapping.values()))
                    with zf.open(member) as fh:
                        frame = pd.read_csv(fh, usecols=usecols, dtype=str, low_memory=False)
                    frame = frame.rename(columns={value: key for key, value in mapping.items()})
                    for column in CANONICAL_COLUMNS:
                        if column not in frame.columns:
                            frame[column] = pd.NA
                    frame = frame[CANONICAL_COLUMNS].copy()
                    frames.append(frame)
            continue
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        mapping = resolve_dsstox_columns(columns)
        if not {"smiles"} & set(mapping):
            continue
        usecols = sorted(set(mapping.values()))
        frame = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
        frame = frame.rename(columns={value: key for key, value in mapping.items()})
        for column in CANONICAL_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[CANONICAL_COLUMNS].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    catalog = pd.concat(frames, ignore_index=True)
    for column in CANONICAL_COLUMNS:
        catalog[column] = catalog[column].fillna("").astype(str).str.strip()
    catalog = catalog[(catalog["smiles"] != "") | (catalog["dtxsid"] != "") | (catalog["casrn"] != "")].copy()
    catalog["dtxsid"] = catalog["dtxsid"].str.upper()
    return catalog


def _read_filtered_frames(
    path: Path,
    *,
    target_dtxsid: set[str],
    target_casrn: set[str],
    chunksize: int = 100_000,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    def consume_reader(reader) -> None:
        for chunk in reader:
            chunk = chunk.rename(columns={value: key for key, value in mapping.items()})
            for column in CANONICAL_COLUMNS:
                if column not in chunk.columns:
                    chunk[column] = pd.NA
            chunk = chunk[CANONICAL_COLUMNS].copy()
            chunk["dtxsid"] = chunk["dtxsid"].fillna("").astype(str).str.strip().str.upper()
            chunk["casrn"] = chunk["casrn"].fillna("").astype(str).str.strip()
            chunk["casrn_normalized"] = chunk["casrn"].map(normalize_casrn)
            mask = chunk["dtxsid"].isin(target_dtxsid) | chunk["casrn_normalized"].isin(target_casrn)
            if mask.any():
                frames.append(chunk.loc[mask, CANONICAL_COLUMNS].copy())

    if path.suffix.lower() == ".zip":
        with ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            for member in members:
                with zf.open(member) as fh:
                    columns = pd.read_csv(fh, nrows=0).columns.tolist()
                mapping = resolve_dsstox_columns(columns)
                if not {"smiles"} & set(mapping):
                    continue
                usecols = sorted(set(mapping.values()))
                with zf.open(member) as fh:
                    reader = pd.read_csv(fh, usecols=usecols, dtype=str, low_memory=False, chunksize=chunksize)
                    consume_reader(reader)
        return frames

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    mapping = resolve_dsstox_columns(columns)
    if not {"smiles"} & set(mapping):
        return frames
    usecols = sorted(set(mapping.values()))
    reader = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, chunksize=chunksize)
    consume_reader(reader)
    return frames


def _empty_like(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _fill_from_match(
    base: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    source_name: str,
) -> pd.DataFrame:
    if left_on not in base.columns or right_on not in catalog.columns:
        return base
    subset = catalog[catalog[right_on] != ""].copy()
    if subset.empty:
        return base
    subset["_non_empty"] = (_empty_like(subset["smiles"]) != "").astype(int)
    if right_on == "casrn":
        subset["_match_key"] = subset[right_on].map(normalize_casrn)
        base_key = base[left_on].map(normalize_casrn)
    else:
        subset["_match_key"] = subset[right_on]
        base_key = base[left_on]
    subset = subset.sort_values(["_non_empty", "_match_key"], ascending=[False, True]).drop_duplicates("_match_key")
    match_cols = ["_match_key", "smiles", "inchikey", "molecular_weight", "logp"]
    matched = pd.DataFrame({"_match_key": base_key}).merge(subset[match_cols], on="_match_key", how="left")
    for column in ["smiles", "inchikey", "molecular_weight", "logp"]:
        base_empty = (_empty_like(base[column]) == "").to_numpy()
        matched_non_empty = (_empty_like(matched[column]) != "").to_numpy()
        mask = base_empty & matched_non_empty
        if mask.any():
            base.loc[mask, column] = matched.loc[mask, column].to_numpy()
            base.loc[mask, "resolved"] = True
            base.loc[mask, "resolution_source"] = source_name
    return base


def resolve_chemical_index(chemical_index: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    resolved = chemical_index.copy().reset_index(drop=True)
    for column in ["smiles", "inchikey", "molecular_weight", "logp"]:
        resolved[column] = ""
    resolved["query"] = resolved["dtxsid"].fillna("").astype(str).str.strip()
    empty_query = resolved["query"] == ""
    resolved.loc[empty_query, "query"] = resolved.loc[empty_query, "cas_number"].fillna("").astype(str).str.strip()
    resolved["resolution_source"] = ""
    resolved["resolved"] = False

    if catalog.empty:
        return resolved[
            ["cas_number", "dtxsid", "chemical_name", "query", "smiles", "molecular_weight", "logp", "inchikey", "resolved", "resolution_source"]
        ]

    working_catalog = catalog.copy()
    working_catalog["dtxsid"] = _empty_like(working_catalog["dtxsid"]).str.upper()
    resolved["dtxsid"] = resolved["dtxsid"].fillna("").astype(str).str.strip().str.upper()
    resolved["cas_number"] = resolved["cas_number"].fillna("").astype(str).str.strip()

    resolved = _fill_from_match(
        resolved,
        working_catalog,
        left_on="dtxsid",
        right_on="dtxsid",
        source_name="dsstox_dtxsid",
    )
    resolved = _fill_from_match(
        resolved,
        working_catalog,
        left_on="cas_number",
        right_on="casrn",
        source_name="dsstox_casrn",
    )
    return resolved[
        ["cas_number", "dtxsid", "chemical_name", "query", "smiles", "molecular_weight", "logp", "inchikey", "resolved", "resolution_source"]
    ]


def resolve_chemical_index_from_sources(
    chemical_index: pd.DataFrame,
    sources: Sequence[str | Path],
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    target_dtxsid = (
        chemical_index.get("dtxsid", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    target_dtxsid = {value for value in target_dtxsid if value}
    target_casrn = (
        chemical_index.get("cas_number", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .map(normalize_casrn)
    )
    target_casrn = {value for value in target_casrn if value}

    frames: list[pd.DataFrame] = []
    for path in expand_source_paths(sources):
        frames.extend(
            _read_filtered_frames(
                path,
                target_dtxsid=target_dtxsid,
                target_casrn=target_casrn,
                chunksize=chunksize,
            )
        )
    if not frames:
        return resolve_chemical_index(chemical_index, pd.DataFrame(columns=CANONICAL_COLUMNS))
    catalog = pd.concat(frames, ignore_index=True).drop_duplicates(["dtxsid", "casrn", "smiles"], keep="first")
    return resolve_chemical_index(chemical_index, catalog)
