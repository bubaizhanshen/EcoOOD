from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import make_scaffold
from .schema import DEFAULT_SCHEMA, EcoOODSchema


@dataclass
class SplitIndices:
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    split_name: str
    test_is_ood: np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _partition(indices: np.ndarray, rng: np.random.Generator, calib_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_calib = max(1, int(round(len(shuffled) * calib_fraction)))
    calib = shuffled[:n_calib]
    train = shuffled[n_calib:]
    return train, calib


def random_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    rng = _rng(seed)
    indices = df.index.to_numpy()
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test = shuffled[:n_test]
    remaining = shuffled[n_test:]
    train, calib = _partition(remaining, rng, calib_fraction)
    mask = np.zeros(len(test), dtype=bool)
    return SplitIndices(train=train, calib=calib, test=test, split_name="random", test_is_ood=mask)


def scaffold_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    working = df.copy()
    working["_scaffold"] = working[schema.smiles].map(make_scaffold)
    return group_holdout_split(
        working,
        group_col="_scaffold",
        split_name="scaffold",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def chemical_random_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    return group_holdout_split(
        df,
        group_col=schema.chemical_id,
        split_name="chemical_random",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def group_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    split_name: str,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    rng = _rng(seed)
    group_sizes = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
    groups = group_sizes.index.to_list()
    rng.shuffle(groups)
    held_out: list[str] = []
    held_out_rows = 0
    target_rows = max(1, int(round(len(df) * holdout_fraction)))
    for group in groups:
        held_out.append(group)
        held_out_rows += int(group_sizes.loc[group])
        if held_out_rows >= target_rows:
            break
    test_mask = df[group_col].isin(held_out).to_numpy()
    test = df.index.to_numpy()[test_mask]
    remaining = df.index.to_numpy()[~test_mask]
    train, calib = _partition(remaining, rng, calib_fraction)
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name=split_name,
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def time_split(
    df: pd.DataFrame,
    year_col: str,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    if year_col not in df.columns:
        raise KeyError(f"Missing temporal column '{year_col}'.")
    ordered = df[[year_col]].copy()
    ordered[year_col] = pd.to_numeric(ordered[year_col], errors="coerce")
    ordered = ordered.sort_values(year_col)
    n_test = max(1, int(round(len(df) * holdout_fraction)))
    test = ordered.tail(n_test).index.to_numpy()
    remaining = ordered.head(len(df) - n_test).index.to_numpy()
    train, calib = _partition(remaining, _rng(seed), calib_fraction)
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name="temporal",
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def hard_ood_split(
    df: pd.DataFrame,
    hard_ood_col: str,
    seed: int = 42,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    if hard_ood_col not in df.columns:
        raise KeyError(f"Missing hard OOD column '{hard_ood_col}'.")
    hard_mask = df[hard_ood_col].fillna(False).astype(bool).to_numpy()
    test = df.index.to_numpy()[hard_mask]
    remaining = df.index.to_numpy()[~hard_mask]
    train, calib = _partition(remaining, _rng(seed), calib_fraction)
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name="hard_ood",
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def species_holdout_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    level: str = "species",
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    if not hasattr(schema, level):
        raise KeyError(f"Unknown taxonomy level '{level}'.")
    return group_holdout_split(
        df,
        group_col=getattr(schema, level),
        split_name=f"{level}_holdout",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def build_split(
    df: pd.DataFrame,
    split: str,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
) -> SplitIndices:
    if split == "random":
        return random_split(df, seed=seed)
    if split == "scaffold":
        return scaffold_split(df, schema=schema, seed=seed)
    if split == "chemical_random":
        return chemical_random_split(df, schema=schema, seed=seed)
    if split == "chemical_class":
        return group_holdout_split(
            df,
            group_col=schema.chemical_class,
            split_name="chemical_class",
            seed=seed,
        )
    if split == "species":
        return species_holdout_split(df, schema=schema, level="species", seed=seed)
    if split == "genus":
        return species_holdout_split(df, schema=schema, level="genus", seed=seed)
    if split == "temporal":
        return time_split(df, year_col=schema.study_year, seed=seed)
    if split == "hard_ood":
        return hard_ood_split(df, hard_ood_col=schema.hard_ood, seed=seed)
    raise ValueError(f"Unsupported split '{split}'.")
