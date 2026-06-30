from __future__ import annotations

from scripts.build_demo_dataset import make_demo_dataset
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


def test_chemical_random_split_holds_out_entire_chemicals() -> None:
    df = make_demo_dataset(n=240, seed=123)
    split = build_split(df, split="chemical_random", schema=DEFAULT_SCHEMA, seed=123)
    train_chemicals = set(df.loc[split.train, DEFAULT_SCHEMA.chemical_id])
    test_chemicals = set(df.loc[split.test, DEFAULT_SCHEMA.chemical_id])
    assert train_chemicals.isdisjoint(test_chemicals)
    assert split.split_name == "chemical_random"
    assert split.test_is_ood.all()
