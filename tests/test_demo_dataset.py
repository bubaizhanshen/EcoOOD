from scripts.build_demo_dataset import make_demo_dataset

from ecoood.features import attach_rdkit_descriptors
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


def test_demo_dataset_supports_documented_benchmark_splits() -> None:
    frame = attach_rdkit_descriptors(make_demo_dataset(n=400, seed=42), DEFAULT_SCHEMA)

    for split_name in ("random", "scaffold", "temporal", "species", "chemical_class"):
        split = build_split(frame, split=split_name, schema=DEFAULT_SCHEMA, seed=42)
        assert len(split.train) > 0
        assert len(split.calib) > 0
        assert len(split.test) > 0
