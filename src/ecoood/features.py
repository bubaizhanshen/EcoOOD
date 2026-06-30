from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import DEFAULT_SCHEMA, EcoOODSchema

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised when rdkit is unavailable.
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None
    MurckoScaffold = None


RDKit_DESCRIPTOR_MAP = {
    "physchem_mol_wt": lambda mol: float(Descriptors.MolWt(mol)),
    "physchem_logp": lambda mol: float(Descriptors.MolLogP(mol)),
    "physchem_tpsa": lambda mol: float(Descriptors.TPSA(mol)),
    "physchem_hba": lambda mol: float(Descriptors.NumHAcceptors(mol)),
    "physchem_hbd": lambda mol: float(Descriptors.NumHDonors(mol)),
    "physchem_rot_bonds": lambda mol: float(Descriptors.NumRotatableBonds(mol)),
    "physchem_ring_count": lambda mol: float(Descriptors.RingCount(mol)),
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def smiles_to_mol(smiles: str | None) -> Any:
    if Chem is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def make_scaffold(smiles: str | None) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None or MurckoScaffold is None:
        token = (smiles or "missing").replace("/", "").replace("\\", "")
        return token[:16] or "missing"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "acyclic"


def binary_fingerprints(smiles: pd.Series, n_bits: int = 2048) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    for value in smiles.fillna(""):
        value = str(value)
        if not value.strip():
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        mol = smiles_to_mol(value)
        if AllChem is None or DataStructs is None:
            rng = np.random.default_rng(abs(hash(value)) % (2**32))
            row = np.zeros(n_bits, dtype=np.float32)
            row[rng.choice(n_bits, size=min(16, n_bits), replace=False)] = 1.0
            rows.append(row)
            continue
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return sparse.csr_matrix(np.vstack(rows))


def attach_rdkit_descriptors(df: pd.DataFrame, schema: EcoOODSchema = DEFAULT_SCHEMA) -> pd.DataFrame:
    if Chem is None or Descriptors is None:
        return df.copy()

    result = df.copy()
    missing_cols = [name for name in RDKit_DESCRIPTOR_MAP if name not in result.columns]
    if not missing_cols:
        return result

    values = {name: [] for name in missing_cols}
    for smiles in result[schema.smiles].fillna(""):
        mol = smiles_to_mol(smiles)
        for name in missing_cols:
            if mol is None:
                values[name].append(np.nan)
            else:
                values[name].append(RDKit_DESCRIPTOR_MAP[name](mol))
    for name, column in values.items():
        result[name] = column
    return result


@dataclass
class FeatureBundle:
    full: sparse.csr_matrix
    fingerprint: sparse.csr_matrix
    descriptor: np.ndarray
    species: np.ndarray
    context: np.ndarray
    mechanism: np.ndarray
    feature_names: list[str]


class EcoFeatureBuilder:
    def __init__(
        self,
        schema: EcoOODSchema = DEFAULT_SCHEMA,
        fingerprint_bits: int = 2048,
    ) -> None:
        self.schema = schema
        self.fingerprint_bits = fingerprint_bits
        self.tabular_transformer: ColumnTransformer | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.descriptor_cols: list[str] = []
        self.species_cols: list[str] = []
        self.context_cols: list[str] = []
        self.mechanism_cols: list[str] = []

    def _infer_columns(self, df: pd.DataFrame) -> None:
        schema = self.schema
        numeric_cols = [
            col
            for col in df.columns
            if (
                col not in schema.protected_columns
                and pd.api.types.is_numeric_dtype(df[col])
            )
        ]
        categorical_seed = [
            schema.endpoint,
            schema.chemical_class,
            schema.species,
            schema.genus,
            schema.family,
            schema.order,
            schema.clazz,
            schema.phylum,
            schema.trophic_group,
            schema.medium,
            schema.source,
        ]
        self.categorical_cols = [col for col in categorical_seed if col in df.columns]

        self.descriptor_cols = [
            col for col in numeric_cols if col.startswith("physchem_")
        ]
        self.mechanism_cols = [col for col in numeric_cols if col.startswith("mech_")]
        self.context_cols = [
            col
            for col in numeric_cols
            if col in {schema.duration_h, schema.temperature_c, schema.ph, schema.study_year}
            or col.startswith("ctx_")
        ]
        self.species_cols = [
            col
            for col in numeric_cols
            if col.startswith("tax_")
        ]
        used = set(
            self.descriptor_cols
            + self.mechanism_cols
            + self.context_cols
            + self.species_cols
        )
        residual = [col for col in numeric_cols if col not in used]
        self.descriptor_cols.extend(residual)
        self.numeric_cols = (
            self.descriptor_cols
            + self.mechanism_cols
            + self.context_cols
            + self.species_cols
        )

    def fit(self, df: pd.DataFrame) -> "EcoFeatureBuilder":
        self._infer_columns(df)
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        self.tabular_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_cols),
                ("categorical", categorical_pipeline, self.categorical_cols),
            ],
            sparse_threshold=0.0,
        )
        self.tabular_transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> FeatureBundle:
        if self.tabular_transformer is None:
            raise RuntimeError("EcoFeatureBuilder must be fit before transform().")
        augmented = attach_rdkit_descriptors(df, self.schema)
        tabular = self.tabular_transformer.transform(augmented)
        tabular = sparse.csr_matrix(tabular)
        fingerprint = binary_fingerprints(augmented[self.schema.smiles], self.fingerprint_bits)

        def _dense(cols: list[str]) -> np.ndarray:
            if not cols:
                return np.zeros((len(augmented), 1), dtype=np.float32)
            subset = augmented[cols].apply(_safe_numeric).to_numpy(dtype=np.float32)
            return np.nan_to_num(subset, nan=0.0)

        descriptor = _dense(self.descriptor_cols)
        species = _dense(self.species_cols)
        context = _dense(self.context_cols)
        mechanism = _dense(self.mechanism_cols)

        full = sparse.hstack([fingerprint, tabular], format="csr")
        feature_names = [f"fp_{i}" for i in range(self.fingerprint_bits)]
        feature_names.extend(self.tabular_feature_names())
        return FeatureBundle(
            full=full,
            fingerprint=fingerprint,
            descriptor=descriptor,
            species=species,
            context=context,
            mechanism=mechanism,
            feature_names=feature_names,
        )

    def fit_transform(self, df: pd.DataFrame) -> FeatureBundle:
        return self.fit(df).transform(df)

    def tabular_feature_names(self) -> list[str]:
        if self.tabular_transformer is None:
            return []
        return list(self.tabular_transformer.get_feature_names_out())
