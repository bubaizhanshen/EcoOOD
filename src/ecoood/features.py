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
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised when rdkit is unavailable.
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None
    rdFingerprintGenerator = None
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

_MORGAN_GENERATORS: dict[int, Any] = {}


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


def _morgan_fingerprint(mol: Any, n_bits: int) -> Any:
    if rdFingerprintGenerator is not None:
        generator = _MORGAN_GENERATORS.get(n_bits)
        if generator is None:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
            _MORGAN_GENERATORS[n_bits] = generator
        return generator.GetFingerprint(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def binary_fingerprints(smiles: pd.Series, n_bits: int = 2048) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    for value in smiles.fillna(""):
        value = str(value)
        if not value.strip():
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        mol = smiles_to_mol(value)
        if DataStructs is None or (rdFingerprintGenerator is None and AllChem is None):
            rng = np.random.default_rng(abs(hash(value)) % (2**32))
            row = np.zeros(n_bits, dtype=np.float32)
            row[rng.choice(n_bits, size=min(16, n_bits), replace=False)] = 1.0
            rows.append(row)
            continue
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = _morgan_fingerprint(mol, n_bits)
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
    species: np.ndarray | sparse.csr_matrix
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
        self.species_categorical_cols: list[str] = []
        self.context_categorical_cols: list[str] = []
        self.dense_transformers: dict[str, ColumnTransformer] = {}

    def _infer_columns(self, df: pd.DataFrame) -> None:
        schema = self.schema
        categorical_seed = [
            schema.endpoint,
            schema.species,
            schema.genus,
            schema.family,
            schema.order,
            schema.clazz,
            schema.phylum,
            schema.trophic_group,
            schema.medium,
        ]
        self.categorical_cols = [col for col in categorical_seed if col in df.columns]

        self.descriptor_cols = [
            col
            for col in df.columns
            if col.startswith("physchem_") and pd.api.types.is_numeric_dtype(df[col])
        ]
        self.mechanism_cols = [
            col
            for col in df.columns
            if col.startswith("mech_") and pd.api.types.is_numeric_dtype(df[col])
        ]
        self.context_cols = [
            col
            for col in df.columns
            if (
                col in {schema.duration_h, schema.temperature_c, schema.ph, schema.study_year}
                or col.startswith("ctx_")
            )
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        self.species_cols = [
            col
            for col in df.columns
            if col.startswith("tax_") and pd.api.types.is_numeric_dtype(df[col])
        ]
        self.species_categorical_cols = [
            col
            for col in (
                schema.species,
                schema.genus,
                schema.family,
                schema.order,
                schema.clazz,
                schema.phylum,
                schema.trophic_group,
            )
            if col in df.columns
        ]
        self.context_categorical_cols = [
            col for col in (schema.endpoint, schema.medium) if col in df.columns
        ]
        self.numeric_cols = (
            self.descriptor_cols
            + self.mechanism_cols
            + self.context_cols
            + self.species_cols
        )

    @staticmethod
    def _numeric_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )

    @staticmethod
    def _categorical_pipeline(*, sparse_output: bool) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "impute",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output),
                ),
            ]
        )

    def _fit_dense_transformer(
        self,
        name: str,
        df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str] | None = None,
        *,
        sparse_output: bool = False,
    ) -> None:
        categorical_cols = categorical_cols or []
        transformers = []
        if numeric_cols:
            transformers.append(("numeric", self._numeric_pipeline(), numeric_cols))
        if categorical_cols:
            transformers.append(
                (
                    "categorical",
                    self._categorical_pipeline(sparse_output=sparse_output),
                    categorical_cols,
                )
            )
        if not transformers:
            return
        transformer = ColumnTransformer(
            transformers=transformers,
            sparse_threshold=1.0 if sparse_output else 0.0,
        )
        transformer.fit(df)
        self.dense_transformers[name] = transformer

    def fit(self, df: pd.DataFrame) -> "EcoFeatureBuilder":
        self._infer_columns(df)
        self.tabular_transformer = ColumnTransformer(
            transformers=[
                ("numeric", self._numeric_pipeline(), self.numeric_cols),
                (
                    "categorical",
                    self._categorical_pipeline(sparse_output=True),
                    self.categorical_cols,
                ),
            ],
            sparse_threshold=1.0,
        )
        self.tabular_transformer.fit(df)
        self.dense_transformers = {}
        self._fit_dense_transformer("descriptor", df, self.descriptor_cols)
        self._fit_dense_transformer(
            "species",
            df,
            self.species_cols,
            self.species_categorical_cols,
            sparse_output=True,
        )
        self._fit_dense_transformer(
            "context",
            df,
            self.context_cols,
            self.context_categorical_cols,
        )
        self._fit_dense_transformer("mechanism", df, self.mechanism_cols)
        return self

    def transform(self, df: pd.DataFrame) -> FeatureBundle:
        if self.tabular_transformer is None:
            raise RuntimeError("EcoFeatureBuilder must be fit before transform().")
        augmented = attach_rdkit_descriptors(df, self.schema)
        tabular = self.tabular_transformer.transform(augmented)
        tabular = sparse.csr_matrix(tabular)
        fingerprint = binary_fingerprints(augmented[self.schema.smiles], self.fingerprint_bits)

        def _dense(name: str) -> np.ndarray | sparse.csr_matrix:
            transformer = self.dense_transformers.get(name)
            if transformer is None:
                return np.empty((len(augmented), 0), dtype=np.float32)
            values = transformer.transform(augmented)
            if sparse.issparse(values):
                return sparse.csr_matrix(values, dtype=np.float32)
            return np.nan_to_num(np.asarray(values, dtype=np.float32))

        descriptor = _dense("descriptor")
        species = _dense("species")
        context = _dense("context")
        mechanism = _dense("mechanism")

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
