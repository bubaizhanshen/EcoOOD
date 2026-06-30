from __future__ import annotations

import io
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


INVITRODB_V43_BLOB_URL = "https://clowder.edap-cluster.com/api/files/68af6b70e4b02565fc7c3a98/blob"
H295R_HORMONE_COLS = [
    "OHPREG",
    "PROG",
    "OHPROG",
    "DOC",
    "CORTIC",
    "X11DCORT",
    "CORT",
    "ANDR",
    "TESTO",
    "E1",
    "E2",
]
LITERATURE_MODE_MAP = {
    "ar_literature_binding": "mech_lit_ar_binding",
    "ar_literature_agonist": "mech_lit_ar_agonist",
    "ar_literature_antagonist": "mech_lit_ar_antagonist",
    "er_literature_binding": "mech_lit_er_binding",
    "er_literature_agonist": "mech_lit_er_agonist",
    "er_literature_antagonist": "mech_lit_er_antagonist",
}
LITERATURE_SCORE_MAP = {
    "inactive": 0.0,
    "very weak": 0.25,
    "veryweak": 0.25,
    "weak": 0.5,
    "moderate": 0.75,
    "medium": 0.75,
    "strong": 1.0,
}


@dataclass(frozen=True)
class ZipMemberSpec:
    key: str
    filename: str
    data_offset: int
    compressed_size: int
    compression: int = 8


INVITRODB_V43_MEMBER_SPECS = {
    "cytotox": ZipMemberSpec(
        key="cytotox",
        filename="cytotox_invitrodb_v4_3_AUG2024.xlsx",
        data_offset=6027992,
        compressed_size=854523,
    ),
    "ar_er_literature": ZipMemberSpec(
        key="ar_er_literature",
        filename="endocrine_models/ar.er.lit.xlsx",
        data_offset=9169761,
        compressed_size=1562474,
    ),
    "ht_h295r": ZipMemberSpec(
        key="ht_h295r",
        filename="endocrine_models/ht.h295r.model.xlsx",
        data_offset=45514958,
        compressed_size=910456,
    ),
    "toxcast_ar_pathway": ZipMemberSpec(
        key="toxcast_ar_pathway",
        filename="endocrine_models/toxcast_ar_pathway_model_scores.xlsx",
        data_offset=46462809,
        compressed_size=66968,
    ),
    "toxcast_er_pathway": ZipMemberSpec(
        key="toxcast_er_pathway",
        filename="endocrine_models/toxcast_er_pathway_model_scores.xlsx",
        data_offset=46568363,
        compressed_size=72934,
    ),
}


def _normalize_casrn(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("-", "", regex=False).str.strip()


def _read_local_range(path: Path, start: int, size: int) -> bytes:
    with path.open("rb") as fh:
        fh.seek(start)
        payload = fh.read(size)
    if len(payload) != size:
        raise ValueError(f"Local archive {path} does not contain the full byte range {start}:{start + size}.")
    return payload


def _read_remote_range(url: str, start: int, size: int, retries: int = 4) -> bytes:
    session = requests.Session()
    session.trust_env = True
    headers = {
        "Range": f"bytes={start}-{start + size - 1}",
        "User-Agent": "EcoOOD/0.1",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(1024 * 256):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= size:
                    break
            payload = b"".join(chunks)
            if len(payload) < size:
                raise ValueError(
                    f"Expected {size} bytes from {url} at {start}, received {len(payload)} bytes."
                )
            return payload[:size]
        except Exception as exc:  # pragma: no cover - exercised on flaky network.
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download byte range {start}:{start + size} from {url}") from last_error


def extract_member_bytes(
    spec: ZipMemberSpec,
    local_archive: Path | None = None,
    remote_url: str | None = None,
) -> bytes:
    end_offset = spec.data_offset + spec.compressed_size
    if local_archive is not None and local_archive.exists() and local_archive.stat().st_size >= end_offset:
        raw = _read_local_range(local_archive, spec.data_offset, spec.compressed_size)
    elif remote_url:
        raw = _read_remote_range(remote_url, spec.data_offset, spec.compressed_size)
    else:
        raise FileNotFoundError(
            f"Unable to extract {spec.filename}: local archive is incomplete and no remote URL was provided."
        )
    if spec.compression == 0:
        return raw
    if spec.compression == 8:
        return zlib.decompress(raw, -15)
    raise ValueError(f"Unsupported compression method {spec.compression} for {spec.filename}")


def load_member_table(
    spec: ZipMemberSpec,
    local_archive: Path | None = None,
    remote_url: str | None = None,
) -> pd.DataFrame:
    payload = extract_member_bytes(spec, local_archive=local_archive, remote_url=remote_url)
    return pd.read_excel(io.BytesIO(payload))


def aggregate_cytotox_table(df: pd.DataFrame) -> pd.DataFrame:
    result = df.rename(
        columns={
            "dsstox_substance_id": "dtxsid",
            "casn": "casrn",
            "cytotox_median_um": "mech_cytotox_median_um",
            "cytotox_lower_bound_um": "mech_cytotox_lower_um",
            "cytotox_median_log": "mech_cytotox_median_log10_um",
            "cytotox_lower_bound_log": "mech_cytotox_lower_log10_um",
            "ntested": "mech_cytotox_ntested",
            "nhit": "mech_cytotox_nhit",
        }
    ).copy()
    numeric_cols = [
        "mech_cytotox_median_um",
        "mech_cytotox_lower_um",
        "mech_cytotox_median_log10_um",
        "mech_cytotox_lower_log10_um",
        "mech_cytotox_ntested",
        "mech_cytotox_nhit",
    ]
    for column in numeric_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["mech_cytotox_hit_rate"] = (
        result["mech_cytotox_nhit"] / result["mech_cytotox_ntested"].replace(0, np.nan)
    )
    return result[["dtxsid", "casrn", *numeric_cols, "mech_cytotox_hit_rate"]].drop_duplicates("dtxsid")


def aggregate_literature_table(df: pd.DataFrame) -> pd.DataFrame:
    working = df.rename(columns={"casrn": "casrn", "dtxsid": "dtxsid"}).copy()
    working["mode_key"] = working["literature_mode"].fillna("").astype(str).str.strip().str.lower().map(LITERATURE_MODE_MAP)
    score_key = working["literature_score"].fillna("").astype(str).str.strip().str.lower()
    working["score_value"] = score_key.map(LITERATURE_SCORE_MAP)
    working = working[working["mode_key"].notna()].copy()
    pivot = (
        working.groupby(["dtxsid", "mode_key"], dropna=True)["score_value"]
        .max()
        .unstack("mode_key")
        .reset_index()
    )
    casrn = working.groupby("dtxsid", dropna=True)["casrn"].agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else "")
    result = pivot.merge(casrn.rename("casrn"), on="dtxsid", how="left")
    return result


def aggregate_pathway_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = df.rename(
        columns={
            "casrn": "casrn",
            "dtxsid": "dtxsid",
            "auc_agonist": f"mech_{prefix}_auc_agonist",
            "auc_antagonist": f"mech_{prefix}_auc_antagonist",
        }
    ).copy()
    result[f"mech_{prefix}_auc_agonist"] = pd.to_numeric(result[f"mech_{prefix}_auc_agonist"], errors="coerce")
    result[f"mech_{prefix}_auc_antagonist"] = pd.to_numeric(result[f"mech_{prefix}_auc_antagonist"], errors="coerce")
    return result[["dtxsid", "casrn", f"mech_{prefix}_auc_agonist", f"mech_{prefix}_auc_antagonist"]].drop_duplicates("dtxsid")


def _max_abs(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    return float(np.abs(values).max()) if len(values) else np.nan


def _mean_abs_frame(frame: pd.DataFrame) -> float:
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    return float(np.abs(values).mean()) if len(values) else np.nan


def aggregate_h295r_table(df: pd.DataFrame) -> pd.DataFrame:
    working = df.rename(
        columns={
            "dsstox_substance_id": "dtxsid",
            "casn": "casrn",
            "BMD": "bmd",
            "mMd": "mmd",
            "maxmMd": "maxmmd",
            "criticalVal": "critical_val",
        }
    ).copy()
    numeric_cols = [*H295R_HORMONE_COLS, "bmd", "mmd", "maxmmd", "critical_val"]
    for column in numeric_cols:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    grouped = working.groupby("dtxsid", dropna=True)
    result = grouped["casrn"].agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else "").to_frame().reset_index()
    for hormone in H295R_HORMONE_COLS:
        result[f"mech_h295r_{hormone.lower()}_max_abs"] = grouped[hormone].apply(_max_abs).to_numpy()
    result["mech_h295r_mean_abs"] = grouped[H295R_HORMONE_COLS].apply(_mean_abs_frame).to_numpy()
    result["mech_h295r_bmd_min"] = grouped["bmd"].min().to_numpy()
    result["mech_h295r_mmd_max"] = grouped["mmd"].max().to_numpy()
    result["mech_h295r_maxmmd_max"] = grouped["maxmmd"].max().to_numpy()
    result["mech_h295r_critical_val"] = grouped["critical_val"].median().to_numpy()
    hormone_cols = [column for column in result.columns if column.startswith("mech_h295r_") and column.endswith("_max_abs")]
    result["mech_h295r_active_endpoint_count"] = result[hormone_cols].ge(1.0).sum(axis=1)
    return result


def build_mechanistic_features(
    local_archive: Path | None = None,
    remote_url: str = INVITRODB_V43_BLOB_URL,
) -> pd.DataFrame:
    cytotox = aggregate_cytotox_table(load_member_table(INVITRODB_V43_MEMBER_SPECS["cytotox"], local_archive, remote_url))
    literature = aggregate_literature_table(load_member_table(INVITRODB_V43_MEMBER_SPECS["ar_er_literature"], local_archive, remote_url))
    toxcast_ar = aggregate_pathway_table(
        load_member_table(INVITRODB_V43_MEMBER_SPECS["toxcast_ar_pathway"], local_archive, remote_url),
        prefix="toxcast_ar",
    )
    toxcast_er = aggregate_pathway_table(
        load_member_table(INVITRODB_V43_MEMBER_SPECS["toxcast_er_pathway"], local_archive, remote_url),
        prefix="toxcast_er",
    )
    h295r = aggregate_h295r_table(load_member_table(INVITRODB_V43_MEMBER_SPECS["ht_h295r"], local_archive, remote_url))

    merged = cytotox.merge(literature, on=["dtxsid", "casrn"], how="outer")
    merged = merged.merge(toxcast_ar, on=["dtxsid", "casrn"], how="outer")
    merged = merged.merge(toxcast_er, on=["dtxsid", "casrn"], how="outer")
    merged = merged.merge(h295r, on=["dtxsid", "casrn"], how="outer")

    mech_cols = [column for column in merged.columns if column.startswith("mech_")]
    merged["casrn"] = merged["casrn"].fillna("").astype(str)
    merged["casrn_norm"] = _normalize_casrn(merged["casrn"])
    merged["mech_feature_count"] = merged[mech_cols].notna().sum(axis=1)
    return merged


def load_or_build_mechanistic_features(
    cache_path: Path,
    local_archive: Path | None = None,
    remote_url: str = INVITRODB_V43_BLOB_URL,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = build_mechanistic_features(local_archive=local_archive, remote_url=remote_url)
    features.to_csv(cache_path, index=False)
    return features


def attach_mechanistic_features(
    dataset: pd.DataFrame,
    mechanism_features: pd.DataFrame | None,
    *,
    dtxsid_col: str = "dtxsid",
    casrn_col: str = "cas_number",
) -> pd.DataFrame:
    if mechanism_features is None or mechanism_features.empty:
        return dataset

    result = dataset.copy()
    mech = mechanism_features.copy()
    mech_cols = [column for column in mech.columns if column.startswith("mech_")]
    if not mech_cols:
        return result

    by_dtxsid = mech[mech["dtxsid"].fillna("").astype(str).str.strip() != ""].copy()
    by_dtxsid = by_dtxsid.drop_duplicates("dtxsid")
    result = result.merge(by_dtxsid[[dtxsid_col, *mech_cols]], on=dtxsid_col, how="left")

    if casrn_col not in result.columns or "casrn_norm" not in mech.columns:
        return result

    empty_mask = result[mech_cols].isna().all(axis=1)
    if not empty_mask.any():
        return result

    by_casrn = mech[mech["casrn_norm"].fillna("").astype(str).str.strip() != ""].copy()
    by_casrn = by_casrn.drop_duplicates("casrn_norm")
    cas_lookup = result.loc[empty_mask, [casrn_col]].copy()
    cas_lookup["casrn_norm"] = _normalize_casrn(cas_lookup[casrn_col])
    cas_lookup = cas_lookup.merge(by_casrn[["casrn_norm", *mech_cols]], on="casrn_norm", how="left")
    result.loc[empty_mask, mech_cols] = cas_lookup[mech_cols].to_numpy()
    return result
