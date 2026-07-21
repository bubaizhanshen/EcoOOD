from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ecoood.dsstox import normalize_casrn


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv"
PMRA_ALRV_URL = (
    "https://www.canada.ca/en/health-canada/services/consumer-product-safety/"
    "pesticides-pest-management/public/protecting-your-health-environment/programs-initiatives/"
    "water-monitoring-pesticides/aquatic-life-reference-values.html/1000?wbdisable=true"
)
DEFAULT_OUT_DIR = ROOT / "results" / "external_regulatory_prep"

EXTERNAL_ENDPOINTS = {
    "fish_96h_lc50": {
        "label_col": "Acute Fish ALRV (µɡ/L) Footnote 4",
        "duration_h": 96,
        "endpoint_label": "Fish acute ALRV",
        "taxon_label": "freshwater fish",
    },
    "daphnia_48h_ec50": {
        "label_col": "Acute invertebrate ALRV (µɡ/L) Footnote 6",
        "duration_h": 48,
        "endpoint_label": "Freshwater invertebrate acute ALRV",
        "taxon_label": "freshwater invertebrates",
    },
    "algae_72_96h_ec50": {
        "label_col": "Non-vascular plants ALRV (µɡ/L) Footnote 8",
        "duration_h": 96,
        "endpoint_label": "Non-vascular plant ALRV",
        "taxon_label": "nonvascular plants",
    },
}

INVALID_CAS_TOKENS = {"", "nr", "nan", "na", "none"}
REFERENCE_CODE_PATTERN = re.compile(
    r"\b(?:PRD|PRVD|PRDD|RVD|REV|PACR|ERC|REG|PSRD|SRD)\d{4}-\d{2}\b|\bEPA-HQ-[A-Z]+-\d{4}-\d{4}\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PMRA aquatic life reference value candidate set for external regulatory-adjacent "
            "validation, including candidate chemicals, a non-EPA subset, a reference-code catalog, "
            "and an endpoint extraction queue."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def write_status(out_dir: Path, message: str) -> None:
    text = message.rstrip()
    (out_dir / "run_status_pmra.txt").write_text(text + "\n")
    print(text, flush=True)


def _parse_numeric_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = series.fillna("").astype(str).str.strip()
    censored = text.str.contains("<", regex=False) | text.str.contains(">", regex=False)
    parsed = pd.to_numeric(text.str.replace(r"[^0-9.]+", "", regex=True), errors="coerce")
    parsed = parsed.mask(censored)
    return parsed, censored


def load_pmra_alrv_table() -> pd.DataFrame:
    frame = pd.read_html(PMRA_ALRV_URL)[0]
    frame = frame.rename(
        columns={
            "ALRV chemical name": "chemical_name",
            "ALRV year updated Footnote 1": "year_updated",
            "ALRV Reference Footnote 2": "alrv_reference",
            "Representative CAS Number Footnote 3": "representative_casrn",
            "Registered CAS Number(s) Footnote 3": "registered_casrns",
        }
    )

    frame["chemical_name"] = frame["chemical_name"].fillna("").astype(str).str.strip()
    frame["alrv_reference"] = frame["alrv_reference"].fillna("").astype(str).str.strip()
    frame["representative_casrn"] = frame["representative_casrn"].fillna("").astype(str).str.strip()
    frame["registered_casrns"] = frame["registered_casrns"].fillna("").astype(str).str.strip()
    frame["year_updated"] = pd.to_numeric(frame["year_updated"], errors="coerce").astype("Int64")

    for endpoint, spec in EXTERNAL_ENDPOINTS.items():
        values, censored = _parse_numeric_label(frame[spec["label_col"]])
        frame[f"{endpoint}_alrv_ugL"] = values
        frame[f"{endpoint}_censored"] = censored

    frame["has_valid_representative_casrn"] = ~(
        frame["representative_casrn"].astype(str).str.strip().str.lower().isin(INVALID_CAS_TOKENS)
    )
    frame["endpoint_count"] = frame[[f"{endpoint}_alrv_ugL" for endpoint in EXTERNAL_ENDPOINTS]].notna().sum(axis=1)
    reference_text = frame["alrv_reference"].astype(str)
    has_epa_benchmark = reference_text.str.contains("EPA Benchmark", case=False, na=False)
    has_epa_docket = reference_text.str.contains(r"\bEPA-HQ-", case=False, na=False)
    has_pmra_code = reference_text.str.contains(
        r"\b(?:PRD|PRVD|PRDD|RVD|REV|PACR|ERC|REG|PSRD|SRD)\d{4}-\d{2}\b",
        case=False,
        na=False,
        regex=True,
    )

    frame["source_family"] = "pmra_decision"
    frame.loc[has_epa_benchmark & has_pmra_code, "source_family"] = "mixed_pmra_and_epa"
    frame.loc[(has_epa_benchmark | has_epa_docket) & ~has_pmra_code, "source_family"] = "adopted_epa_source"
    frame["external_independence"] = frame["source_family"].eq("pmra_decision")
    frame["reference_codes"] = frame["alrv_reference"].apply(
        lambda text: "; ".join(REFERENCE_CODE_PATTERN.findall(text))
    )
    frame["reference_code_count"] = frame["reference_codes"].apply(
        lambda text: 0 if not text else len([part for part in text.split("; ") if part])
    )
    return frame


def load_training_overlap() -> set[str]:
    train = pd.read_csv(TRAIN_PATH, usecols=["casrn"])
    cas = train["casrn"].map(normalize_casrn)
    return {c for c in cas if c}


def build_reference_catalog(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, record in candidates.iterrows():
        codes = [code for code in str(record["reference_codes"]).split("; ") if code]
        if not codes:
            rows.append(
                {
                    "reference_code": "",
                    "source_family": record["source_family"],
                    "chemical_name": record["chemical_name"],
                    "representative_casrn": record["representative_casrn"],
                    "year_updated": record["year_updated"],
                    "alrv_reference_raw": record["alrv_reference"],
                    "external_independence": bool(record["external_independence"]),
                }
            )
            continue
        for code in codes:
            rows.append(
                {
                    "reference_code": code,
                    "source_family": record["source_family"],
                    "chemical_name": record["chemical_name"],
                    "representative_casrn": record["representative_casrn"],
                    "year_updated": record["year_updated"],
                    "alrv_reference_raw": record["alrv_reference"],
                    "external_independence": bool(record["external_independence"]),
                }
            )

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog

    summary = (
        catalog.groupby(["reference_code", "source_family", "alrv_reference_raw", "external_independence"], dropna=False)
        .agg(
            n_chemicals=("chemical_name", "nunique"),
            min_year=("year_updated", "min"),
            max_year=("year_updated", "max"),
            chemical_examples=("chemical_name", lambda vals: "; ".join(sorted(set(vals))[:8])),
        )
        .reset_index()
        .sort_values(["external_independence", "n_chemicals", "reference_code"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    summary["suggested_lookup_query"] = summary.apply(
        lambda row: (
            row["reference_code"]
            if str(row["reference_code"]).strip()
            else row["alrv_reference_raw"]
        ),
        axis=1,
    )
    return summary


def build_extraction_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, record in candidates.iterrows():
        for endpoint, spec in EXTERNAL_ENDPOINTS.items():
            value = record.get(f"{endpoint}_alrv_ugL")
            if pd.isna(value):
                continue

            if not bool(record.get("has_valid_representative_casrn", False)):
                priority = "low"
            elif bool(record.get("training_overlap_by_casrn", False)):
                priority = "low"
            elif bool(record.get("external_independence", False)) and int(record.get("endpoint_count", 0)) >= 2:
                priority = "high"
            elif bool(record.get("external_independence", False)):
                priority = "medium"
            else:
                priority = "low"

            rows.append(
                {
                    "chemical_name": record["chemical_name"],
                    "representative_casrn": record["representative_casrn"],
                    "registered_casrns": record["registered_casrns"],
                    "year_updated": record["year_updated"],
                    "endpoint": endpoint,
                    "endpoint_label": spec["endpoint_label"],
                    "taxon_label": spec["taxon_label"],
                    "duration_h": spec["duration_h"],
                    "alrv_value_ugL": value,
                    "endpoint_count": int(record.get("endpoint_count", 0)),
                    "has_valid_representative_casrn": bool(record.get("has_valid_representative_casrn", False)),
                    "training_overlap_by_casrn": bool(record.get("training_overlap_by_casrn", False)),
                    "source_family": record["source_family"],
                    "external_independence": bool(record.get("external_independence", False)),
                    "alrv_reference_raw": record["alrv_reference"],
                    "reference_codes": record["reference_codes"],
                    "reference_code_count": int(record.get("reference_code_count", 0)),
                    "suggested_source_type": (
                        "PMRA decision/re-evaluation document"
                        if bool(record.get("external_independence", False))
                        else "PMRA-adopted EPA benchmark"
                    ),
                    "extraction_priority": priority,
                    "selected_for_validation": (
                        bool(record.get("has_valid_representative_casrn", False))
                        and not bool(record.get("training_overlap_by_casrn", False))
                    ),
                }
            )
    queue = pd.DataFrame(rows)
    if queue.empty:
        return queue
    return queue.sort_values(
        ["extraction_priority", "endpoint_count", "chemical_name", "endpoint"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def build_external_candidate_chemicals(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select independent, multi-endpoint PMRA chemicals for ECHA row recovery."""
    selected = candidates.loc[
        candidates["external_independence"]
        & candidates["has_valid_representative_casrn"]
        & ~candidates["training_overlap_by_casrn"]
        & candidates["endpoint_count"].ge(2)
    ].copy()
    selected["min_year"] = selected["year_updated"]
    selected["max_year"] = selected["year_updated"]
    columns = [
        "chemical_name",
        "representative_casrn",
        "casrn_normalized",
        "reference_codes",
        "endpoint_count",
        "min_year",
        "max_year",
    ]
    return selected[columns].sort_values(
        ["endpoint_count", "chemical_name"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_status(out_dir, "Loading PMRA ALRV table.")
    candidates = load_pmra_alrv_table()

    write_status(out_dir, "Checking representative CAS overlap against training benchmark.")
    training_cas = load_training_overlap()
    candidates["casrn_normalized"] = candidates["representative_casrn"].map(normalize_casrn)
    candidates["training_overlap_by_casrn"] = candidates["casrn_normalized"].isin(training_cas)

    candidate_path = out_dir / "pmra_alrv_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    non_epa = candidates[candidates["external_independence"]].copy().reset_index(drop=True)
    non_epa_path = out_dir / "pmra_alrv_non_epa_candidates.csv"
    non_epa.to_csv(non_epa_path, index=False)

    reference_catalog = build_reference_catalog(candidates)
    reference_catalog_path = out_dir / "pmra_reference_catalog.csv"
    reference_catalog.to_csv(reference_catalog_path, index=False)

    extraction_queue = build_extraction_queue(candidates)
    extraction_queue_path = out_dir / "pmra_endpoint_extraction_queue.csv"
    extraction_queue.to_csv(extraction_queue_path, index=False)

    external_candidates = build_external_candidate_chemicals(candidates)
    external_candidate_path = out_dir / "pmra_external_candidate_chemicals.csv"
    external_candidates.to_csv(external_candidate_path, index=False)

    summary = {
        "pmra_alrv_url": PMRA_ALRV_URL,
        "n_candidate_chemicals": int(len(candidates)),
        "n_non_epa_candidates": int(non_epa["chemical_name"].nunique()),
        "n_candidates_with_valid_representative_casrn": int(candidates["has_valid_representative_casrn"].sum()),
        "n_non_epa_candidates_without_training_overlap": int(
            non_epa.loc[~non_epa["training_overlap_by_casrn"], "chemical_name"].nunique()
        ),
        "n_extraction_rows": int(len(extraction_queue)),
        "n_external_candidate_chemicals": int(len(external_candidates)),
        "n_reference_codes": int(
            reference_catalog.loc[reference_catalog["reference_code"].astype(str).str.strip().ne(""), "reference_code"].nunique()
        ),
        "n_adopted_epa_source_rows": int(candidates["source_family"].eq("adopted_epa_source").sum()),
        "n_mixed_pmra_and_epa_rows": int(candidates["source_family"].eq("mixed_pmra_and_epa").sum()),
        "n_pmra_decision_rows": int(candidates["source_family"].eq("pmra_decision").sum()),
    }
    (out_dir / "pmra_alrv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        f"PMRA ALRV URL: {PMRA_ALRV_URL}",
        f"Candidate chemicals: {summary['n_candidate_chemicals']}",
        f"Non-EPA PMRA candidates: {summary['n_non_epa_candidates']}",
        f"Candidates with valid representative CASRN: {summary['n_candidates_with_valid_representative_casrn']}",
        f"Non-EPA candidates without training overlap: {summary['n_non_epa_candidates_without_training_overlap']}",
        f"Endpoint extraction rows: {summary['n_extraction_rows']}",
        f"Independent multi-endpoint external candidates: {summary['n_external_candidate_chemicals']}",
        f"Distinct reference codes: {summary['n_reference_codes']}",
        f"Rows marked as PMRA-adopted EPA source: {summary['n_adopted_epa_source_rows']}",
        f"Rows marked as mixed PMRA + EPA source: {summary['n_mixed_pmra_and_epa_rows']}",
        f"Rows marked as PMRA decision-backed: {summary['n_pmra_decision_rows']}",
        f"Candidate table: {candidate_path}",
        f"Non-EPA candidate table: {non_epa_path}",
        f"Reference catalog: {reference_catalog_path}",
        f"Extraction queue: {extraction_queue_path}",
        f"External candidate table: {external_candidate_path}",
    ]
    (out_dir / "pmra_alrv_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_status(out_dir, "Completed PMRA ALRV candidate set build.")


if __name__ == "__main__":
    main()
