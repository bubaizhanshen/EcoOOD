from __future__ import annotations

import pandas as pd
from zipfile import ZipFile

from ecoood.dsstox import parse_clowder_zip_listing, read_dsstox_catalog, resolve_chemical_index, resolve_chemical_index_from_sources


def test_parse_clowder_zip_listing_deduplicates_members() -> None:
    html = """
    DSSTox_CCD_dump_12092025/
    DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv
    DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv
    DSSTox_CCD_dump_12092025/DSSToxCCDdump1.csv
    """
    listing = parse_clowder_zip_listing(html)
    assert listing["path"].tolist() == [
        "DSSTox_CCD_dump_12092025/",
        "DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv",
        "DSSTox_CCD_dump_12092025/DSSToxCCDdump1.csv",
    ]
    assert listing["is_directory"].tolist() == [True, False, False]


def test_resolve_chemical_index_uses_dtxsid_then_casrn(tmp_path) -> None:
    catalog_path = tmp_path / "dsstox.csv"
    pd.DataFrame(
        [
            {
                "DTXSID": "DTXSID001",
                "CASRN": "50-00-0",
                "PREFERRED_NAME": "Chem A",
                "SMILES": "CCO",
                "INCHIKEY": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "AVERAGE_MASS": "46.07",
                "XLOGP": "-0.3",
            },
            {
                "DTXSID": "",
                "CASRN": "64-17-5",
                "PREFERRED_NAME": "Chem B",
                "SMILES": "CCCO",
                "INCHIKEY": "BDERNNFJNOPAEC-UHFFFAOYSA-N",
                "AVERAGE_MASS": "60.10",
                "XLOGP": "0.2",
            },
        ]
    ).to_csv(catalog_path, index=False)
    catalog = read_dsstox_catalog([catalog_path])
    chemicals = pd.DataFrame(
        [
            {"cas_number": "50-00-0", "chemical_name": "A", "dtxsid": "DTXSID001"},
            {"cas_number": "64-17-5", "chemical_name": "B", "dtxsid": ""},
        ]
    )
    resolved = resolve_chemical_index(chemicals, catalog)
    assert resolved.loc[0, "smiles"] == "CCO"
    assert resolved.loc[0, "resolution_source"] == "dsstox_dtxsid"
    assert resolved.loc[1, "smiles"] == "CCCO"
    assert resolved.loc[1, "resolution_source"] == "dsstox_casrn"


def test_read_dsstox_catalog_from_zip(tmp_path) -> None:
    csv_path = tmp_path / "dsstox.csv"
    pd.DataFrame(
        [
            {
                "DTXSID": "DTXSID001",
                "PREFERRED_NAME": "Chem A",
                "CASRN": "50-00-0",
                "SMILES": "CCO",
                "QSAR_READY_SMILES": "CCO",
                "INCHIKEY": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "AVERAGE_MASS": "46.07",
            }
        ]
    ).to_csv(csv_path, index=False)
    zip_path = tmp_path / "dsstox.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv")
    catalog = read_dsstox_catalog([zip_path])
    assert len(catalog) == 1
    assert catalog.loc[0, "dtxsid"] == "DTXSID001"
    assert catalog.loc[0, "smiles"] == "CCO"


def test_resolve_chemical_index_from_sources_filters_targets(tmp_path) -> None:
    csv_path = tmp_path / "dsstox.csv"
    pd.DataFrame(
        [
            {"DTXSID": "DTXSID001", "PREFERRED_NAME": "Chem A", "CASRN": "50-00-0", "SMILES": "CCO", "INCHIKEY": "K1", "AVERAGE_MASS": "46.07"},
            {"DTXSID": "DTXSID999", "PREFERRED_NAME": "Chem Z", "CASRN": "99-99-9", "SMILES": "CCCC", "INCHIKEY": "K9", "AVERAGE_MASS": "58.00"},
        ]
    ).to_csv(csv_path, index=False)
    chemicals = pd.DataFrame([{"cas_number": "50-00-0", "chemical_name": "A", "dtxsid": "DTXSID001"}])
    resolved = resolve_chemical_index_from_sources(chemicals, [csv_path], chunksize=1)
    assert len(resolved) == 1
    assert resolved.loc[0, "smiles"] == "CCO"
    assert resolved.loc[0, "resolution_source"] == "dsstox_dtxsid"


def test_resolve_chemical_index_matches_cas_without_hyphens(tmp_path) -> None:
    csv_path = tmp_path / "dsstox.csv"
    pd.DataFrame(
        [
            {"DTXSID": "", "PREFERRED_NAME": "Chem A", "CASRN": "50-00-0", "SMILES": "CCO", "INCHIKEY": "K1", "AVERAGE_MASS": "46.07"},
        ]
    ).to_csv(csv_path, index=False)
    chemicals = pd.DataFrame([{"cas_number": "50000", "chemical_name": "A", "dtxsid": ""}])
    resolved = resolve_chemical_index_from_sources(chemicals, [csv_path], chunksize=1)
    assert resolved.loc[0, "smiles"] == "CCO"
    assert resolved.loc[0, "resolution_source"] == "dsstox_casrn"
