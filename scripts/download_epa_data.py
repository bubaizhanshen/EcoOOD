from __future__ import annotations

import argparse
from pathlib import Path

import requests


ECOTOX_ASCII_URL = "https://gaftp.epa.gov/ecotox/ecotox_ascii_03_12_2026.zip"
DSSTOX_METADATA_URL = "https://api.figshare.com/v2/articles/5588566"


def download(url: str, output: Path, chunk_size: int = 1024 * 1024) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official EPA data assets used by EcoOOD.")
    parser.add_argument("--ecotox-output", type=Path, default=Path("data/raw/ecotox_ascii_03_12_2026.zip"))
    parser.add_argument("--dsstox-metadata-output", type=Path, default=Path("data/raw/dsstox_figshare_article_5588566.json"))
    args = parser.parse_args()

    download(ECOTOX_ASCII_URL, args.ecotox_output)
    download(DSSTOX_METADATA_URL, args.dsstox_metadata_output)
    print(f"Downloaded ECOTOX archive to {args.ecotox_output}")
    print(f"Downloaded DSSTox figshare metadata to {args.dsstox_metadata_output}")


if __name__ == "__main__":
    main()
