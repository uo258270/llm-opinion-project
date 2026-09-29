"""Download and load the CIS Study 3194 microdata (Redes Sociales (I) / Religión (III), ISSP).

Source: https://www.cis.es/es/estudios/redes-sociales-i-/-religion-iii-issp-?origen=estudio&idEstudio=14366
The microdata ZIP contains an SPSS file (3194.sav) with variable and value labels,
plus the questionnaire, codebook, and technical sheet as PDFs.

Usage:
    python src/load_cis3194.py            # download (if needed), extract, load, print summary
    python src/load_cis3194.py --force    # re-download even if the ZIP already exists

From other code:
    from load_cis3194 import load_cis3194
    df, meta = load_cis3194()
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd
import pyreadstat
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cis3194"

ZIP_URL = (
    "https://www.cis.es/documents/20117/1556601/MD3194.zip/"
    "11318565-cd48-913b-9667-092d691a2644?version=1.0&t=1696371436383"
)
ZIP_PATH = RAW_DIR / "MD3194.zip"
SAV_PATH = RAW_DIR / "3194.sav"


def download_cis3194(force: bool = False) -> Path:
    """Download the microdata ZIP (if not already present) and extract it. Returns the .sav path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if force or not ZIP_PATH.exists():
        print(f"Downloading {ZIP_URL}")
        resp = requests.get(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):
            raise RuntimeError(
                "Response is not a ZIP file; the CIS download URL may have changed. "
                "Download 'Fichero datos' manually from the study page into "
                f"{RAW_DIR}"
            )
        ZIP_PATH.write_bytes(resp.content)
        print(f"Saved {ZIP_PATH} ({len(resp.content) / 1e6:.2f} MB)")

    if force or not SAV_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
        print(f"Extracted {ZIP_PATH.name} -> {RAW_DIR}")

    return SAV_PATH


def load_cis3194(force_download: bool = False) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
    """Load the microdata as numeric codes, plus metadata with variable/value labels.

    Codes are kept numeric (not converted to label strings) so that non-substantive
    answers (e.g. 8 = N.S., 9 = N.C.) can be filtered explicitly later.
    Use meta.variable_value_labels[var] to map codes to labels.
    """
    sav_path = download_cis3194(force=force_download)
    df, meta = pyreadstat.read_sav(str(sav_path), encoding="latin1" if _needs_latin1(sav_path) else None)
    return df, meta


def _needs_latin1(sav_path: Path) -> bool:
    """CIS .sav files are sometimes Latin-1 encoded without declaring it; detect decoding failures."""
    try:
        pyreadstat.read_sav(str(sav_path), metadataonly=True)
        return False
    except Exception:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download and re-extract the ZIP")
    args = parser.parse_args()

    df, meta = load_cis3194(force_download=args.force)
    print(f"\nLoaded {df.shape[0]} respondents x {df.shape[1]} variables")
    print(f"File label: {meta.file_label}")
    print("\nFirst 40 variables:")
    for name in meta.column_names[:40]:
        print(f"  {name:<12} {meta.column_names_to_labels.get(name) or ''}")


if __name__ == "__main__":
    main()
