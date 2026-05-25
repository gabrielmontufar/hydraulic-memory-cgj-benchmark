from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "external_data" / "chen_2024_multisurface_cyclic_hardening" / "1-s2.0-S0266352X24004361-mmc1.docx"
SOURCE_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0266352X24004361-mmc1.docx"
ARTICLE_URL = "https://researchonline.jcu.edu.au/85642/"
DOI = "10.1016/j.compgeo.2024.106500"

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def cell_text(tc: ET.Element) -> str:
    pieces = []
    for node in tc.findall(".//w:t", NS):
        if node.text:
            pieces.append(node.text)
    return " ".join(" ".join(pieces).split())


def extract_tables(path: Path) -> list[list[list[str]]]:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    tables: list[list[list[str]]] = []
    for tbl in root.findall(".//w:tbl", NS):
        rows: list[list[str]] = []
        for tr in tbl.findall("./w:tr", NS):
            rows.append([cell_text(tc) for tc in tr.findall("./w:tc", NS)])
        tables.append(rows)
    return tables


def geotechnical_properties(table: list[list[str]]) -> pd.DataFrame:
    rows = []
    for row in table[1:]:
        if len(row) >= 2 and row[0]:
            rows.append(
                {
                    "source": "Chen et al. 2024 supplementary material",
                    "source_url": SOURCE_URL,
                    "doi": DOI,
                    "soil": "SPARC sand",
                    "property": row[0],
                    "value": row[1],
                    "standard": row[2] if len(row) > 2 else "",
                }
            )
    return pd.DataFrame(rows)


def sparc_monotonic_conditions(table: list[list[str]]) -> pd.DataFrame:
    header_idx = None
    for idx, row in enumerate(table):
        if row and row[0] == "Test case":
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("SPARC monotonic test-condition header not found")

    out = []
    for row in table[header_idx + 1 :]:
        if len(row) < 6 or not row[0]:
            continue
        out.append(
            {
                "source": "Chen et al. 2024 supplementary material",
                "source_url": SOURCE_URL,
                "doi": DOI,
                "soil": "SPARC sand",
                "test_case": row[0],
                "initial_degree_saturation_pct": float(row[1]),
                "gravimetric_water_content_pct": float(row[2]),
                "initial_void_ratio": float(row[3]),
                "initial_suction_kpa": float(row[4]),
                "confining_pressure_kpa": float(row[5]),
                "raw_response_time_history_in_supplement": False,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing Chen 2024 supplementary file: {RAW}")
    DATA.mkdir(exist_ok=True)
    tables = extract_tables(RAW)
    inventory = pd.DataFrame(
        [
            {
                "source": "Chen et al. 2024 supplementary material",
                "source_url": SOURCE_URL,
                "article_url": ARTICLE_URL,
                "doi": DOI,
                "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
                "tables_extracted": len(tables),
                "table_index": idx,
                "rows": len(tbl),
                "columns_max": max((len(row) for row in tbl), default=0),
                "first_row": " | ".join(tbl[0]) if tbl else "",
                "claim_boundary": "supplementary calibration audit only; not public raw cyclic validation data",
            }
            for idx, tbl in enumerate(tables)
        ]
    )
    inventory.to_csv(DATA / "chen2024_supplementary_inventory.csv", index=False)

    props = geotechnical_properties(tables[0])
    props.to_csv(DATA / "chen2024_sparc_geotechnical_properties.csv", index=False)

    sparc = sparc_monotonic_conditions(tables[2])
    sparc.to_csv(DATA / "chen2024_sparc_monotonic_test_conditions.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "source": "Chen et al. 2024",
                "doi": DOI,
                "supplementary_docx_present": True,
                "supplementary_tables_extracted": len(tables),
                "sparc_properties_rows": len(props),
                "sparc_monotonic_condition_rows": len(sparc),
                "suction_min_kpa": sparc["initial_suction_kpa"].min(),
                "suction_max_kpa": sparc["initial_suction_kpa"].max(),
                "saturation_min_pct": sparc["initial_degree_saturation_pct"].min(),
                "saturation_max_pct": sparc["initial_degree_saturation_pct"].max(),
                "raw_cyclic_time_histories_found_in_supplement": False,
                "mrnb_interpretation": "recent calibrated multi-surface comparator documented; raw cyclic validation data still not public in the supplementary file",
            }
        ]
    )
    summary.to_csv(DATA / "chen2024_supplementary_calibration_summary.csv", index=False)
    print(f"chen2024_supplementary_audit=ok tables={len(tables)} sparc_cases={len(sparc)}")


if __name__ == "__main__":
    main()
