from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "external_data" / "kinikles_2024_unsaturated_seismic_compression"

KINIKLES_DOI = "10.1016/j.compgeo.2024.106113"
KINIKLES_URL = "https://www.sciencedirect.com/science/article/pii/S0266352X24000491"
CHEN_DOI = "10.1016/j.compgeo.2024.106500"
CHEN_URL = "https://researchonline.jcu.edu.au/85642/"


def cyclic_test_matrix() -> pd.DataFrame:
    # Table 2 in Kinikles, Rong and McCartney (2024), read from the open-access article page.
    # These are test-condition metadata, not raw response time histories.
    rows = [
        ("set_1", 1.99, 0.560, "1.00:-1.00", "0.98:-1.00", "0.86:-0.95", False),
        ("set_2", 1.99, 0.558, "1.00:-1.00", "0.94:-0.82", "1.19:-0.69", False),
        ("set_1", 2.99, 0.400, "1.00:-1.00", "0.80:-0.89", "0.61:-1.22", False),
        ("set_2", 2.99, 0.400, "1.00:-1.00", "0.88:-0.81", "0.51:-1.16", False),
        ("set_1", 3.99, 0.300, "1.00:-1.00", "0.85:-0.94", "0.68:-1.04", False),
        ("set_2", 3.99, 0.300, "1.00:-1.00", "0.88:-0.81", "0.51:-1.16", False),
        ("set_1", 6.00, 0.206, "1.00:-1.00", "0.89:-1.08", "0.83:-1.03", True),
        ("set_2", 5.99, 0.206, "1.00:-1.00", "0.91:-0.72", "0.87:-0.82", False),
        ("set_1", 9.99, 0.118, "1.00:-1.00", "1.10:-0.81", "1.03:-0.94", False),
        ("set_2", 10.00, 0.117, "1.00:-1.00", "0.89:-1.04", "0.78:-1.18", False),
        ("set_1", 100.00, 0.000, "1.00:-1.00", "0.92:-0.94", "0.78:-1.06", True),
    ]
    out = []
    for i, row in enumerate(rows, start=1):
        dataset, suction, sat, target, initial, final, unreported = row
        out.append(
            {
                "source": "Kinikles et al. 2024 / Rong and McCartney cyclic simple-shear experiments",
                "source_url": KINIKLES_URL,
                "doi": KINIKLES_DOI,
                "license": "CC BY 4.0 open-access article; raw time histories are not redistributed here",
                "source_table": "Table 2",
                "test_id": f"KRM2024-{i:02d}",
                "dataset": dataset,
                "initial_matric_suction_kpa": suction,
                "initial_degree_saturation": sat,
                "target_shear_strain_range_pct": target,
                "initial_achieved_strain_range_pct": initial,
                "final_achieved_strain_range_pct": final,
                "cycles": 200,
                "previously_unreported_in_2020_source": unreported,
                "usable_as_raw_holdout": False,
                "reason_not_raw_holdout": "Open article gives test matrix and figures; complete tabulated hydro-mechanical time series are data-on-request.",
            }
        )
    return pd.DataFrame(out)


def comparison_matrix() -> pd.DataFrame:
    rows = [
        {
            "item": "Current package",
            "source_url": "",
            "direct_cyclic_experimental_validation": "No",
            "available_reproducible_files": "Yes: scripts, CSVs, figures and validation logs bundled",
            "primary_strength": "Auditable plausibility-envelope benchmark with verification, sensitivity checks and explicit claim boundaries",
            "primary_gap": "No blind direct cyclic simple-shear or RLT response fit to raw external time histories",
            "mrnb_role": "Sets the honest lower bound for validation/comparison score",
        },
        {
            "item": "Kinikles, Rong and McCartney 2024",
            "source_url": KINIKLES_URL,
            "direct_cyclic_experimental_validation": "Yes: undrained cyclic simple shear, 200 cycles",
            "available_reproducible_files": "Partial: article and figures open; raw time histories data-on-request",
            "primary_strength": "Direct cyclic hydro-mechanical validation across initial saturation states in the funicular regime",
            "primary_gap": "Article itself reports that some variables are underpredicted at larger cycle counts and higher saturation",
            "mrnb_role": "Recent competing validation target; the present benchmark must not claim equivalent direct cyclic validation",
        },
        {
            "item": "Chen et al. 2024 multi-surface model",
            "source_url": CHEN_URL,
            "direct_cyclic_experimental_validation": "Yes: repeated-load triaxial and cyclic simple-shear tests",
            "available_reproducible_files": "Partial: open article/PDF; data availability states data on request",
            "primary_strength": "State-of-art cyclic hardening model with direct cyclic comparisons and adaptive sub-stepping",
            "primary_gap": "High parameter count and no located public raw dataset/code package for independent rerun",
            "mrnb_role": "Recent novelty comparator; the present benchmark novelty must be reproducibility/claim-boundary, not first cyclic hardening law",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    expected_images = [
        RAW / "1-s2.0-S0266352X24000491-gr5_lrg.jpg",
        RAW / "1-s2.0-S0266352X24000491-gr12_lrg.jpg",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected_images if not path.exists()]

    test_matrix = cyclic_test_matrix()
    test_matrix.to_csv(DATA / "kinikles2024_cyclic_simple_shear_test_matrix.csv", index=False)

    comparison = comparison_matrix()
    comparison.to_csv(DATA / "recent_cyclic_model_comparison_gate.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "source": "Kinikles et al. 2024",
                "doi": KINIKLES_DOI,
                "cyclic_test_conditions_recorded": len(test_matrix),
                "saturation_min": test_matrix["initial_degree_saturation"].min(),
                "saturation_max": test_matrix["initial_degree_saturation"].max(),
                "suction_min_kpa": test_matrix["initial_matric_suction_kpa"].min(),
                "suction_max_kpa": test_matrix["initial_matric_suction_kpa"].max(),
                "cycles_per_test": 200,
                "target_shear_strain_range_pct": "+/-1",
                "downloaded_article_figures_present": len(missing) == 0,
                "missing_downloaded_figures": "; ".join(missing),
                "direct_cyclic_raw_holdout_available_in_package": False,
                "gate_result": "external direct-cyclic target identified, but not passed as blind validation",
                "allowed_claim": "The manuscript is benchmarked against a recent direct-cyclic validation target and acknowledges the remaining raw-data holdout gap.",
                "prohibited_claim": "Do not claim calibrated prediction of Kinikles/Rong/McCartney cyclic simple-shear time histories.",
            }
        ]
    )
    summary.to_csv(DATA / "kinikles2024_cyclic_gate_summary.csv", index=False)
    print(f"kinikles2024_cyclic_gate=ok rows={len(test_matrix)} missing_figures={len(missing)}")


if __name__ == "__main__":
    main()
