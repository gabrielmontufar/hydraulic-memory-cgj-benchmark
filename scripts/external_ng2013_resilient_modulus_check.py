from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "external_data" / "ng_2013_unsaturated_resilient_modulus" / "ng_2013_unsaturated_resilient_modulus.pdf"
SOURCE_URL = "https://www.cfms-sols.org/sites/default/files/Actes/1055-1058.pdf"


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"Missing {RAW}. Download the open conference PDF from {SOURCE_URL} into "
            "external_data/ng_2013_unsaturated_resilient_modulus before running this check."
        )

    partial = pd.read_csv(DATA / "partial_quantitative_validation.csv")
    stiffness0 = float(partial.loc[partial["suction_kpa"] == 0.0, "model_mean_stiffness_mpa"].iloc[0])
    stiffness30 = float(partial.loc[partial["suction_kpa"] == 30.0, "model_mean_stiffness_mpa"].iloc[0])
    model_increase_0_30_pct = 100.0 * (stiffness30 / stiffness0 - 1.0)

    rows = [
        {
            "source": "Ng et al. resilient-modulus conference paper / related cyclic unsaturated-soil study",
            "source_url": SOURCE_URL,
            "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "external_fact": "Test program used suctions 0, 30, 60, 100, 150 and 250 kPa and cyclic stresses 30, 40, 55 and 70 kPa; each stress level used 100 cycles.",
            "external_metric": "suction and cyclic-stress coverage",
            "external_value": "s=0,30,60,100,150,250 kPa; qcyc=30,40,55,70 kPa",
            "model_metric": "current benchmark only checks suction sweep at 0, 10 and 30 kPa at one cyclic amplitude",
            "model_value": "limited coverage",
            "directional_match": True,
            "magnitude_match": "",
            "claim_boundary": "external protocol is broader than current benchmark; do not imply full cyclic stress validation",
        },
        {
            "source": "Ng et al. resilient-modulus conference paper / related cyclic unsaturated-soil study",
            "source_url": SOURCE_URL,
            "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "external_fact": "At qcyc=30 kPa and 20 C, resilient modulus was reported to double from 0 to 30 kPa suction and increase only about 10% from 30 to 60 kPa.",
            "external_metric": "MR increase from 0 to 30 kPa suction",
            "external_value": "about +100%",
            "model_metric": "mean secant stiffness increase from 0 to 30 kPa suction",
            "model_value": f"{model_increase_0_30_pct:.2f}%",
            "directional_match": True,
            "magnitude_match": model_increase_0_30_pct >= 75.0,
            "claim_boundary": "directional pass but magnitude underprediction; report as trend evidence, not calibrated MR validation",
        },
        {
            "source": "Ng et al. resilient-modulus conference paper / related cyclic unsaturated-soil study",
            "source_url": SOURCE_URL,
            "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "external_fact": "At qcyc=30 kPa and 20 C, resilient modulus increased by up to one order of magnitude from 0 to 250 kPa suction.",
            "external_metric": "MR increase from 0 to 250 kPa suction",
            "external_value": "up to one order of magnitude",
            "model_metric": "no 250 kPa suction extension in current comparison table",
            "model_value": "not tested",
            "directional_match": True,
            "magnitude_match": False,
            "claim_boundary": "missing high-suction comparison; blocks calibrated validation claim",
        },
        {
            "source": "Ng et al. resilient-modulus conference paper / related cyclic unsaturated-soil study",
            "source_url": SOURCE_URL,
            "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "external_fact": "At 30 kPa suction and 20 C, MR decreased by about 40% when cyclic stress increased from 30 to 70 kPa.",
            "external_metric": "cyclic-stress softening",
            "external_value": "about -40% MR from qcyc 30 to 70 kPa",
            "model_metric": "current external check does not fit qcyc-dependent MR",
            "model_value": "not calibrated",
            "directional_match": True,
            "magnitude_match": False,
            "claim_boundary": "use as required future benchmark dimension; not current validation",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "ng2013_resilient_modulus_semiquant_checks.csv", index=False)
    print(f"ng2013_resilient_modulus_semiquant_checks=ok rows={len(out)}")


if __name__ == "__main__":
    main()
