from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from run_unsat_cyclic_benchmark import simulate_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_URL = "https://www.cfms-sols.org/sites/default/files/Actes/1055-1058.pdf"


def mean_stiffness(suction_kpa: float, cyclic_amp: float) -> float:
    out = simulate_case(
        "full_hysteretic_damage",
        suction_amp=0.0,
        cyclic_amp=cyclic_amp,
        n_steps=720,
        suction_mean=suction_kpa,
    )
    return float(out["secant_stiffness_mpa"].mean())


def suction_correction_parameters(raw_r30: float, raw_r60: float) -> tuple[float, float, float, float]:
    """Fit F(s)=1+A*(1-exp(-s/B)) to two approximate Ng et al. ratios."""
    target_r30 = 2.0
    target_r60 = 2.2
    correction_30 = target_r30 / raw_r30
    correction_60 = target_r60 / raw_r60
    x = (correction_60 - 1.0) / (correction_30 - 1.0) - 1.0
    x = min(max(x, 1e-6), 0.999999)
    b = -30.0 / math.log(x)
    a = (correction_30 - 1.0) / (1.0 - x)
    return a, b, correction_30, correction_60


def suction_factor(suction_kpa: float, a: float, b: float) -> float:
    return 1.0 + a * (1.0 - math.exp(-suction_kpa / b))


def stress_exponent(raw_q70_over_q30: float) -> float:
    """Fit H(q)=(q/30)^-b to Ng et al. approximate 40% MR drop from q=30 to 70 kPa."""
    target_q70_over_q30 = 0.60
    needed_factor = target_q70_over_q30 / raw_q70_over_q30
    return -math.log(needed_factor) / math.log(70.0 / 30.0)


def stress_factor(q_kpa: float, exponent: float) -> float:
    return (q_kpa / 30.0) ** (-exponent)


def main() -> None:
    DATA.mkdir(exist_ok=True)

    suction_levels = [0.0, 30.0, 60.0, 250.0]
    raw = {s: mean_stiffness(s, 0.30) for s in suction_levels}
    raw_r30 = raw[30.0] / raw[0.0]
    raw_r60 = raw[60.0] / raw[0.0]
    raw_r250 = raw[250.0] / raw[0.0]
    a, b, correction_30, correction_60 = suction_correction_parameters(raw_r30, raw_r60)

    suction_rows = []
    for suction in suction_levels:
        factor = suction_factor(suction, a, b)
        calibrated = raw[suction] * factor
        suction_rows.append(
            {
                "check_family": "suction semi-calibration",
                "source": "Ng et al. resilient-modulus evidence",
                "source_url": SOURCE_URL,
                "suction_kpa": suction,
                "cyclic_stress_kpa": 30.0,
                "raw_model_stiffness_mpa": raw[suction],
                "raw_model_ratio_vs_0kpa": raw[suction] / raw[0.0],
                "suction_correction_factor": factor,
                "semicalibrated_stiffness_mpa": calibrated,
                "semicalibrated_ratio_vs_0kpa": calibrated / raw[0.0],
                "external_target": {
                    0.0: "baseline",
                    30.0: "about 2x relative to 0 kPa",
                    60.0: "about 2.2x relative to 0 kPa, from 2x at 30 kPa plus about 10%",
                    250.0: "up to one order of magnitude relative to 0 kPa",
                }[suction],
                "used_for_parameter_fit": suction in {30.0, 60.0},
                "holdout_status": "not_holdout" if suction in {30.0, 60.0} else ("fails_order_of_magnitude_holdout" if suction == 250.0 else "baseline"),
            }
        )

    raw_q30 = mean_stiffness(30.0, 0.30)
    raw_q70 = mean_stiffness(30.0, 0.70)
    raw_q_ratio = raw_q70 / raw_q30
    q_exponent = stress_exponent(raw_q_ratio)
    stress_rows = []
    for q, raw_val in [(30.0, raw_q30), (70.0, raw_q70)]:
        factor = stress_factor(q, q_exponent)
        stress_rows.append(
            {
                "check_family": "cyclic stress semi-calibration",
                "source": "Ng et al. resilient-modulus evidence",
                "source_url": SOURCE_URL,
                "suction_kpa": 30.0,
                "cyclic_stress_kpa": q,
                "raw_model_stiffness_mpa": raw_val,
                "raw_model_ratio_vs_q30": raw_val / raw_q30,
                "stress_correction_factor": factor,
                "semicalibrated_stiffness_mpa": raw_val * factor,
                "semicalibrated_ratio_vs_q30": (raw_val * factor) / raw_q30,
                "external_target": "baseline" if q == 30.0 else "about 0.60 relative to qcyc=30 kPa",
                "used_for_parameter_fit": q == 70.0,
                "holdout_status": "not_holdout",
            }
        )

    pd.DataFrame(suction_rows).to_csv(DATA / "ng2013_suction_semicalibrated_envelope.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(DATA / "ng2013_cyclic_stress_semicalibrated_envelope.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "source": "Ng et al. resilient-modulus evidence",
                "source_url": SOURCE_URL,
                "calibration_type": "secondary semi-quantitative calibration from published approximate ratios",
                "suction_factor_form": "F(s)=1+A*(1-exp(-s/B))",
                "suction_factor_A": a,
                "suction_factor_B_kpa": b,
                "raw_ratio_0_to_30": raw_r30,
                "semicalibrated_ratio_0_to_30": suction_rows[1]["semicalibrated_ratio_vs_0kpa"],
                "raw_ratio_0_to_60": raw_r60,
                "semicalibrated_ratio_0_to_60": suction_rows[2]["semicalibrated_ratio_vs_0kpa"],
                "raw_ratio_0_to_250": raw_r250,
                "semicalibrated_ratio_0_to_250": suction_rows[3]["semicalibrated_ratio_vs_0kpa"],
                "external_250kpa_target": "up to one order of magnitude",
                "stress_factor_form": "H(q)=(q/30)^-b",
                "stress_exponent_b": q_exponent,
                "raw_q70_over_q30": raw_q_ratio,
                "semicalibrated_q70_over_q30": stress_rows[1]["semicalibrated_ratio_vs_q30"],
                "claim_boundary": "calibration-path demonstration only; approximate published ratios are not raw-data blind validation",
            }
        ]
    )
    summary.to_csv(DATA / "ng2013_semicalibrated_envelope_summary.csv", index=False)
    print("ng2013_semicalibrated_envelope=ok rows=7")


if __name__ == "__main__":
    main()
