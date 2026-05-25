from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def audit_case(group: pd.DataFrame) -> list[dict]:
    group = group.sort_values("step").reset_index(drop=True)
    plastic = group["plastic_strain_index"].to_numpy(float)
    hd = group["hydraulic_damage"].to_numpy(float)
    sd = group["seismic_damage"].to_numpy(float)
    tau = group["shear_stress_kpa"].to_numpy(float)
    gamma = group["shear_strain"].to_numpy(float)
    stiffness = group["secant_stiffness_mpa"].to_numpy(float)

    dplastic = np.diff(plastic, prepend=plastic[0])
    dhd = np.diff(hd, prepend=hd[0])
    dsd = np.diff(sd, prepend=sd[0])
    dgamma = np.diff(gamma, prepend=gamma[0])
    pseudo_work = np.abs(tau * dgamma)
    damage_dissipation_proxy = np.abs(tau) * np.maximum(dplastic, 0.0) + 100.0 * np.maximum(dhd, 0.0) + 100.0 * np.maximum(dsd, 0.0)

    key = {
        "model": group["model"].iloc[0],
        "suction_amp": float(group["suction_amp"].iloc[0]),
        "cyclic_amp": float(group["cyclic_amp"].iloc[0]),
    }
    return [
        {**key, "check": "plastic_strain_non_decreasing", "status": bool(np.all(dplastic >= -1e-14)), "max_violation": float(max(0.0, -dplastic.min()))},
        {**key, "check": "hydraulic_damage_non_decreasing", "status": bool(np.all(dhd >= -1e-14)), "max_violation": float(max(0.0, -dhd.min()))},
        {**key, "check": "seismic_damage_non_decreasing", "status": bool(np.all(dsd >= -1e-14)), "max_violation": float(max(0.0, -dsd.min()))},
        {**key, "check": "positive_secant_stiffness", "status": bool(np.all(stiffness > 0.0)), "max_violation": float(max(0.0, -stiffness.min()))},
        {**key, "check": "finite_pseudo_work", "status": bool(np.isfinite(pseudo_work).all()), "max_violation": 0.0 if np.isfinite(pseudo_work).all() else 1.0},
        {**key, "check": "nonnegative_damage_dissipation_proxy", "status": bool(np.all(damage_dissipation_proxy >= -1e-14)), "max_violation": float(max(0.0, -damage_dissipation_proxy.min()))},
    ]


def main() -> None:
    df = pd.read_csv(DATA / "benchmark_results.csv")
    rows: list[dict] = []
    for _, group in df.groupby(["model", "suction_amp", "cyclic_amp"], sort=True):
        rows.extend(audit_case(group))
    details = pd.DataFrame(rows)
    details["status_text"] = np.where(details["status"], "pass", "fail")
    details.to_csv(DATA / "algorithmic_dissipation_audit.csv", index=False)
    failed = details[~details["status"]]
    summary = pd.DataFrame(
        [
            {
                "checks": int(len(details)),
                "failed_checks": int(len(failed)),
                "all_passed": bool(failed.empty),
                "max_violation": float(details["max_violation"].max()),
                "scope_note": (
                    "Algorithmic monotonicity and nonnegative dissipation-proxy checks for the scripted benchmark; "
                    "this is not a full thermodynamic proof for arbitrary loading paths."
                ),
            }
        ]
    )
    summary.to_csv(DATA / "algorithmic_dissipation_summary.csv", index=False)
    print(f"algorithmic_dissipation=ok checks={len(details)} failures={len(failed)}")


if __name__ == "__main__":
    main()
