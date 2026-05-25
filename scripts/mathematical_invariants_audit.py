from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def load_params() -> dict[str, float]:
    params = pd.read_csv(DATA / "model_parameters.csv")
    out: dict[str, float] = {}
    for _, row in params.iterrows():
        try:
            out[str(row["parameter"])] = float(row["value"])
        except ValueError:
            continue
    return out


def row(check: str, status: str, max_violation: float, detail: str) -> dict[str, object]:
    return {
        "check": check,
        "status": status,
        "max_violation": max_violation,
        "detail": detail,
    }


def main() -> None:
    tol = 1e-9
    params = load_params()
    df = pd.read_csv(DATA / "benchmark_results.csv")
    hmai_path = DATA / "hydraulic_memory_amplification_index.csv"
    checks: list[dict[str, object]] = []

    numeric_cols = [
        "suction_kpa",
        "degree_saturation",
        "bishop_chi",
        "mean_effective_stress_kpa",
        "preconsolidation_kpa",
        "yield_strength_kpa",
        "overstress",
        "hydraulic_damage",
        "seismic_damage",
        "shear_strain",
        "shear_stress_kpa",
        "secant_stiffness_mpa",
        "plastic_strain_index",
    ]
    finite_mask = np.isfinite(df[numeric_cols].to_numpy(dtype=float))
    checks.append(
        row(
            "finite_numeric_state",
            "pass" if finite_mask.all() else "fail",
            0.0 if finite_mask.all() else 1.0,
            "All core state columns must be finite.",
        )
    )

    sr_res = params["sr_res"]
    sr_sat = params["sr_sat"]
    sat_low = float(max(0.0, sr_res - df["degree_saturation"].min()))
    sat_high = float(max(0.0, df["degree_saturation"].max() - sr_sat))
    checks.append(
        row(
            "saturation_bounds",
            "pass" if max(sat_low, sat_high) <= tol else "fail",
            max(sat_low, sat_high),
            f"Degree of saturation must remain in [{sr_res}, {sr_sat}].",
        )
    )

    chi_low = float(max(0.0, -df["bishop_chi"].min()))
    chi_high = float(max(0.0, df["bishop_chi"].max() - 1.0))
    checks.append(
        row(
            "bishop_chi_bounds",
            "pass" if max(chi_low, chi_high) <= tol else "fail",
            max(chi_low, chi_high),
            "Bishop chi is clipped to the closed interval [0, 1].",
        )
    )

    p0 = params["p0"]
    p_eff_expected = p0 + df["bishop_chi"] * df["suction_kpa"]
    p_eff_residual = float((df["mean_effective_stress_kpa"] - p_eff_expected).abs().max())
    checks.append(
        row(
            "effective_stress_identity",
            "pass" if p_eff_residual <= 1e-8 else "fail",
            p_eff_residual,
            "p_eff must equal p0 + chi*suction for the scalar benchmark.",
        )
    )

    min_pc = float(df["preconsolidation_kpa"].min())
    min_g = float(df["secant_stiffness_mpa"].min())
    min_y = float(df["yield_strength_kpa"].min())
    checks.append(
        row(
            "positive_stress_and_stiffness",
            "pass" if min(min_pc, min_g, min_y) > 0.0 else "fail",
            float(max(0.0, -min(min_pc, min_g, min_y))),
            "Preconsolidation pressure, yield strength and secant stiffness must remain positive.",
        )
    )

    dh_low = float(max(0.0, -df["hydraulic_damage"].min()))
    dh_high = float(max(0.0, df["hydraulic_damage"].max() - params["beta_h"]))
    ds_low = float(max(0.0, -df["seismic_damage"].min()))
    ds_high = float(max(0.0, df["seismic_damage"].max() - params["beta_c"]))
    checks.append(
        row(
            "damage_bounds",
            "pass" if max(dh_low, dh_high, ds_low, ds_high) <= tol else "fail",
            max(dh_low, dh_high, ds_low, ds_high),
            "Damage variables must remain non-negative and below their declared beta bounds.",
        )
    )

    min_plastic_diff = 0.0
    for _, group in df.groupby(["model", "suction_amp", "cyclic_amp"], sort=False):
        diffs = group["plastic_strain_index"].diff().fillna(0.0)
        min_plastic_diff = min(min_plastic_diff, float(diffs.min()))
    checks.append(
        row(
            "plastic_strain_monotonicity",
            "pass" if min_plastic_diff >= -tol else "fail",
            float(max(0.0, -min_plastic_diff)),
            "Plastic-strain index must be non-decreasing within every simulated case.",
        )
    )

    cycles_s = 18.0
    cyclic_ratio = df["cyclic_amp"] * np.sin(2.0 * math.pi * cycles_s * df["time"])
    demand = np.abs(cyclic_ratio) * p0
    tau_expected = df["secant_stiffness_mpa"] * 1000.0 * df["shear_strain"] + np.sign(df["shear_strain"]) * demand * 0.0015
    tau_residual = float((df["shear_stress_kpa"] - tau_expected).abs().max())
    checks.append(
        row(
            "shear_stress_unit_consistency",
            "pass" if tau_residual <= 1e-8 else "fail",
            tau_residual,
            "Shear stress is computed in kPa as G(MPa)*1000*gamma plus the small demand marker term.",
        )
    )

    alpha_s_max = max(
        float((params["alpha_dry"] * df["suction_kpa"]).abs().max()),
        float((params["alpha_wet"] * df["suction_kpa"]).abs().max()),
    )
    suction_ratio_max = float((df["suction_kpa"] / params["s0"]).abs().max())
    checks.append(
        row(
            "dimensionless_groups_defined",
            "pass",
            0.0,
            f"alpha*s max={alpha_s_max:.6g}; s/s0 max={suction_ratio_max:.6g}; p_eff/p0 min={(df['mean_effective_stress_kpa']/p0).min():.6g}.",
        )
    )

    if hmai_path.exists():
        hmai = pd.read_csv(hmai_path)
        hmai_col = "hydraulic_memory_amplification_index"
        low = float(max(0.0, -hmai[hmai_col].min()))
        high = float(max(0.0, hmai[hmai_col].max() - 1.0))
        checks.append(
            row(
                "hmai_bounds",
                "pass" if max(low, high) <= tol else "fail",
                max(low, high),
                "HMAI must remain a bounded dimensionless contrast in [0, 1].",
            )
        )

    details = pd.DataFrame(checks)
    details.to_csv(DATA / "mathematical_invariants_audit.csv", index=False)
    failed = details[details["status"] != "pass"]
    summary = pd.DataFrame(
        [
            {
                "checks": len(details),
                "failed_checks": len(failed),
                "all_passed": len(failed) == 0,
                "max_abs_shear_stress_kpa": float(df["shear_stress_kpa"].abs().max()),
                "max_tau_unit_residual_kpa": tau_residual,
                "min_plastic_increment": min_plastic_diff,
                "scope_note": "Mathematical invariant audit for the scripted benchmark; supports numerical consistency, not experimental validation.",
            }
        ]
    )
    summary.to_csv(DATA / "mathematical_invariants_summary.csv", index=False)
    print(f"mathematical_invariants=ok checks={len(details)} failures={len(failed)}")
    if len(failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
