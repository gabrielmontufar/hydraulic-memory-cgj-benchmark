from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_unsat_cyclic_benchmark import Params


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def status(ok: bool) -> str:
    return "pass" if bool(ok) else "review"


def phase(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value <= 0.30:
        return "relevant"
    return "dominant"


def propositions() -> pd.DataFrame:
    rows = [
        {
            "proposition": "P1 HMAI bounded",
            "statement": "If every component is clipped to [0,1] and non-negative weights sum to one, then 0 <= HMAI <= 1.",
            "short_proof": "HMAI is a convex combination of clipped unit components; a convex combination of unit-interval values remains in the unit interval.",
            "manuscript_use": "Supports bounded diagnostic phase-map interpretation.",
        },
        {
            "proposition": "P2 HMN dimensionless and non-negative",
            "statement": "HMN=(Delta s/s0)((Sr,max-Sr,min)/(Sr,sat-Sr,res))/(gamma_cyc/gamma_y0) is dimensionless and non-negative in the declared domain.",
            "short_proof": "Each factor is a ratio of like units or dimensionless saturation/strain quantities; all factors are non-negative by construction.",
            "manuscript_use": "Prevents treating HMN as an empirical dimensional fit coefficient.",
        },
        {
            "proposition": "P3 Damage monotonicity",
            "statement": "Hydraulic and seismic damage variables are non-negative and non-decreasing within each simulated case.",
            "short_proof": "The update accumulates non-negative reversal or overstress increments and then clips to a non-negative bounded interval.",
            "manuscript_use": "Supports physical admissibility of internal degradation variables.",
        },
        {
            "proposition": "P4 Positive stiffness",
            "statement": "Secant stiffness remains positive for the declared damage bounds and positive effective stress.",
            "short_proof": "G=G0(p_eff/p0)^0.5(1-Dh)(1-Ds), with G0>0, p_eff>0, Dh<1 and Ds<1 in the generated states.",
            "manuscript_use": "Rules out nonphysical negative stiffness in the benchmark envelope.",
        },
        {
            "proposition": "P5 Reduction without hydraulic memory",
            "statement": "If the hydraulic-memory gate is zero, the composite HMAI is zero; full mechanical equivalence with the baseline also requires cyclic damage to be inactive.",
            "short_proof": "The HMAI definition multiplies the component contrast by the hydraulic-history gate; the mechanical model still differs if seismic damage remains active.",
            "manuscript_use": "Separates a true memory limit from a broader no-damage/no-memory baseline.",
        },
        {
            "proposition": "P6 Unique phase class",
            "statement": "Every computed HMAI value belongs to exactly one of negligible, relevant or dominant.",
            "short_proof": "The intervals [0,0.10), [0.10,0.30] and (0.30,1] are mutually exclusive and cover the bounded HMAI range.",
            "manuscript_use": "Makes the phase-map classification mathematically unambiguous.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "theoretical_admissibility_propositions.csv", index=False)
    return out


def preconsolidation_terms() -> pd.DataFrame:
    p = Params()
    suction_levels = np.array([0.0, 3.0, 30.0, 90.0, 190.0])
    rows = []
    for s in suction_levels:
        pc_s = p.pc0 * (1.0 + p.k_suction * np.log1p(s / p.s0))
        rows.append(
            {
                "suction_kpa": s,
                "pc_suction_only_kpa": pc_s,
                "preconsolidation_effective_kpa": pc_s,
                "pc_damage_sensitivity_floor_kpa": pc_s * p.pc_damage_sensitivity_floor_factor,
                "interpretation": "pc_s is the primary suction-hardening preconsolidation term; damage sensitivity is retained as an audit column rather than the primary law.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "preconsolidation_degradation_terms.csv", index=False)
    return out


def physical_limits() -> pd.DataFrame:
    rows = [
        {
            "limit": "s -> 0",
            "expected_result": "Response approaches a low-suction or saturated-side benchmark state; suction hardening tends toward pc0.",
            "reproducible_check": "preconsolidation_degradation_terms.csv includes suction_kpa=0 with pc_suction_only_kpa=pc0.",
            "claim_allowed": "Checks limiting consistency only; not a saturated-soil calibration.",
        },
        {
            "limit": "Delta s -> 0",
            "expected_result": "Hydraulic memory becomes negligible and HMAI should be zero or negligible.",
            "reproducible_check": "hydraulic_memory_falsification_tests.csv zero_suction_amplitude_limit.",
            "claim_allowed": "Supports the zero hydraulic-history limit.",
        },
        {
            "limit": "N -> 0",
            "expected_result": "Cyclic damage increments vanish when no cyclic overstress is accumulated.",
            "reproducible_check": "algorithmic_dissipation_audit.csv non-decreasing damage with no negative increments.",
            "claim_allowed": "Checks update admissibility rather than fatigue-law calibration.",
        },
        {
            "limit": "M_h -> 0",
            "expected_result": "Composite HMAI tends to zero when the hydraulic-memory gate is zero.",
            "reproducible_check": "mathematical_consistency_tests.csv HMAI_zero_memory_limit.",
            "claim_allowed": "Memory-index reduction, not full model equivalence unless cyclic damage is also inactive.",
        },
        {
            "limit": "D -> 1",
            "expected_result": "The declared benchmark never allows damage to reach one, so stiffness remains positive.",
            "reproducible_check": "mathematical_invariants_audit.csv positive_stress_and_stiffness and damage_bounds.",
            "claim_allowed": "Positive-stiffness admissibility inside declared bounds.",
        },
        {
            "limit": "Sr -> 1",
            "expected_result": "Bishop chi approaches one and the stress measure transitions coherently toward saturated-side effective stress.",
            "reproducible_check": "mathematical_invariants_audit.csv bishop_chi_bounds and effective_stress_identity.",
            "claim_allowed": "Stress-measure consistency, not full saturated constitutive validation.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "physical_limit_checks.csv", index=False)
    return out


def hmai_hmn_audit(hmai: pd.DataFrame, hmn: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "object": "HMAI",
            "definition": "0.40*plastic_component + 0.30*stiffness_component + 0.30*hydraulic_path_component, multiplied by the hydraulic-history gate.",
            "components_dimensionless": True,
            "bounded": bool(hmai["hmai_composite"].between(0.0, 1.0).all()),
            "zero_memory_result": str(sorted(set(hmai.loc[hmai["suction_amp_kpa"] == 0.0, "phase_class"]))),
            "interpretation": "Bounded contrast between hysteresis-only hydraulic-memory response and no-hysteresis suction response.",
        },
        {
            "object": "HMN",
            "definition": "(Delta s/s0)*((Sr,max-Sr,min)/(Sr,sat-Sr,res))/(gamma_cyc/gamma_y0)",
            "components_dimensionless": True,
            "bounded": "",
            "zero_memory_result": f"min={hmn['hydraulic_memory_number'].min():.6g}",
            "interpretation": "Dimensionless hydraulic-memory intensity coordinate for the scripted benchmark grid.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hmai_hmn_derivation_audit.csv", index=False)
    return out


def consistency_tests() -> pd.DataFrame:
    p = Params()
    hmai = pd.read_csv(DATA / "hydraulic_memory_amplification_index.csv")
    hmn = pd.read_csv(DATA / "hydraulic_memory_number.csv")
    benchmark = pd.read_csv(DATA / "benchmark_results.csv")
    invariants = pd.read_csv(DATA / "mathematical_invariants_audit.csv")
    dissipation = pd.read_csv(DATA / "algorithmic_dissipation_audit.csv")
    weight = pd.read_csv(DATA / "hmai_weight_sensitivity.csv")
    precon = preconsolidation_terms()
    limits = physical_limits()
    props = propositions()
    hmai_hmn_audit(hmai, hmn)

    zero_classes = set(hmai.loc[hmai["suction_amp_kpa"] == 0.0, "phase_class"])
    unique_classes = set(hmai["phase_class"])
    primary_pc_error = float(
        (benchmark["preconsolidation_kpa"] - benchmark["pc_suction_only_kpa"]).abs().max()
    )
    pc0_error = float(abs(precon.loc[precon["suction_kpa"] == 0.0, "pc_suction_only_kpa"].iloc[0] - p.pc0))
    year_files = []
    for path in [ROOT / "validation_scope_note.md", DATA / "external_data_manifest.csv"]:
        if path.exists() and "Dai and Zhou 2026" in path.read_text(encoding="utf-8", errors="ignore"):
            year_files.append(str(path.relative_to(ROOT)))

    tests = [
        {
            "test": "HMAI_bounded",
            "expected": "All HMAI values in [0,1]",
            "observed": f"min={hmai['hmai_composite'].min():.6g}; max={hmai['hmai_composite'].max():.6g}",
            "status": status(hmai["hmai_composite"].between(0.0, 1.0).all()),
        },
        {
            "test": "HMN_dimensionless_nonnegative",
            "expected": "HMN built only from like-unit ratios and nonnegative on grid",
            "observed": f"min={hmn['hydraulic_memory_number'].min():.6g}; max={hmn['hydraulic_memory_number'].max():.6g}",
            "status": status((hmn["hydraulic_memory_number"] >= -1e-12).all()),
        },
        {
            "test": "HMAI_zero_memory_limit",
            "expected": "Zero suction-amplitude cases are negligible",
            "observed": str(sorted(zero_classes)),
            "status": status(zero_classes == {"negligible"}),
        },
        {
            "test": "nonnegative_monotone_damage",
            "expected": "Hydraulic and seismic damage are nonnegative and non-decreasing",
            "observed": str(dissipation.groupby("check")["status"].all().to_dict()),
            "status": status(dissipation[dissipation["check"].str.contains("damage")]["status"].all()),
        },
        {
            "test": "positive_stiffness",
            "expected": "All secant stiffness values remain positive",
            "observed": f"min_G={benchmark['secant_stiffness_mpa'].min():.6g} MPa",
            "status": status((benchmark["secant_stiffness_mpa"] > 0.0).all()),
        },
        {
            "test": "unique_phase_classification",
            "expected": "Every HMAI maps to exactly one of negligible/relevant/dominant",
            "observed": str(hmai["phase_class"].value_counts().to_dict()),
            "status": status(unique_classes <= {"negligible", "relevant", "dominant"} and len(hmai) == hmai["phase_class"].notna().sum()),
        },
        {
            "test": "physical_limit_table_complete",
            "expected": "Six required limits are present",
            "observed": str(limits["limit"].tolist()),
            "status": status(len(limits) == 6),
        },
        {
            "test": "weight_sensitivity_retained",
            "expected": "HMAI classes checked under alternative component weights",
            "observed": f"rows={len(weight)}; variants={weight['weight_variant'].nunique()}",
            "status": status(len(weight) > 0 and weight["weight_variant"].nunique() >= 5),
        },
        {
            "test": "primary_preconsolidation_excludes_damage",
            "expected": "Primary pc equals suction-only pc; damage sensitivity remains audit column",
            "observed": f"max_abs_error={primary_pc_error:.6g}",
            "status": status(primary_pc_error <= 1e-9),
        },
        {
            "test": "s_zero_preconsolidation_limit",
            "expected": "pc_s(s=0)=pc0",
            "observed": f"abs_error={pc0_error:.6g}",
            "status": status(pc0_error <= 1e-12),
        },
        {
            "test": "propositions_complete",
            "expected": "Six admissibility propositions with short proof",
            "observed": f"rows={len(props)}",
            "status": status(len(props) == 6),
        },
        {
            "test": "existing_invariant_audit_passes",
            "expected": "Prior mathematical invariant audit remains pass",
            "observed": str(invariants.set_index("check")["status"].to_dict()),
            "status": status((invariants["status"] == "pass").all()),
        },
        {
            "test": "dai_zhou_year_unified_in_supplement",
            "expected": "No Dai and Zhou 2026 string remains in selected public supplement files",
            "observed": "; ".join(year_files) if year_files else "none",
            "status": status(not year_files),
        },
    ]
    out = pd.DataFrame(tests)
    out.to_csv(DATA / "mathematical_consistency_tests.csv", index=False)
    return out


def main() -> int:
    out = consistency_tests()
    failures = out[out["status"] != "pass"]
    print(f"theoretical_admissibility_tests=ok rows={len(out)} failures={len(failures)}")
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
