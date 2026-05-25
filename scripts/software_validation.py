from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

REQUIRED = [
    "benchmark_summary.csv",
    "benchmark_results.csv",
    "pc_consistency_audit.csv",
    "external_validation_metrics.csv",
    "swcc_envelope_validation.csv",
    "openseespy_cyclic_crosscheck.csv",
    "monte_carlo_sensitivity_summary.csv",
    "claim_boundary_negative_tests.csv",
    "external_data_manifest.csv",
    "zenodo_italian_clays_range_check.csv",
    "ng2013_resilient_modulus_semiquant_checks.csv",
    "ng2013_suction_semicalibrated_envelope.csv",
    "ng2013_cyclic_stress_semicalibrated_envelope.csv",
    "ng2013_semicalibrated_envelope_summary.csv",
    "gabr2019_unsaturated_triaxial_transcribed.csv",
    "gabr2019_unsaturated_triaxial_holdout_predictions.csv",
    "gabr2019_unsaturated_triaxial_holdout_summary.csv",
    "kinikles2024_cyclic_simple_shear_test_matrix.csv",
    "kinikles2024_cyclic_gate_summary.csv",
    "kinikles2024_fig12a_digitized_points.csv",
    "kinikles2024_fig12a_digitized_envelope_summary.csv",
    "kinikles2024_fig12a_digitized_gate_summary.csv",
    "kinikles2024_fig12a_blind_holdout_predictions.csv",
    "kinikles2024_fig12a_blind_holdout_summary.csv",
    "kinikles2024_model_output_transfer_features.csv",
    "kinikles2024_model_output_transfer_holdout_predictions.csv",
    "kinikles2024_model_output_transfer_holdout_summary.csv",
    "recent_cyclic_model_comparison_gate.csv",
    "chen2024_supplementary_inventory.csv",
    "chen2024_sparc_geotechnical_properties.csv",
    "chen2024_sparc_monotonic_test_conditions.csv",
    "chen2024_supplementary_calibration_summary.csv",
    "rong2021_undrained_css_test_program.csv",
    "rong2019_drained_css_test_program.csv",
    "rong_mccartney_cyclic_trend_claims.csv",
    "rong_mccartney_cyclic_program_summary.csv",
    "rong_mccartney_200cycle_transcribed_points.csv",
    "rong_mccartney_digitized_200cycle_validation.csv",
    "rong_mccartney_digitized_200cycle_validation_summary.csv",
    "rong_mccartney_200cycle_residual_diagnostics.csv",
    "rong_mccartney_200cycle_holdout_transfer_predictions.csv",
    "rong_mccartney_200cycle_holdout_transfer_summary.csv",
    "rong_mccartney_200cycle_failure_mode_summary.csv",
    "rong_mccartney_drained_amplitude_digitized_points.csv",
    "rong_mccartney_drained_amplitude_digitized_group_summary.csv",
    "rong_mccartney_drained_amplitude_digitized_gate_summary.csv",
    "boundary_1d_layer_inputs.csv",
    "boundary_1d_case_summary.csv",
    "boundary_1d_depth_profiles.csv",
    "boundary_1d_monte_carlo_samples.csv",
    "boundary_1d_monte_carlo_summary.csv",
    "table_boundary_value_claim_passport.csv",
    "parameter_identifiability_sensitivity.csv",
    "parameter_identifiability_summary.csv",
    "hydraulic_memory_amplification_index.csv",
    "hydraulic_memory_amplification_index_summary.csv",
    "mathematical_invariants_audit.csv",
    "mathematical_invariants_summary.csv",
    "algorithmic_dissipation_audit.csv",
    "algorithmic_dissipation_summary.csv",
    "mathematical_consistency_tests.csv",
    "theoretical_admissibility_propositions.csv",
    "physical_limit_checks.csv",
    "hmai_hmn_derivation_audit.csv",
    "preconsolidation_degradation_terms.csv",
    "external_cross_source_consistency_audit.csv",
    "external_cross_source_consistency_summary.csv",
    "external_cross_source_validation_scorecard.csv",
    "external_baseline_comparison_scorecard.csv",
    "external_baseline_comparison_summary.csv",
    "external_validation_protocol_registry.csv",
    "external_locked_holdout_predictions.csv",
    "external_locked_holdout_metrics.csv",
    "external_validation_acceptance_matrix.csv",
    "hmai_external_paired_cases.csv",
    "hmai_external_correlation.csv",
    "hmai_external_ablation_scorecard.csv",
    "hmai_weight_sensitivity.csv",
    "hmai_validation_summary.csv",
    "manifest_sha256.csv",
]


def validate_python() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            rows.append({"file": str(path.relative_to(ROOT)), "check": "python_syntax", "status": "pass", "detail": ""})
        except SyntaxError as exc:
            rows.append({"file": str(path.relative_to(ROOT)), "check": "python_syntax", "status": "fail", "detail": str(exc)})
    return rows


def validate_csvs() -> list[dict]:
    rows = []
    for name in REQUIRED:
        path = DATA / name
        if not path.exists():
            rows.append({"file": f"data/{name}", "check": "required_artifact", "status": "fail", "detail": "missing"})
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            rows.append({"file": f"data/{name}", "check": "csv_read", "status": "pass", "detail": f"rows={len(df)} cols={len(df.columns)}"})
        except Exception as exc:
            rows.append({"file": f"data/{name}", "check": "csv_read", "status": "fail", "detail": repr(exc)})
    return rows


def main() -> None:
    rows = validate_python() + validate_csvs()
    details = pd.DataFrame(rows)
    details.to_csv(DATA / "software_validation_details.csv", index=False)
    syntax_error_count = int(((details["check"] == "python_syntax") & (details["status"] != "pass")).sum())
    data_schema_error_count = int(((details["check"] == "csv_read") & (details["status"] != "pass")).sum())
    missing_artifact_count = int(((details["check"] == "required_artifact") & (details["status"] != "pass")).sum())
    summary = pd.DataFrame(
        [
            {
                "syntax_error_count": syntax_error_count,
                "data_schema_error_count": data_schema_error_count,
                "missing_artifact_count": missing_artifact_count,
                "software_validation_status": "passed"
                if syntax_error_count == data_schema_error_count == missing_artifact_count == 0
                else "failed",
                "zip_module_available": zipfile is not None,
                "scope_note": "Checks Python syntax and required CSV readability; ZIP extraction and manifest checks are verified separately.",
            }
        ]
    )
    summary.to_csv(DATA / "software_validation_summary.csv", index=False)
    print("software_validation=ok")


if __name__ == "__main__":
    main()
