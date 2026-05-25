from __future__ import annotations

import csv
from pathlib import Path

from run_unsat_cyclic_benchmark import Params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clipped_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def phase_class(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value <= 0.30:
        return "relevant"
    return "dominant"


def main() -> int:
    params = Params()
    rows = read_rows(DATA / "benchmark_summary.csv")
    by_case: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["suction_amp"], row["cyclic_amp"])
        by_case.setdefault(key, {})[row["model"]] = row

    out_rows: list[dict[str, object]] = []
    for (suction_amp, cyclic_amp), models in sorted(by_case.items(), key=lambda item: (float(item[0][0]), float(item[0][1]))):
        baseline = models.get("constant_suction")
        full = models.get("full_hysteretic_damage")
        no_hyst = models.get("no_hysteresis")
        hyst_only = models.get("hysteresis_only")
        if not no_hyst or not hyst_only:
            continue

        base_plastic_raw = float(baseline["final_plastic_strain_index"]) if baseline else 0.0
        base_plastic = max(base_plastic_raw, 0.0)
        full_plastic = float(full["final_plastic_strain_index"]) if full else 0.0
        no_hyst_plastic = float(no_hyst["final_plastic_strain_index"])
        hyst_only_plastic = float(hyst_only["final_plastic_strain_index"])
        full_stiffness_loss = float(full["stiffness_loss_pct"]) if full else 0.0
        baseline_stiffness_loss = float(baseline["stiffness_loss_pct"]) if baseline else 0.0
        no_hyst_stiffness_loss = float(no_hyst["stiffness_loss_pct"])
        hyst_only_stiffness_loss = float(hyst_only["stiffness_loss_pct"])
        min_saturation = min(float(no_hyst["min_saturation"]), float(hyst_only["min_saturation"]))
        max_saturation = max(float(no_hyst["max_saturation"]), float(hyst_only["max_saturation"]))
        case_scale = max(
            base_plastic,
            full_plastic,
            no_hyst_plastic,
            hyst_only_plastic,
            1e-9,
        )
        stiffness_scale = max(
            full_stiffness_loss,
            baseline_stiffness_loss,
            no_hyst_stiffness_loss,
            hyst_only_stiffness_loss,
            1e-9,
        )
        plastic_hmai = clipped_unit((hyst_only_plastic - no_hyst_plastic) / case_scale)
        stiffness_hmai = clipped_unit((hyst_only_stiffness_loss - no_hyst_stiffness_loss) / stiffness_scale)
        hydraulic_path_intensity = clipped_unit((max_saturation - min_saturation) / max(params.sr_sat - params.sr_res, 1e-9))
        suction_amp_float = float(suction_amp)
        hydraulic_history_gate = 0.0 if abs(suction_amp_float) < 1e-12 else 1.0
        composite_hmai = clipped_unit(
            hydraulic_history_gate
            * (0.40 * plastic_hmai + 0.30 * stiffness_hmai + 0.30 * hydraulic_path_intensity)
        )
        legacy_full_vs_constant = clipped_unit((full_plastic - base_plastic) / case_scale)
        ratio = "" if base_plastic < 1e-6 else full_plastic / base_plastic
        phase = phase_class(composite_hmai)

        out_rows.append(
            {
                "suction_amp_kpa": suction_amp_float,
                "cyclic_amp": float(cyclic_amp),
                "hydraulic_memory_amplification_index": composite_hmai,
                "hmai_composite": composite_hmai,
                "hmai_plastic_component": plastic_hmai,
                "hmai_stiffness_component": stiffness_hmai,
                "hmai_hydraulic_path_component": hydraulic_path_intensity,
                "legacy_full_vs_constant_plastic_contrast": legacy_full_vs_constant,
                "hydraulic_damage_normalized": clipped_unit(float(full["final_hydraulic_damage"]) / max(params.beta_h, 1e-9)) if full else 0.0,
                "seismic_damage_normalized": clipped_unit(float(full["final_seismic_damage"]) / max(params.beta_c, 1e-9)) if full else 0.0,
                "phase_class": phase,
                "phase_thresholds": "negligible <0.10; relevant 0.10-0.30; dominant >0.30",
                "plastic_strain_ratio_vs_constant": ratio,
                "plastic_strain_delta_vs_constant": full_plastic - base_plastic,
                "plastic_strain_delta_vs_no_hysteresis": full_plastic - no_hyst_plastic,
                "plastic_strain_delta_vs_hysteresis_only": full_plastic - hyst_only_plastic,
                "hysteresis_only_plastic_delta_vs_no_hysteresis": hyst_only_plastic - no_hyst_plastic,
                "full_model_stiffness_loss_pct": full_stiffness_loss,
                "constant_suction_stiffness_loss_pct": baseline_stiffness_loss,
                "interpretation": "Composite dimensionless benchmark index: 0.40 plastic-strain contrast, 0.30 stiffness-loss contrast, and 0.30 hydraulic-path intensity for the hysteresis-only hydraulic-memory branch against the no-hysteresis suction branch. Phase thresholds are diagnostic classes for this benchmark only, not design limits.",
            }
        )

    out_path = DATA / "hydraulic_memory_amplification_index.csv"
    fieldnames = [
        "suction_amp_kpa",
        "cyclic_amp",
        "hydraulic_memory_amplification_index",
        "hmai_composite",
        "hmai_plastic_component",
        "hmai_stiffness_component",
        "hmai_hydraulic_path_component",
        "legacy_full_vs_constant_plastic_contrast",
        "hydraulic_damage_normalized",
        "seismic_damage_normalized",
        "phase_class",
        "phase_thresholds",
        "plastic_strain_ratio_vs_constant",
        "plastic_strain_delta_vs_constant",
        "plastic_strain_delta_vs_no_hysteresis",
        "plastic_strain_delta_vs_hysteresis_only",
        "hysteresis_only_plastic_delta_vs_no_hysteresis",
        "full_model_stiffness_loss_pct",
        "constant_suction_stiffness_loss_pct",
        "interpretation",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    max_row = max(out_rows, key=lambda row: float(row["hmai_composite"]))
    phase_counts = {name: sum(1 for row in out_rows if row["phase_class"] == name) for name in ["negligible", "relevant", "dominant"]}
    summary_path = DATA / "hydraulic_memory_amplification_index_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cases",
                "max_hmai",
                "max_hmai_suction_amp_kpa",
                "max_hmai_cyclic_amp",
                "negligible_cases",
                "relevant_cases",
                "dominant_cases",
                "component_weights",
                "scope_note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cases": len(out_rows),
                "max_hmai": max_row["hydraulic_memory_amplification_index"],
                "max_hmai_suction_amp_kpa": max_row["suction_amp_kpa"],
                "max_hmai_cyclic_amp": max_row["cyclic_amp"],
                "negligible_cases": phase_counts["negligible"],
                "relevant_cases": phase_counts["relevant"],
                "dominant_cases": phase_counts["dominant"],
                "component_weights": "0.40 plastic contrast; 0.30 stiffness-loss contrast; 0.30 hydraulic-path intensity",
                "scope_note": "Composite HMAI is a bounded reproducible phase-map contrast used to diagnose when hydraulic memory materially changes cyclic response. It is not raw-data validation, site response, or a universal design coefficient.",
            }
        )

    print(f"hydraulic_memory_amplification_index=ok rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
