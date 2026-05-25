from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from run_unsat_cyclic_benchmark import Params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def phase(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value <= 0.30:
        return "relevant"
    return "dominant"


def main() -> int:
    params = Params()
    hmai = pd.read_csv(DATA / "hydraulic_memory_amplification_index.csv")
    summary = pd.read_csv(DATA / "benchmark_summary.csv")

    hyst = summary[summary["model"] == "hysteresis_only"][
        ["suction_amp", "cyclic_amp", "min_saturation", "max_saturation"]
    ].copy()
    merged = hmai.merge(
        hyst,
        left_on=["suction_amp_kpa", "cyclic_amp"],
        right_on=["suction_amp", "cyclic_amp"],
        how="left",
    )
    saturation_span = (merged["max_saturation"] - merged["min_saturation"]).clip(lower=0.0)
    normalized_history = saturation_span / max(params.sr_sat - params.sr_res, 1e-9)
    normalized_suction = merged["suction_amp_kpa"] / max(params.s0, 1e-9)
    normalized_cyclic_demand = merged["cyclic_amp"] / max(params.gamma_y0, 1e-9)
    merged["hydraulic_memory_number"] = (
        normalized_suction * normalized_history / normalized_cyclic_demand.clip(lower=1e-9)
    )
    merged["hmn_definition"] = (
        "HMN=(suction_amp/s0)*((Sr_max-Sr_min)/(Sr_sat-Sr_res))/(cyclic_amp/gamma_y0)"
    )
    merged["hmn_scope_note"] = (
        "Diagnostic benchmark number only; not a universal constitutive parameter or design coefficient."
    )
    merged["hmai_phase_from_hmn_table"] = merged["hydraulic_memory_amplification_index"].map(phase)

    out_cols = [
        "suction_amp_kpa",
        "cyclic_amp",
        "min_saturation",
        "max_saturation",
        "hydraulic_memory_number",
        "hydraulic_memory_amplification_index",
        "hmai_phase_from_hmn_table",
        "hmn_definition",
        "hmn_scope_note",
    ]
    merged[out_cols].to_csv(DATA / "hydraulic_memory_number.csv", index=False)

    transition_rows = []
    for cyclic_amp, group in merged.sort_values("hydraulic_memory_number").groupby("cyclic_amp"):
        relevant = group[group["hydraulic_memory_amplification_index"] >= 0.10]
        dominant = group[group["hydraulic_memory_amplification_index"] > 0.30]
        transition_rows.append(
            {
                "cyclic_amp": cyclic_amp,
                "hmai_crit_relevant": 0.10,
                "hmncrit_relevant_min_observed": ""
                if relevant.empty
                else float(relevant["hydraulic_memory_number"].min()),
                "hmai_crit_dominant": 0.30,
                "hmncrit_dominant_min_observed": ""
                if dominant.empty
                else float(dominant["hydraulic_memory_number"].min()),
                "negligible_cases": int((group["hmai_phase_from_hmn_table"] == "negligible").sum()),
                "relevant_cases": int((group["hmai_phase_from_hmn_table"] == "relevant").sum()),
                "dominant_cases": int((group["hmai_phase_from_hmn_table"] == "dominant").sum()),
            }
        )
    pd.DataFrame(transition_rows).to_csv(DATA / "hydraulic_memory_transition_thresholds.csv", index=False)

    phase_counts = merged["hmai_phase_from_hmn_table"].value_counts().to_dict()
    falsification = pd.DataFrame(
        [
            {
                "test": "zero_suction_amplitude_limit",
                "rule": "Cases with suction_amp_kpa=0 should remain negligible.",
                "observed": str(
                    sorted(set(merged[merged["suction_amp_kpa"] == 0.0]["hmai_phase_from_hmn_table"]))
                ),
                "status": "pass"
                if set(merged[merged["suction_amp_kpa"] == 0.0]["hmai_phase_from_hmn_table"]) == {"negligible"}
                else "review",
            },
            {
                "test": "three_regime_phase_map",
                "rule": "HMAI grid must contain negligible, relevant and dominant classes before claiming a phase-map transition.",
                "observed": str(phase_counts),
                "status": "pass"
                if all(phase_counts.get(k, 0) > 0 for k in ["negligible", "relevant", "dominant"])
                else "review",
            },
            {
                "test": "hmn_nonnegative",
                "rule": "HMN must be nonnegative for the scripted benchmark grid.",
                "observed": f"min={merged['hydraulic_memory_number'].min():.6g}",
                "status": "pass" if merged["hydraulic_memory_number"].min() >= -1e-12 else "review",
            },
        ]
    )
    falsification.to_csv(DATA / "hydraulic_memory_falsification_tests.csv", index=False)

    originality = pd.DataFrame(
        [
            {
                "evidence_dimension": "hydraulic hysteresis",
                "prior_literature_covers": "yes",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "background component, not sufficient alone",
            },
            {
                "evidence_dimension": "cyclic unsaturated loading",
                "prior_literature_covers": "yes",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "background component, not sufficient alone",
            },
            {
                "evidence_dimension": "reproducible HMAI index",
                "prior_literature_covers": "partial_or_no",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "core contribution",
            },
            {
                "evidence_dimension": "HMN diagnostic number",
                "prior_literature_covers": "no",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "core contribution",
            },
            {
                "evidence_dimension": "three-regime phase map",
                "prior_literature_covers": "partial_or_no",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "core contribution",
            },
            {
                "evidence_dimension": "explicit falsification boundaries",
                "prior_literature_covers": "partial",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "claim-boundary contribution",
            },
            {
                "evidence_dimension": "open reproducibility package",
                "prior_literature_covers": "partial",
                "article117_cgj_package_covers": "yes",
                "novelty_role": "reviewer-verifiability contribution",
            },
        ]
    )
    originality.to_csv(DATA / "cgj_originality_matrix.csv", index=False)

    with (DATA / "hydraulic_memory_number_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cases",
                "min_hmn",
                "max_hmn",
                "negligible_cases",
                "relevant_cases",
                "dominant_cases",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cases": len(merged),
                "min_hmn": merged["hydraulic_memory_number"].min(),
                "max_hmn": merged["hydraulic_memory_number"].max(),
                "negligible_cases": phase_counts.get("negligible", 0),
                "relevant_cases": phase_counts.get("relevant", 0),
                "dominant_cases": phase_counts.get("dominant", 0),
                "status": "pass"
                if all(phase_counts.get(k, 0) > 0 for k in ["negligible", "relevant", "dominant"])
                else "review",
            }
        )

    print(
        "hydraulic_memory_number_transition=ok "
        f"cases={len(merged)} phases={phase_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
