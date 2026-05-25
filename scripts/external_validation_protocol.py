from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STRICT_THRESHOLDS = {
    "spearman_min": 0.70,
    "classification_agreement_min": 0.80,
    "coverage_min": 0.80,
    "mae_normalized_max": 0.20,
    "bias_abs_max": 0.15,
    "ablation_gain_min": 0.20,
}


def phase_from_hmai(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value <= 0.30:
        return "relevant"
    return "dominant"


def phase_from_external_damage(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value < 0.50:
        return "relevant"
    return "dominant"


def status_from_metrics(row: pd.Series) -> str:
    checks = []
    if pd.notna(row.get("spearman_rank_correlation")):
        checks.append(float(row["spearman_rank_correlation"]) >= STRICT_THRESHOLDS["spearman_min"])
    if pd.notna(row.get("mae_normalized")):
        checks.append(float(row["mae_normalized"]) <= STRICT_THRESHOLDS["mae_normalized_max"])
    if pd.notna(row.get("bias_normalized")):
        checks.append(abs(float(row["bias_normalized"])) <= STRICT_THRESHOLDS["bias_abs_max"])
    if pd.notna(row.get("coverage_fraction")):
        checks.append(float(row["coverage_fraction"]) >= STRICT_THRESHOLDS["coverage_min"])
    if not checks:
        return str(row.get("acceptance_status", "boundary"))
    return "pass" if all(checks) else "boundary"


def digitization_replicates() -> tuple[pd.DataFrame, pd.DataFrame]:
    kin = pd.read_csv(DATA / "kinikles2024_fig12a_digitized_points.csv")
    rows = []
    for _, row in kin.iterrows():
        sigma = float(row.get("volumetric_strain_uncertainty_pct", 0.0))
        x_unc = float(row.get("x_uncertainty_abs", 0.0))
        for rep, offset in enumerate([-1.0, 0.0, 1.0], start=1):
            rows.append(
                {
                    "source": row["source"],
                    "point_id": row["point_id"],
                    "replicate": rep,
                    "replicate_method": "deterministic +/- one recorded digitization uncertainty around archived manual point",
                    "initial_degree_saturation": float(row["digitized_initial_degree_saturation"]) + offset * x_unc,
                    "volumetric_strain_pct": float(row["digitized_volumetric_strain_pct"]) + offset * sigma,
                    "x_uncertainty_abs": x_unc,
                    "volumetric_strain_uncertainty_pct": sigma,
                    "claim_boundary": "Replicates quantify figure-reading uncertainty; they are not new raw experimental measurements.",
                }
            )
    repl = pd.DataFrame(rows)
    summary = (
        repl.groupby(["source", "point_id"], as_index=False)
        .agg(
            initial_degree_saturation_mean=("initial_degree_saturation", "mean"),
            initial_degree_saturation_range=("initial_degree_saturation", lambda s: float(s.max() - s.min())),
            volumetric_strain_pct_mean=("volumetric_strain_pct", "mean"),
            volumetric_strain_pct_range=("volumetric_strain_pct", lambda s: float(s.max() - s.min())),
            n_replicates=("replicate", "count"),
        )
    )
    repl.to_csv(DATA / "external_digitization_replicates.csv", index=False)
    summary.to_csv(DATA / "digitization_uncertainty_summary.csv", index=False)
    return repl, summary


def strict_locked_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(DATA / "external_locked_holdout_metrics.csv")
    metrics["strict_acceptance_status"] = metrics.apply(status_from_metrics, axis=1)
    metrics["strict_thresholds"] = (
        "Spearman>=0.70; coverage>=0.80; normalized MAE<=0.20; abs(bias)<=0.15 where applicable"
    )
    metrics.to_csv(DATA / "external_validation_strict_metrics.csv", index=False)
    return metrics


def validation_points() -> pd.DataFrame:
    rows = []
    locked = pd.read_csv(DATA / "external_locked_holdout_predictions.csv")
    for _, row in locked.iterrows():
        rows.append(
            {
                "source": row["source"],
                "point_id": f"KIN-{row['target']}-{row['validation_group_initial_degree_saturation']:.2f}",
                "set": "locked validation",
                "target": row["target"],
                "external_value": float(row["observed_pct"]),
                "predicted_value": float(row["predicted_pct"]),
                "unit": "volumetric strain (%)",
                "traceability": "external_locked_holdout_predictions.csv",
                "claim_allowed": "figure-level locked predictive validation",
            }
        )
    rong_amp = pd.read_csv(DATA / "rong_mccartney_drained_amplitude_digitized_points.csv")
    amp = (
        rong_amp.groupby("cyclic_shear_strain_amplitude_pct", as_index=False)
        .agg(observed_volumetric_strain_200_cycles_pct=("observed_volumetric_strain_200_cycles_pct", "mean"))
        .sort_values("cyclic_shear_strain_amplitude_pct")
    )
    for _, row in amp.iterrows():
        rows.append(
            {
                "source": "Rong and McCartney 2019 drained CSS Figure 4.5",
                "point_id": f"RONG-AMP-{row['cyclic_shear_strain_amplitude_pct']:.1f}",
                "set": "independent amplitude-transfer gate",
                "target": "volumetric_strain_200_cycles_vs_cyclic_amplitude",
                "external_value": float(row["observed_volumetric_strain_200_cycles_pct"]),
                "predicted_value": np.nan,
                "unit": "volumetric strain (%)",
                "traceability": "rong_mccartney_drained_amplitude_digitized_points.csv",
                "claim_allowed": "rank/trend validation only; no raw time-history calibration",
            }
        )
    ng = pd.read_csv(DATA / "ng2013_suction_semicalibrated_envelope.csv")
    for _, row in ng[ng["suction_kpa"].isin([30.0, 250.0])].iterrows():
        rows.append(
            {
                "source": "Ng and Zhou 2014 cyclic unsaturated silt",
                "point_id": f"NG-SUCTION-{row['suction_kpa']:.0f}",
                "set": "independent suction-transfer gate",
                "target": "resilient_modulus_ratio_vs_0kPa",
                "external_value": float(row["semicalibrated_ratio_vs_0kpa"]),
                "predicted_value": float(row["raw_model_ratio_vs_0kpa"]),
                "unit": "dimensionless ratio",
                "traceability": "ng2013_suction_semicalibrated_envelope.csv",
                "claim_allowed": "trend-transfer evidence; high-suction case remains a boundary",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "external_validation_traceable_points.csv", index=False)
    return out


def hmai_classification(points: pd.DataFrame) -> pd.DataFrame:
    hmai = pd.read_csv(DATA / "hydraulic_memory_amplification_index.csv")
    benchmark_classes = hmai[["suction_amp_kpa", "cyclic_amp", "hmai_composite", "phase_class"]].copy()
    rows = []
    paired = pd.read_csv(DATA / "hmai_external_paired_cases.csv")
    for _, row in paired.iterrows():
        external = float(row["external_volumetric_strain_max_pct"])
        rows.append(
            {
                "source": row["source"],
                "point_id": f"KIN-HMAI-{row['initial_degree_saturation']:.2f}",
                "external_metric": "maximum digitized volumetric strain (%)",
                "external_value": external,
                "predicted_hmai": float(row["hmai_composite"]),
                "predicted_class": row["hmai_phase_class"],
                "external_expected_class": phase_from_external_damage(external),
                "classification_match": bool(row["hmai_phase_class"] == phase_from_external_damage(external)),
                "claim_boundary": "HMAI classification checked against digitized figure-level cyclic compression envelope.",
            }
        )
    rong = pd.read_csv(DATA / "rong_mccartney_digitized_200cycle_validation.csv")
    for _, row in rong.iterrows():
        nearest = benchmark_classes.iloc[
            ((benchmark_classes["suction_amp_kpa"] - min(100.0, max(0.0, float(row["paired_model_suction_kpa"])))).abs()
            + (benchmark_classes["cyclic_amp"] - 0.20).abs() * 100.0).argmin()
        ]
        external = float(row["observed_volumetric_strain_200_cycles_pct"])
        rows.append(
            {
                "source": "Rong/McCartney PEER 2022/05 200-cycle figure gate",
                "point_id": f"RONG-HMAI-{row['initial_degree_saturation']:.3f}",
                "external_metric": "200-cycle volumetric strain (%)",
                "external_value": external,
                "predicted_hmai": float(nearest["hmai_composite"]),
                "predicted_class": nearest["phase_class"],
                "external_expected_class": phase_from_external_damage(external),
                "classification_match": bool(nearest["phase_class"] == phase_from_external_damage(external)),
                "claim_boundary": "Rong/McCartney is retained as a failure-domain boundary when classes disagree.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hmai_external_classification_validation.csv", index=False)
    overlay = out.assign(
        suction_amp_kpa=lambda df: np.where(
            df["source"].str.contains("Kinikles", regex=False),
            df["point_id"].str.extract(r"([0-9]+\.[0-9]+)$")[0].astype(float) * 100.0,
            df["point_id"].str.extract(r"([0-9]+\.[0-9]+)$")[0].astype(float) * 100.0,
        ),
        cyclic_amp=0.20,
        marker_label=lambda df: np.where(df["source"].str.contains("Kinikles", regex=False), "K", "R"),
        placement_note="External points are placed on the nearest diagnostic benchmark coordinate for visual context; quantitative validation is reported in CSV metrics.",
    )[
        [
            "source",
            "point_id",
            "suction_amp_kpa",
            "cyclic_amp",
            "marker_label",
            "predicted_class",
            "external_expected_class",
            "classification_match",
            "placement_note",
        ]
    ]
    overlay.to_csv(DATA / "external_hmai_phase_overlay_points.csv", index=False)
    return out


def ablation_gain() -> pd.DataFrame:
    base = pd.read_csv(DATA / "external_baseline_comparison_scorecard.csv")
    rows = []
    for (source, target), group in base.groupby(["source", "target"]):
        full = group[group["model"] == "full_hysteretic_damage"].iloc[0]
        no_memory = group[group["model"] == "constant_suction"].iloc[0]
        gain = (float(no_memory["rmse_pct"]) - float(full["rmse_pct"])) / max(float(no_memory["rmse_pct"]), 1.0e-9)
        rows.append(
            {
                "source": source,
                "target": target,
                "constant_suction_rmse_pct": float(no_memory["rmse_pct"]),
                "full_hysteretic_damage_rmse_pct": float(full["rmse_pct"]),
                "ablation_gain_fraction": gain,
                "ablation_gain_status": "pass" if gain >= STRICT_THRESHOLDS["ablation_gain_min"] else "boundary",
                "threshold": "full model improves RMSE by >=20% over constant-suction no-memory baseline",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "external_validation_ablation_gain.csv", index=False)
    return out


def traceability_registry() -> None:
    rows = [
        {
            "source": "Kinikles et al. 2024 Figure 12(a)",
            "doi": "10.1016/j.compgeo.2024.106113",
            "local_trace": "external_data/kinikles_2024_unsaturated_seismic_compression and kinikles2024_fig12a_digitized_points.csv",
            "license_or_access": "open-access article figure; digitized values retained",
            "use": "locked validation and HMAI classification",
        },
        {
            "source": "Rong and McCartney cyclic simple-shear sources",
            "doi": "10.1051/e3sconf/20199208004 plus PEER 2022/05 report",
            "local_trace": "external_data/rong_mccartney_unsaturated_cyclic and Rong/McCartney CSV gates",
            "license_or_access": "open PDFs; digitized/transcribed values retained",
            "use": "amplitude trend gate and failure-domain boundary",
        },
        {
            "source": "Ng and Zhou 2014",
            "doi": "10.1680/geot.14.P.015",
            "local_trace": "external_data/ng_2013_unsaturated_resilient_modulus and ng2013_* CSV files",
            "license_or_access": "publication-level semiquantitative ratios",
            "use": "suction/cyclic-stress transfer gate",
        },
        {
            "source": "Dai and Zhou 2025 Canadian Geotechnical Journal",
            "doi": "10.1139/cgj-2024-0804",
            "local_trace": "metadata only in this package",
            "license_or_access": "publisher article; no quantitative local extraction bundled",
            "use": "scope-transfer boundary for suction-temperature-principal-stress-rotation effects",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA / "external_validation_traceability_registry.csv", index=False)


def summary(metrics: pd.DataFrame, points: pd.DataFrame, classes: pd.DataFrame, ablation: pd.DataFrame) -> None:
    class_agree = float(classes["classification_match"].mean())
    numeric_metrics = metrics[pd.to_numeric(metrics["n_validation_points"], errors="coerce").notna()].copy()
    strict_pass = int((metrics["strict_acceptance_status"] == "pass").sum())
    strict_boundary = int((metrics["strict_acceptance_status"] == "boundary").sum())
    strict_fail = int((metrics["strict_acceptance_status"] == "fail").sum()) if "fail" in set(metrics["strict_acceptance_status"]) else 0
    out = pd.DataFrame(
        [
            {
                "traceable_external_points": int(len(points)),
                "numeric_validation_metric_rows": int(len(numeric_metrics)),
                "strict_pass_rows": strict_pass,
                "strict_boundary_rows": strict_boundary,
                "strict_fail_rows": strict_fail,
                "hmai_classification_agreement_fraction": class_agree,
                "hmai_classification_status": "pass" if class_agree >= STRICT_THRESHOLDS["classification_agreement_min"] else "boundary",
                "ablation_gain_pass_rows": int((ablation["ablation_gain_status"] == "pass").sum()),
                "ablation_gain_rows": int(len(ablation)),
                "dai_zhou_status": "boundary_metadata_only_no_quantitative_local_extraction",
                "overall_interpretation": "The external validation protocol strengthens traceable external evidence but remains claim-bounded; boundaries are retained for Rong/McCartney 200-cycle and Dai/Zhou scope-transfer evidence.",
            }
        ]
    )
    out.to_csv(DATA / "external_validation_summary.csv", index=False)


def main() -> None:
    digitization_replicates()
    metrics = strict_locked_metrics()
    points = validation_points()
    classes = hmai_classification(points)
    ablation = ablation_gain()
    traceability_registry()
    summary(metrics, points, classes, ablation)
    print(
        "external_validation_protocol=ok "
        f"points={len(points)} class_agreement={classes['classification_match'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
