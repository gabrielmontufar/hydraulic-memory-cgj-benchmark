from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"

KINIKLES_SOURCE = "Kinikles et al. 2024 Figure 12(a)"
RONG_SOURCE = "Rong/McCartney PEER 2022/05 200-cycle figure gate"
NG_SOURCE = "Ng and Zhou 2014 cyclic unsaturated silt"
DAI_SOURCE = "Dai and Zhou 2025 Canadian Geotechnical Journal"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def metric_row(source: str, target: str, validation_type: str, obs: np.ndarray, pred: np.ndarray, coverage: np.ndarray) -> dict:
    err = pred - obs
    obs_scale = max(float(np.max(np.abs(obs))), 1.0e-9)
    bias_norm = float(np.mean(err) / obs_scale)
    rmse_norm = float(np.sqrt(np.mean(err**2)) / obs_scale)
    mae_norm = float(np.mean(np.abs(err)) / obs_scale)
    rho = spearman(obs, pred)
    coverage_fraction = float(np.mean(coverage))
    status = "pass"
    if np.isnan(rho) or rho < 0.70 or rmse_norm > 0.30 or mae_norm > 0.20 or abs(bias_norm) > 0.15 or coverage_fraction < 0.80:
        status = "boundary"
    if rmse_norm > 0.60 or mae_norm > 0.50 or coverage_fraction < 0.34:
        status = "fail"
    return {
        "source": source,
        "target": target,
        "validation_type": validation_type,
        "n_validation_points": int(len(obs)),
        "rmse_pct": float(np.sqrt(np.mean(err**2))),
        "mae_pct": float(np.mean(np.abs(err))),
        "bias_pct": float(np.mean(err)),
        "rmse_normalized": rmse_norm,
        "mae_normalized": mae_norm,
        "bias_normalized": bias_norm,
        "spearman_rank_correlation": rho,
        "coverage_fraction": coverage_fraction,
        "acceptance_status": status,
            "claim_boundary": "Locked figure-level validation using digitized published evidence; normalized errors use maximum observed target magnitude; not raw cyclic time-history validation.",
    }


def write_registry() -> None:
    rows = [
        {
            "source": KINIKLES_SOURCE,
            "data_type": "digitized cyclic simple-shear volumetric-strain envelope",
            "extracted_variable": "median and maximum volumetric strain grouped by initial degree of saturation",
            "use_in_protocol": "primary locked holdout validation",
            "split_rule": "predeclared 3 train groups and 3 validation groups",
            "limitation": "figure-level envelope; no public raw time histories located in the package",
        },
        {
            "source": RONG_SOURCE,
            "data_type": "digitized 200-cycle volumetric-strain boundary",
            "extracted_variable": "volumetric strain at 200 cycles versus initial degree of saturation",
            "use_in_protocol": "failure-domain stress test",
            "split_rule": "not calibrated; reported as boundary diagnostic",
            "limitation": "rank trend conflicts with benchmark proxy and must bound claims",
        },
        {
            "source": NG_SOURCE,
            "data_type": "published suction and cyclic-stress trend evidence",
            "extracted_variable": "semiquantitative resilient-modulus and trend ratios",
            "use_in_protocol": "trend-transfer gate",
            "split_rule": "not used for parameter tuning",
            "limitation": "approximate publication-level ratios, not raw-data validation",
        },
        {
            "source": DAI_SOURCE,
            "data_type": "cyclic unsaturated-loess benchmark literature",
            "extracted_variable": "reported trend: suction raises resilient modulus and lowers permanent strain; PSR/temperature broaden scope",
            "use_in_protocol": "recent suction-temperature-PSR scope and trend-transfer benchmark",
            "split_rule": "not used for parameter tuning",
            "limitation": "model lacks temperature and principal-stress rotation variables",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA / "external_validation_protocol_registry.csv", index=False)


def locked_kinikles_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(DATA / "kinikles2024_model_output_transfer_features.csv")
    train_groups = {0.00, 0.20, 0.56}
    validation_groups = {0.12, 0.30, 0.40}
    rows = []
    metrics = []
    for target in ["volumetric_strain_median_pct", "volumetric_strain_max_pct"]:
        train = features[features["group_initial_degree_saturation"].round(2).isin(train_groups)].copy()
        validation = features[features["group_initial_degree_saturation"].round(2).isin(validation_groups)].copy()
        x_train = train["model_final_plastic_strain_index"].to_numpy(float)
        y_train = train[target].to_numpy(float)
        coeff = np.polyfit(x_train, y_train, 1)
        for _, row in validation.iterrows():
            pred = float(np.polyval(coeff, float(row["model_final_plastic_strain_index"])))
            lo = float(row["volumetric_strain_min_pct"])
            hi = float(row["volumetric_strain_max_pct"])
            rows.append(
                {
                    "source": KINIKLES_SOURCE,
                    "target": target,
                    "validation_type": "locked_50_50_saturation_group_holdout",
                    "train_groups": "0.00;0.20;0.56",
                    "validation_group_initial_degree_saturation": float(row["group_initial_degree_saturation"]),
                    "locked_transfer_model": "affine fit from benchmark final plastic-strain index to digitized target; coefficients fit on training groups only",
                    "coefficient_intercept": float(coeff[1]),
                    "coefficient_slope": float(coeff[0]),
                    "model_final_plastic_strain_index": float(row["model_final_plastic_strain_index"]),
                    "observed_pct": float(row[target]),
                    "predicted_pct": pred,
                    "absolute_error_pct": abs(pred - float(row[target])),
                    "error_pct": pred - float(row[target]),
                    "observed_group_min_pct": lo,
                    "observed_group_max_pct": hi,
                    "prediction_inside_digitized_group_range": bool(lo <= pred <= hi),
                }
            )
        val_df = pd.DataFrame([r for r in rows if r["target"] == target])
        metrics.append(
            metric_row(
                KINIKLES_SOURCE,
                target,
                "locked_50_50_saturation_group_holdout",
                val_df["observed_pct"].to_numpy(float),
                val_df["predicted_pct"].to_numpy(float),
                val_df["prediction_inside_digitized_group_range"].to_numpy(bool),
            )
        )
    pred_df = pd.DataFrame(rows)
    metric_df = pd.DataFrame(metrics)
    return pred_df, metric_df


def boundary_and_transfer_metrics() -> pd.DataFrame:
    rows = []
    rong = pd.read_csv(DATA / "rong_mccartney_200cycle_failure_mode_summary.csv").iloc[0]
    rows.append(
        {
            "source": RONG_SOURCE,
            "target": "volumetric_strain_200_cycles_pct",
            "validation_type": "failure_domain_stress_test",
            "n_validation_points": 6,
            "rmse_pct": float(rong["best_transfer_rmse_pct"]),
            "mae_pct": "",
            "bias_pct": "",
            "rmse_normalized": "",
            "mae_normalized": "",
            "bias_normalized": "",
            "spearman_rank_correlation": float(rong["spearman_rank_correlation"]),
            "coverage_fraction": "",
            "acceptance_status": "boundary",
            "claim_boundary": str(rong["claim_boundary"]),
        }
    )
    ng = pd.read_csv(DATA / "ng2013_semicalibrated_envelope_summary.csv").iloc[0]
    rows.append(
        {
            "source": NG_SOURCE,
            "target": "suction_resilient_modulus_trend",
            "validation_type": "trend_transfer_gate",
            "n_validation_points": 3,
            "rmse_pct": "",
            "mae_pct": "",
            "bias_pct": "",
            "rmse_normalized": "",
            "mae_normalized": "",
            "bias_normalized": "",
            "spearman_rank_correlation": 1.0,
            "coverage_fraction": "",
            "acceptance_status": "pass",
            "claim_boundary": str(ng["claim_boundary"]),
        }
    )
    rows.append(
        {
            "source": DAI_SOURCE,
            "target": "recent_suction_temperature_PSR_scope",
            "validation_type": "scope_and_trend_transfer_gate",
            "n_validation_points": "",
            "rmse_pct": "",
            "mae_pct": "",
            "bias_pct": "",
            "rmse_normalized": "",
            "mae_normalized": "",
            "bias_normalized": "",
            "spearman_rank_correlation": "",
            "coverage_fraction": "",
            "acceptance_status": "boundary",
            "claim_boundary": "Used to benchmark scope against recent cyclic unsaturated-loess evidence; not used as quantitative validation because the benchmark does not include temperature or principal-stress rotation.",
        }
    )
    return pd.DataFrame(rows)


def acceptance_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    thresholds = [
        ("spearman_rank_correlation", ">= 0.70", "monotonic trend agreement"),
        ("mae_normalized", "<= 0.20", "average normalized error"),
        ("rmse_normalized", "<= 0.30", "normalized error energy"),
        ("bias_normalized", "within +/-0.15", "mean bias"),
        ("coverage_fraction", ">= 0.80", "prediction inside digitized envelope/range"),
    ]
    rows = []
    for _, metric in metrics.iterrows():
        for field, threshold, purpose in thresholds:
            value = metric.get(field, "")
            if value == "" or pd.isna(value):
                status = "not_applicable"
            elif field == "spearman_rank_correlation":
                status = "pass" if float(value) >= 0.70 else "boundary"
            elif field in {"mae_normalized", "rmse_normalized"}:
                limit = 0.20 if field == "mae_normalized" else 0.30
                status = "pass" if float(value) <= limit else "boundary"
            elif field == "bias_normalized":
                status = "pass" if abs(float(value)) <= 0.15 else "boundary"
            else:
                status = "pass" if float(value) >= 0.80 else "boundary"
            rows.append(
                {
                    "source": metric["source"],
                    "target": metric["target"],
                    "validation_type": metric["validation_type"],
                    "metric": field,
                    "threshold": threshold,
                    "purpose": purpose,
                    "observed_value": value,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    write_registry()
    pred, kin_metrics = locked_kinikles_holdout()
    pred.to_csv(DATA / "external_locked_holdout_predictions.csv", index=False)
    metrics = pd.concat([kin_metrics, boundary_and_transfer_metrics()], ignore_index=True)
    metrics.to_csv(DATA / "external_locked_holdout_metrics.csv", index=False)
    acceptance_matrix(metrics).to_csv(DATA / "external_validation_acceptance_matrix.csv", index=False)
    print(f"external_locked_validation_protocol=ok metrics={len(metrics)} locked_predictions={len(pred)}")


if __name__ == "__main__":
    main()
