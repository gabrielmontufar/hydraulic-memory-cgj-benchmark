from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "external_data" / "gabr_2019_unsaturated_residual_soil_triaxial" / "Binder1.pdf"
SOURCE_URL = "https://data.mendeley.com/datasets/p9tmzckdpt/1"
DOI = "10.17632/p9tmzckdpt.1"


def rows() -> list[dict[str, object]]:
    # Manual transcription from text-extracted Tables F-1, F-2 and F-3 in Binder1.pdf.
    # Columns in the source table are interpreted as net confining pressure, matric suction,
    # peak deviator stress, dry density and degree of saturation.
    f1 = [
        ("ST-36", "single_stage", "F-1", "0.5:1 slope", "MH", 27.6, 152.0, 264.5, 1.27, 0.60),
        ("ST-91", "single_stage", "F-1", "0.5:1 slope", "MH", 69.0, 27.6, 187.5, 1.30, 0.92),
        ("ST-40", "single_stage", "F-1", "0.5:1 slope", "MH", 34.5, 34.5, 160.0, 1.16, 0.75),
        ("ST-42", "single_stage", "F-1", "0.5:1 slope", "MH", 75.8, 20.7, 200.5, 1.32, 0.98),
        ("ST-43", "single_stage", "F-1", "0.5:1 slope", "ML", 103.4, 13.8, 388.9, 1.67, 0.98),
        ("ST-94", "single_stage", "F-1", "0.5:1 slope", "ML", 113.8, 37.9, 433.7, 1.36, 0.94),
        ("ST-47", "single_stage", "F-1", "0.25:1 slope", "MH", 82.7, 89.6, 292.8, 1.09, 0.65),
        ("ST-124", "single_stage", "F-1", "0.25:1 slope", "MH", 137.9, 62.1, 353.0, 1.20, 0.81),
        ("ST-50", "single_stage", "F-1", "0.25:1 slope", "MH", 41.4, 62.1, 197.2, 1.11, 0.71),
        ("ST-53", "single_stage", "F-1", "0.25:1 slope", "MH", 82.7, 27.6, 202.7, 1.17, 0.92),
        ("ST-54", "single_stage", "F-1", "0.25:1 slope", "ML", 93.1, 24.1, 239.9, 1.36, 0.94),
        ("ST-59B", "single_stage", "F-1", "1:1 slope", "ML", 20.7, 72.4, 175.1, 1.74, 0.52),
        ("ST-60", "single_stage", "F-1", "1:1 slope", "ML", 20.7, 55.2, 175.1, 1.82, 0.61),
        ("ST-114B", "single_stage", "F-1", "1:1 slope", "ML", 82.7, 55.2, 336.5, 1.72, 0.63),
        ("ST-116E", "single_stage", "F-1", "1:1 slope", "ML", 137.9, 10.3, 392.3, 1.76, 0.79),
        ("ST-65A", "single_stage", "F-1", "sheet pile", "ML", 82.7, 31.0, 388.2, 1.67, 0.98),
        ("ST-65B", "single_stage", "F-1", "sheet pile", "ML", 100.0, 31.0, 432.3, 1.71, 0.99),
        ("ST-75", "single_stage", "F-1", "sheet pile", "SC", 75.8, 27.6, 281.3, 1.63, 0.88),
        ("ST-77", "single_stage", "F-1", "sheet pile", "SC", 69.0, 55.2, 315.8, 1.61, 0.85),
        ("ST-85", "single_stage", "F-1", "sheet pile", "SC", 103.4, 27.6, 330.3, 1.53, 0.83),
    ]
    f2_values = [
        (20.7, 82.7, 177.4, 1.19, 0.54), (20.7, 124.1, 195.6, 1.21, 0.49),
        (20.7, 93.1, 177.2, 1.08, 0.65), (20.7, 124.1, 197.2, 1.09, 0.60),
        (27.6, 75.8, 186.5, 1.15, 0.63), (27.6, 137.9, 180.1, 1.16, 0.59),
        (27.6, 20.7, 103.4, 1.08, 0.91), (27.6, 89.6, 195.1, 1.09, 0.65),
        (20.7, 55.2, 146.1, 1.08, 0.67), (20.7, 82.7, 166.2, 1.09, 0.61),
        (34.5, 55.2, 143.5, 1.10, 0.66), (34.5, 106.9, 207.3, 1.11, 0.59),
        (24.1, 75.8, 168.2, 1.13, 0.65), (24.1, 120.7, 202.7, 1.13, 0.62),
        (24.1, 27.6, 111.7, 1.12, 0.90), (24.1, 89.6, 169.6, 1.12, 0.69),
        (24.1, 137.9, 206.2, 1.13, 0.58), (41.4, 44.8, 159.3, 1.08, 0.76),
        (41.4, 89.6, 200.0, 1.09, 0.72), (62.1, 31.0, 215.5, 1.19, 0.91),
        (62.1, 62.1, 266.3, 1.20, 0.86), (131.0, 10.3, 318.4, 1.69, 0.90),
        (131.0, 41.4, 366.5, 1.69, 0.79), (131.0, 62.1, 377.8, 1.70, 0.70),
        (20.7, 20.7, 136.5, 1.83, 0.61), (20.7, 137.9, 252.4, 1.83, 0.43),
        (20.7, 10.3, 89.5, 1.79, 0.69), (20.7, 103.4, 206.2, 1.79, 0.58),
        (41.4, 41.4, 174.4, 1.72, 0.81), (41.4, 69.0, 218.6, 1.73, 0.78),
        (75.8, 34.5, 237.9, 1.68, 0.80), (75.8, 69.0, 257.9, 1.69, 0.70),
    ]
    f3_values = [
        (20.7, 96.5, 150.1, 1.24, 0.42), (20.7, 117.2, 163.6, 1.25, 0.39),
        (41.4, 117.2, 199.6, 1.26, 0.39), (62.1, 117.2, 236.9, 1.26, 0.39),
        (48.3, 6.9, 133.6, 1.40, 0.84), (48.3, 48.3, 196.2, 1.40, 0.70),
        (96.5, 48.3, 311.9, 1.41, 0.71), (41.4, 75.8, 161.7, 1.43, 0.49),
        (41.4, 103.4, 190.9, 1.43, 0.44), (41.4, 124.1, 217.4, 1.43, 0.41),
        (62.1, 55.2, 272.0, 1.73, 0.71), (62.1, 82.7, 321.7, 1.73, 0.69),
        (117.2, 20.7, 318.0, 1.55, 0.89), (117.2, 55.2, 388.2, 1.55, 0.67),
        (117.2, 75.8, 443.3, 1.56, 0.62), (89.6, 6.9, 253.3, 1.56, 0.81),
        (89.6, 27.6, 316.1, 1.56, 0.80), (89.6, 55.2, 351.2, 1.56, 0.73),
        (44.8, 6.9, 173.4, 1.54, 0.78), (89.6, 6.9, 254.0, 1.57, 0.77),
        (89.6, 27.6, 288.9, 1.57, 0.75), (89.6, 27.6, 319.5, 1.71, 0.82),
        (62.1, 56.2, 357.2, 1.71, 0.77), (62.1, 82.7, 383.5, 1.72, 0.71),
        (144.8, 13.8, 288.6, 1.63, 0.90), (144.8, 41.4, 374.9, 1.64, 0.83),
        (131.0, 89.6, 469.1, 1.64, 0.63), (144.8, 27.6, 409.1, 1.66, 0.87),
        (144.8, 69.0, 500.6, 1.68, 0.65), (144.8, 96.5, 550.9, 1.68, 0.56),
        (69.0, 6.9, 234.3, 1.66, 0.89), (124.1, 6.9, 297.9, 1.66, 0.89),
        (124.1, 55.2, 389.9, 1.67, 0.72), (165.5, 55.2, 468.8, 1.68, 0.72),
    ]

    out = []
    for sample, test_type, table, area, soil, conf, suction, peak, rho, sat in f1:
        out.append(pack(sample, test_type, table, area, soil, conf, suction, peak, rho, sat, "holdout_single_stage"))
    for i, (conf, suction, peak, rho, sat) in enumerate(f2_values, start=1):
        out.append(pack(f"F2-{i:02d}", "multi_stage", "F-2", "slope area", "mixed", conf, suction, peak, rho, sat, "train_multistage"))
    for i, (conf, suction, peak, rho, sat) in enumerate(f3_values, start=1):
        out.append(pack(f"F3-{i:02d}", "multi_stage", "F-3", "sheet pile", "ML", conf, suction, peak, rho, sat, "train_multistage"))
    return out


def pack(sample, test_type, source_table, area, soil_type, conf, suction, peak, rho, saturation, split):
    return {
        "source": "Gabr et al. Mendeley Data p9tmzckdpt",
        "source_url": SOURCE_URL,
        "doi": DOI,
        "license": "CC0 1.0",
        "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "sample": sample,
        "test_type": test_type,
        "source_table": source_table,
        "area": area,
        "soil_type": soil_type,
        "net_confining_pressure_kpa": conf,
        "matric_suction_kpa": suction,
        "peak_deviator_stress_kpa": peak,
        "dry_density_g_cm3": rho,
        "degree_saturation": saturation,
        "split": split,
    }


def fit_linear(train: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in cols])
    y = train["peak_deviator_stress_kpa"].to_numpy(float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict(df: pd.DataFrame, cols: list[str], beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    return x @ beta


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    obs = df["peak_deviator_stress_kpa"].to_numpy(float)
    err = pred - obs
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "n": int(len(df)),
        "rmse_kpa": float(np.sqrt(np.mean(err**2))),
        "mae_kpa": float(np.mean(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err) / obs) * 100.0),
        "bias_kpa": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
    }


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing downloaded source PDF: {RAW}")
    DATA.mkdir(exist_ok=True)
    df = pd.DataFrame(rows())
    df.to_csv(DATA / "gabr2019_unsaturated_triaxial_transcribed.csv", index=False)

    cols = ["net_confining_pressure_kpa", "matric_suction_kpa", "dry_density_g_cm3"]
    train = df[df["split"] == "train_multistage"].copy()
    holdout = df[df["split"] == "holdout_single_stage"].copy()
    beta = fit_linear(train, cols)
    train_pred = predict(train, cols, beta)
    holdout_pred = predict(holdout, cols, beta)
    train["predicted_peak_deviator_stress_kpa"] = train_pred
    holdout["predicted_peak_deviator_stress_kpa"] = holdout_pred
    pred = pd.concat([train, holdout], ignore_index=True)
    pred["residual_kpa"] = pred["predicted_peak_deviator_stress_kpa"] - pred["peak_deviator_stress_kpa"]
    pred.to_csv(DATA / "gabr2019_unsaturated_triaxial_holdout_predictions.csv", index=False)

    summary = {
        "source": "Gabr et al. Mendeley Data p9tmzckdpt",
        "source_url": SOURCE_URL,
        "doi": DOI,
        "license": "CC0 1.0",
        "calibration_scope": "fit simple peak-strength surrogate on multistage tests and hold out all single-stage tests",
        "features": "net confining pressure, matric suction, dry density",
        "intercept": float(beta[0]),
        "coef_net_confining_kpa": float(beta[1]),
        "coef_matric_suction_kpa": float(beta[2]),
        "coef_dry_density_g_cm3": float(beta[3]),
        "suction_coefficient_positive": bool(beta[2] > 0),
        "train_metrics": metrics(train, train_pred),
        "holdout_metrics": metrics(holdout, holdout_pred),
        "claim_boundary": "independent static unsaturated triaxial holdout for suction-strength trend only; not cyclic validation",
    }
    pd.json_normalize(summary).to_csv(DATA / "gabr2019_unsaturated_triaxial_holdout_summary.csv", index=False)
    print(f"gabr2019_unsaturated_triaxial_holdout=ok rows={len(df)} train={len(train)} holdout={len(holdout)}")


if __name__ == "__main__":
    main()
