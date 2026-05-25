from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
SAMPLES = DATA / "monte_carlo_sensitivity_samples.csv"


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def standardize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    std = arr.std(ddof=0)
    return np.zeros_like(arr) if std == 0 else (arr - arr.mean()) / std


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_linear_importance(df: pd.DataFrame, output: str, predictors: list[str]) -> tuple[pd.DataFrame, float]:
    y = standardize(df[output].to_numpy(float))
    x = np.column_stack([np.ones(len(df)), *[standardize(df[p].to_numpy(float)) for p in predictors]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    base_r2 = r2_score(y, x @ beta)
    rows: list[dict] = []
    for idx, predictor in enumerate(predictors, start=1):
        x_perm = x.copy()
        x_perm[:, idx] = np.roll(x_perm[:, idx], len(df) // 3)
        perm_r2 = r2_score(y, x_perm @ beta)
        pearson = float(np.corrcoef(standardize(df[predictor]), y)[0, 1])
        spearman = float(
            np.corrcoef(
                standardize(rankdata(df[predictor].to_numpy(float))),
                standardize(rankdata(df[output].to_numpy(float))),
            )[0, 1]
        )
        rows.append(
            {
                "output": output,
                "predictor": predictor,
                "standardized_linear_coefficient": float(beta[idx]),
                "absolute_standardized_coefficient": float(abs(beta[idx])),
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "base_r2": base_r2,
                "permuted_r2": perm_r2,
                "permutation_importance_delta_r2": float(max(0.0, base_r2 - perm_r2)),
                "claim_boundary": "Local identifiability screen around the declared benchmark perturbation range; not a global parameter-identification study.",
            }
        )
    return pd.DataFrame(rows), base_r2


def draw_importance(importance: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1850, 1010
    left, right, top, bottom = 515, 120, 105, 190
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(26, True)
    tick_font = font(22)
    d.text((left, 42), "Local identifiability screen from Monte Carlo perturbations", fill="#111111", font=title_font)
    rows = importance.copy()
    output_labels = {
        "final_plastic_strain_index": "plastic index",
        "stiffness_loss_pct": "stiffness loss",
        "final_seismic_damage": "damage",
    }
    predictor_labels = {"suction_amp_kpa": "suction amplitude", "cyclic_amp": "cyclic amplitude"}
    rows["label"] = rows["output"].map(output_labels).fillna(rows["output"]) + " / " + rows["predictor"].map(
        predictor_labels
    ).fillna(rows["predictor"])
    rows = rows.sort_values("permutation_importance_delta_r2", ascending=True)
    max_v = max(0.01, float(rows["permutation_importance_delta_r2"].max()) * 1.15)
    d.rectangle((left, top, width - right, height - bottom), outline="#222222", width=3)

    def xp(v: float) -> float:
        return left + (v / max_v) * (width - right - left)

    for xval in np.linspace(0, max_v, 6):
        x = xp(float(xval))
        d.line((x, top, x, height - bottom), fill="#eeeeee")
        d.text((x - 28, height - bottom + 16), f"{xval:.2f}", fill="#222222", font=tick_font)
    bar_h = 44
    gap = 20
    y_start = height - bottom - (len(rows) * (bar_h + gap)) + gap
    colors = {"final_plastic_strain_index": "#d94801", "stiffness_loss_pct": "#2171b5", "final_seismic_damage": "#756bb1"}
    for i, (_, row) in enumerate(rows.iterrows()):
        y = y_start + i * (bar_h + gap)
        color = colors.get(row["output"], "#4d4d4d")
        label = str(row["label"])
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((left - 24 - (bbox[2] - bbox[0]), y + 10), label, fill="#111111", font=tick_font)
        d.rectangle((left, y, xp(float(row["permutation_importance_delta_r2"])), y + bar_h), fill=color)
        d.text((xp(float(row["permutation_importance_delta_r2"])) + 8, y + 10), f"{row['permutation_importance_delta_r2']:.3f}", fill="#111111", font=tick_font)
    d.text((left + 160, height - 72), "Permutation importance, delta R2 under fixed fitted local model", fill="#111111", font=label_font)
    d.text((left, height - 122), "Scope: local perturbation screen for declared benchmark inputs; not global calibration.", fill="#333333", font=tick_font)
    img.save(FIGS / "fig18_parameter_identifiability_screen.png")


def main() -> None:
    if not SAMPLES.exists():
        raise FileNotFoundError(f"Missing {SAMPLES}; run hydro_validation_envelope.py first")
    df = pd.read_csv(SAMPLES)
    predictors = ["suction_amp_kpa", "cyclic_amp"]
    outputs = ["final_plastic_strain_index", "stiffness_loss_pct", "final_seismic_damage"]
    parts = []
    r2_rows = []
    for output in outputs:
        importance, base_r2 = fit_linear_importance(df, output, predictors)
        parts.append(importance)
        r2_rows.append({"output": output, "local_linear_r2": base_r2})
    out = pd.concat(parts, ignore_index=True).sort_values(
        ["output", "permutation_importance_delta_r2"], ascending=[True, False]
    )
    out.to_csv(DATA / "parameter_identifiability_sensitivity.csv", index=False)
    pd.DataFrame(r2_rows).to_csv(DATA / "parameter_identifiability_summary.csv", index=False)
    draw_importance(out)
    print(f"parameter_identifiability_sensitivity=ok outputs={len(outputs)} predictors={len(predictors)}")


if __name__ == "__main__":
    main()
