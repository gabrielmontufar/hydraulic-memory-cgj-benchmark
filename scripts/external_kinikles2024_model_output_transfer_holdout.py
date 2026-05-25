from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_unsat_cyclic_benchmark import Params, simulate_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
ENVELOPE = DATA / "kinikles2024_fig12a_digitized_envelope_summary.csv"


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def suction_for_initial_saturation(sr0: float) -> float:
    p = Params()
    se = np.clip((sr0 - p.sr_res) / (p.sr_sat - p.sr_res), 1.0e-4, 0.9999)
    m = 1.0 - 1.0 / p.n_dry
    return float(((se ** (-1.0 / m) - 1.0) ** (1.0 / p.n_dry)) / p.alpha_dry)


def model_features(sr0: float, suction_amp: float = 25.0, cyclic_amp: float = 0.20) -> dict:
    target_suction_at_first_step = suction_for_initial_saturation(sr0)
    suction_mean = target_suction_at_first_step + suction_amp * math.sqrt(0.5)
    df = simulate_case(
        "full_hysteretic_damage",
        suction_amp=suction_amp,
        cyclic_amp=cyclic_amp,
        n_steps=720,
        suction_mean=suction_mean,
    )
    stiffness_loss = 100.0 * (1.0 - df["secant_stiffness_mpa"].iloc[-1] / df["secant_stiffness_mpa"].iloc[0])
    return {
        "target_initial_degree_saturation": sr0,
        "model_initial_degree_saturation": float(df["degree_saturation"].iloc[0]),
        "model_suction_mean_kpa": float(suction_mean),
        "model_final_plastic_strain_index": float(df["plastic_strain_index"].iloc[-1]),
        "model_stiffness_loss_pct": float(stiffness_loss),
        "model_final_seismic_damage": float(df["seismic_damage"].iloc[-1]),
    }


def leave_one_out_transfer(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict] = []
    groups = sorted(df["group_initial_degree_saturation"].unique())
    for holdout in groups:
        train = df[df["group_initial_degree_saturation"] != holdout]
        test = df[df["group_initial_degree_saturation"] == holdout].iloc[0]
        x_train = train["model_final_plastic_strain_index"].to_numpy(float)
        y_train = train[target].to_numpy(float)
        coeff = np.polyfit(x_train, y_train, 2)
        pred = float(np.polyval(coeff, float(test["model_final_plastic_strain_index"])))
        obs = float(test[target])
        lo = float(test["volumetric_strain_min_pct"])
        hi = float(test["volumetric_strain_max_pct"])
        rows.append(
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "validation_type": "leave_one_saturation_group_out_model_output_transfer_holdout",
                "target": target,
                "holdout_group_initial_degree_saturation": float(holdout),
                "training_groups": ";".join(f"{g:.2f}" for g in groups if g != holdout),
                "model_connection": "full_hysteretic_damage benchmark output at matched initial saturation",
                "transfer_model": "quadratic transfer from benchmark final plastic-strain index fitted only on training groups",
                "model_final_plastic_strain_index": float(test["model_final_plastic_strain_index"]),
                "model_stiffness_loss_pct": float(test["model_stiffness_loss_pct"]),
                "observed_value_pct": obs,
                "predicted_value_pct": pred,
                "absolute_error_pct": abs(pred - obs),
                "squared_error": (pred - obs) ** 2,
                "observed_group_min_pct": lo,
                "observed_group_max_pct": hi,
                "prediction_inside_digitized_group_range": bool(lo <= pred <= hi),
                "claim_boundary": "Model-output transfer holdout links the benchmark response to a digitized external envelope; it is not raw time-history calibration.",
            }
        )
    return pd.DataFrame(rows)


def summarize(holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in holdout.groupby("target"):
        rows.append(
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "validation_type": "leave_one_saturation_group_out_model_output_transfer_holdout",
                "target": target,
                "n_holdout_groups": len(group),
                "rmse_pct": float(np.sqrt(group["squared_error"].mean())),
                "mae_pct": float(group["absolute_error_pct"].mean()),
                "max_abs_error_pct": float(group["absolute_error_pct"].max()),
                "range_coverage_fraction": float(group["prediction_inside_digitized_group_range"].mean()),
                "interpretation": "Directly connects benchmark output to the digitized external envelope while retaining the figure-level limitation.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "kinikles2024_model_output_transfer_holdout_summary.csv", index=False)
    return out


def draw(holdout: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    med = holdout[holdout["target"] == "volumetric_strain_median_pct"].copy()
    peak = holdout[holdout["target"] == "volumetric_strain_max_pct"].copy()
    width, height = 1750, 1120
    left, right, top, bottom = 230, 90, 120, 270
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, tick_font = font(35, True), font(27, True), font(23)
    d.rectangle((left, top, width - right, height - bottom), outline="#222222", width=3)

    def xp(x: float) -> float:
        return left + (x / 0.6) * (width - right - left)

    def yp(y: float) -> float:
        return height - bottom - (y / 3.0) * (height - bottom - top)

    for x in np.linspace(0, 0.6, 7):
        px = xp(float(x))
        d.line((px, top, px, height - bottom), fill="#eeeeee")
        d.text((px - 18, height - bottom + 16), f"{x:.1f}", font=tick_font, fill="#222222")
    for y in np.linspace(0, 3.0, 7):
        py = yp(float(y))
        d.line((left, py, width - right, py), fill="#eeeeee")
        d.text((left - 72, py - 12), f"{y:.1f}", font=tick_font, fill="#222222")

    d.text((left, 42), "Model-output transfer holdout on digitized Kinikles et al. Fig. 12(a)", font=title_font, fill="#111111")
    d.text((610, height - 180), "Holdout initial degree of saturation, S0", font=label_font, fill="#111111")
    y_label = Image.new("RGBA", (380, 44), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Volumetric strain (%)", font=label_font, fill="#111111")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label, (55, 360), y_label)

    for _, row in med.iterrows():
        x = float(row["holdout_group_initial_degree_saturation"])
        px = xp(x)
        obs = float(row["observed_value_pct"])
        pred = float(row["predicted_value_pct"])
        d.ellipse((px - 10, yp(obs) - 10, px + 10, yp(obs) + 10), fill="#114c8d")
        d.rectangle((px - 9, yp(pred) - 9, px + 9, yp(pred) + 9), fill="#d94801")
        d.line((px, yp(obs), px, yp(pred)), fill="#444444", width=2)
    for _, row in peak.iterrows():
        x = float(row["holdout_group_initial_degree_saturation"])
        px = xp(x)
        pred = float(row["predicted_value_pct"])
        d.polygon([(px, yp(pred) - 11), (px - 10, yp(pred) + 8), (px + 10, yp(pred) + 8)], fill="#6a1b9a")

    legend_x, legend_y = left + 36, top + 34
    d.rectangle((legend_x - 18, legend_y - 18, legend_x + 625, legend_y + 120), fill="#ffffff", outline="#777777")
    d.ellipse((legend_x, legend_y, legend_x + 20, legend_y + 20), fill="#114c8d")
    d.text((legend_x + 34, legend_y - 3), "Observed holdout median", font=tick_font, fill="#111111")
    d.rectangle((legend_x, legend_y + 40, legend_x + 20, legend_y + 60), fill="#d94801")
    d.text((legend_x + 34, legend_y + 36), "Model-output transfer prediction, median", font=tick_font, fill="#111111")
    d.polygon([(legend_x + 10, legend_y + 80), (legend_x, legend_y + 100), (legend_x + 20, legend_y + 100)], fill="#6a1b9a")
    d.text((legend_x + 34, legend_y + 76), "Model-output transfer prediction, max envelope", font=tick_font, fill="#111111")

    sm = {row["target"]: row for _, row in summary.iterrows()}
    txt = (
        f"Median transfer holdout: RMSE={sm['volumetric_strain_median_pct']['rmse_pct']:.3f}%, "
        f"MAE={sm['volumetric_strain_median_pct']['mae_pct']:.3f}%. "
        f"Max transfer holdout: RMSE={sm['volumetric_strain_max_pct']['rmse_pct']:.3f}%, "
        f"MAE={sm['volumetric_strain_max_pct']['mae_pct']:.3f}%."
    )
    d.text((left, height - 118), txt, font=tick_font, fill="#222222")
    d.text((left, height - 84), "Scope: benchmark-output transfer to a digitized figure envelope; not raw cyclic time-history validation.", font=tick_font, fill="#222222")
    img.save(FIGS / "fig19_kinikles_model_output_transfer_holdout.png")


def main() -> None:
    if not ENVELOPE.exists():
        raise FileNotFoundError(f"Missing {ENVELOPE}; run external_kinikles2024_fig12a_digitized_envelope.py first")
    envelope = pd.read_csv(ENVELOPE)
    features = pd.DataFrame([model_features(float(sr)) for sr in envelope["group_initial_degree_saturation"]])
    merged = envelope.merge(
        features,
        left_on="group_initial_degree_saturation",
        right_on="target_initial_degree_saturation",
        how="left",
    )
    merged.to_csv(DATA / "kinikles2024_model_output_transfer_features.csv", index=False)
    holdout = pd.concat(
        [
            leave_one_out_transfer(merged, "volumetric_strain_median_pct"),
            leave_one_out_transfer(merged, "volumetric_strain_max_pct"),
        ],
        ignore_index=True,
    )
    holdout.to_csv(DATA / "kinikles2024_model_output_transfer_holdout_predictions.csv", index=False)
    summary = summarize(holdout)
    draw(holdout, summary)
    print(f"kinikles2024_model_output_transfer_holdout=ok targets={summary['target'].nunique()} groups={int(summary['n_holdout_groups'].max())}")


if __name__ == "__main__":
    main()
