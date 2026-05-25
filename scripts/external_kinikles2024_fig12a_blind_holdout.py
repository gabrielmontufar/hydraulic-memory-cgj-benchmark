from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"

POINTS = DATA / "kinikles2024_fig12a_digitized_points.csv"
ENVELOPE = DATA / "kinikles2024_fig12a_digitized_envelope_summary.csv"


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_predict_leave_one_group_out(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict] = []
    groups = sorted(df["group_initial_degree_saturation"].unique())
    for holdout in groups:
        train = df[df["group_initial_degree_saturation"] != holdout]
        test = df[df["group_initial_degree_saturation"] == holdout].iloc[0]
        x_train = train["group_initial_degree_saturation"].to_numpy(float)
        y_train = train[target].to_numpy(float)
        degree = 2
        coeff = np.polyfit(x_train, y_train, degree)
        pred = float(np.polyval(coeff, float(holdout)))
        obs = float(test[target])
        lo = float(test["volumetric_strain_min_pct"])
        hi = float(test["volumetric_strain_max_pct"])
        rows.append(
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "validation_type": "leave_one_saturation_group_out_blind_digitized_figure_holdout",
                "target": target,
                "holdout_group_initial_degree_saturation": float(holdout),
                "training_groups": ";".join(f"{g:.2f}" for g in groups if g != holdout),
                "model_form": "predeclared quadratic in initial degree of saturation, fitted only on training groups",
                "observed_value_pct": obs,
                "predicted_value_pct": pred,
                "absolute_error_pct": abs(pred - obs),
                "squared_error": (pred - obs) ** 2,
                "observed_group_min_pct": lo,
                "observed_group_max_pct": hi,
                "prediction_inside_digitized_group_range": bool(lo <= pred <= hi),
                "claim_boundary": "Blind holdout uses digitized figure points only; it is not raw time-history validation or soil-specific calibration.",
            }
        )
    return pd.DataFrame(rows)


def write_summary(holdout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target, group in holdout.groupby("target"):
        rows.append(
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "validation_type": "leave_one_saturation_group_out_blind_digitized_figure_holdout",
                "target": target,
                "n_holdout_groups": len(group),
                "rmse_pct": float(np.sqrt(group["squared_error"].mean())),
                "mae_pct": float(group["absolute_error_pct"].mean()),
                "max_abs_error_pct": float(group["absolute_error_pct"].max()),
                "range_coverage_fraction": float(group["prediction_inside_digitized_group_range"].mean()),
                "interpretation": (
                    "Median-envelope holdout is useful as a figure-level blind plausibility check; "
                    "max-envelope holdout is weaker at saturation extremes and must be reported as a limitation."
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "kinikles2024_fig12a_blind_holdout_summary.csv", index=False)
    return out


def draw_holdout(holdout: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    median = holdout[holdout["target"] == "volumetric_strain_median_pct"].copy()
    peak = holdout[holdout["target"] == "volumetric_strain_max_pct"].copy()

    width, height = 1750, 1050
    left, right, top, bottom = 235, 80, 115, 185
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = font(36, True)
    label_font = font(28, True)
    tick_font = font(23)
    note_font = font(23)

    plot = (left, top, width - right, height - bottom)
    d.rectangle(plot, outline="#222222", width=3)

    def xp(x: float) -> float:
        return left + (x / 0.6) * (width - right - left)

    def yp(y: float) -> float:
        return height - bottom - (y / 3.0) * (height - bottom - top)

    for x in np.linspace(0, 0.6, 7):
        px = xp(float(x))
        d.line((px, top, px, height - bottom), fill="#eeeeee", width=1)
        d.text((px - 20, height - bottom + 18), f"{x:.1f}", font=tick_font, fill="#222222")
    for y in np.linspace(0, 3.0, 7):
        py = yp(float(y))
        d.line((left, py, width - right, py), fill="#eeeeee", width=1)
        d.text((left - 70, py - 13), f"{y:.1f}", font=tick_font, fill="#222222")

    d.text((left, 42), "Blind leave-one-saturation-group-out check on digitized Kinikles et al. Fig. 12(a)", font=title_font, fill="#111111")
    d.text((610, height - 80), "Holdout initial degree of saturation, S0", font=label_font, fill="#111111")
    y_label = Image.new("RGBA", (380, 44), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Volumetric strain (%)", font=label_font, fill="#111111")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label, (55, 360), y_label)

    for _, row in peak.iterrows():
        x = float(row["holdout_group_initial_degree_saturation"])
        lo = float(row["observed_group_min_pct"])
        hi = float(row["observed_group_max_pct"])
        px = xp(x)
        d.line((px, yp(lo), px, yp(hi)), fill="#b9c2ce", width=8)

    for _, row in median.iterrows():
        x = float(row["holdout_group_initial_degree_saturation"])
        obs = float(row["observed_value_pct"])
        pred = float(row["predicted_value_pct"])
        px = xp(x)
        d.ellipse((px - 9, yp(obs) - 9, px + 9, yp(obs) + 9), fill="#114c8d", outline="#114c8d")
        d.rectangle((px - 8, yp(pred) - 8, px + 8, yp(pred) + 8), fill="#d94801", outline="#d94801")
        d.line((px, yp(obs), px, yp(pred)), fill="#555555", width=2)

    for _, row in peak.iterrows():
        x = float(row["holdout_group_initial_degree_saturation"])
        pred = float(row["predicted_value_pct"])
        px = xp(x)
        d.polygon([(px, yp(pred) - 11), (px - 10, yp(pred) + 7), (px + 10, yp(pred) + 7)], fill="#6a1b9a")

    d.rectangle((left + 40, top + 35, left + 720, top + 175), fill="#ffffff", outline="#777777")
    d.ellipse((left + 65, top + 63, left + 83, top + 81), fill="#114c8d")
    d.text((left + 100, top + 56), "Observed holdout median", font=note_font, fill="#111111")
    d.rectangle((left + 65, top + 101, left + 83, top + 119), fill="#d94801")
    d.text((left + 100, top + 94), "Blind quadratic prediction of median", font=note_font, fill="#111111")
    d.polygon([(left + 74, top + 145), (left + 63, top + 164), (left + 85, top + 164)], fill="#6a1b9a")
    d.text((left + 100, top + 134), "Blind quadratic prediction of max envelope", font=note_font, fill="#111111")

    sm = {row["target"]: row for _, row in summary.iterrows()}
    txt = (
        f"Median holdout: RMSE={sm['volumetric_strain_median_pct']['rmse_pct']:.3f}%, "
        f"MAE={sm['volumetric_strain_median_pct']['mae_pct']:.3f}%, "
        f"range coverage={sm['volumetric_strain_median_pct']['range_coverage_fraction']:.2f}. "
        f"Max holdout: RMSE={sm['volumetric_strain_max_pct']['rmse_pct']:.3f}%, "
        f"coverage={sm['volumetric_strain_max_pct']['range_coverage_fraction']:.2f}."
    )
    d.text((left, height - 132), txt, font=note_font, fill="#222222")
    d.text(
        (left, height - 100),
        "Scope: blind figure-level holdout from digitized points; not raw cyclic time-history validation or site-design calibration.",
        font=note_font,
        fill="#222222",
    )
    img.save(FIGS / "fig15_kinikles2024_blind_holdout.png")


def main() -> None:
    if not ENVELOPE.exists():
        raise FileNotFoundError(f"Missing {ENVELOPE}; run external_kinikles2024_fig12a_digitized_envelope.py first")
    envelope = pd.read_csv(ENVELOPE)
    holdout = pd.concat(
        [
            fit_predict_leave_one_group_out(envelope, "volumetric_strain_median_pct"),
            fit_predict_leave_one_group_out(envelope, "volumetric_strain_max_pct"),
        ],
        ignore_index=True,
    )
    holdout.to_csv(DATA / "kinikles2024_fig12a_blind_holdout_predictions.csv", index=False)
    summary = write_summary(holdout)
    draw_holdout(holdout, summary)
    print(
        "kinikles2024_fig12a_blind_holdout=ok "
        f"targets={summary['target'].nunique()} groups={int(summary['n_holdout_groups'].max())}"
    )


if __name__ == "__main__":
    main()
