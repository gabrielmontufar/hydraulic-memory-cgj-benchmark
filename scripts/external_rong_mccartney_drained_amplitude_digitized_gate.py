from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_unsat_cyclic_benchmark import simulate_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RAW = ROOT / "external_data" / "rong_mccartney_unsaturated_cyclic" / "qt9wx3t712_noSplash_44f8caf288d18fc6f196549851c054b7.pdf"

MODELS = ["constant_suction", "no_hysteresis", "hysteresis_only", "full_hysteretic_damage"]


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def transcribed_points() -> pd.DataFrame:
    # Approximate figure-level values digitized from Rong (2021) / Rong and McCartney
    # drained CSS Figure 4.5. They are a trend gate only; no raw time histories are implied.
    rows = [
        (0.3, 0.0, 1.00, 0.10, "dry condition"),
        (0.3, 0.0, 0.56, 0.15, "psi=2 kPa"),
        (0.3, 4.0, 0.30, 0.20, "psi=4 kPa"),
        (0.3, 6.0, 0.20, 0.15, "psi=6 kPa"),
        (0.3, 10.0, 0.12, 0.10, "psi=10 kPa"),
        (1.0, 0.0, 1.00, 1.85, "dry condition"),
        (1.0, 0.0, 0.56, 1.75, "psi=2 kPa"),
        (1.0, 4.0, 0.30, 1.55, "psi=4 kPa"),
        (1.0, 6.0, 0.20, 1.65, "psi=6 kPa"),
        (1.0, 10.0, 0.12, 1.45, "psi=10 kPa"),
        (3.0, 0.0, 1.00, 5.20, "dry condition"),
        (3.0, 0.0, 0.56, 4.85, "psi=2 kPa"),
        (3.0, 4.0, 0.30, 5.05, "psi=4 kPa"),
        (3.0, 6.0, 0.20, 4.70, "psi=6 kPa"),
        (3.0, 10.0, 0.12, 4.10, "psi=10 kPa"),
        (5.0, 0.0, 1.00, 7.65, "dry condition"),
        (5.0, 0.0, 0.56, 7.85, "psi=2 kPa"),
        (5.0, 4.0, 0.30, 8.55, "psi=4 kPa"),
        (5.0, 6.0, 0.20, 7.20, "psi=6 kPa"),
        (5.0, 10.0, 0.12, 5.85, "psi=10 kPa"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "cyclic_shear_strain_amplitude_pct",
            "matric_suction_kpa",
            "initial_degree_saturation",
            "observed_volumetric_strain_200_cycles_pct",
            "source_series",
        ],
    )


def model_feature(model: str, cyclic_shear_strain_amplitude_pct: float) -> float:
    # The benchmark input is a dimensionless cyclic demand ratio, not direct CSS shear strain.
    # The monotone map below intentionally supports only a rank/trend gate.
    cyclic_ratio_proxy = 0.04 * float(cyclic_shear_strain_amplitude_pct)
    df = simulate_case(
        model,
        suction_amp=10.0,
        cyclic_amp=cyclic_ratio_proxy,
        n_steps=720,
        suction_mean=10.0,
    )
    return float(100.0 * df["plastic_strain_index"].iloc[-1])


def summarize(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        points.groupby("cyclic_shear_strain_amplitude_pct", as_index=False)
        .agg(
            observed_median_pct=("observed_volumetric_strain_200_cycles_pct", "median"),
            observed_min_pct=("observed_volumetric_strain_200_cycles_pct", "min"),
            observed_max_pct=("observed_volumetric_strain_200_cycles_pct", "max"),
            n_series=("observed_volumetric_strain_200_cycles_pct", "size"),
        )
        .sort_values("cyclic_shear_strain_amplitude_pct")
    )
    rows = []
    for model in MODELS:
        feature = grouped["cyclic_shear_strain_amplitude_pct"].map(lambda x: model_feature(model, float(x))).to_numpy(float)
        obs = grouped["observed_median_pct"].to_numpy(float)
        coeff = np.polyfit(feature, obs, 1)
        pred = np.polyval(coeff, feature)
        rows.append(
            {
                "source": "Rong/McCartney drained CSS Figure 4.5",
                "target": "volumetric_strain_200_cycles_vs_cyclic_shear_strain_amplitude",
                "model": model,
                "n_amplitude_groups": int(len(grouped)),
                "spearman_rank_correlation": float(pd.Series(obs).corr(pd.Series(feature), method="spearman")),
                "affine_rescaled_rmse_pct": float(np.sqrt(np.mean((pred - obs) ** 2))),
                "affine_rescaled_mae_pct": float(np.mean(np.abs(pred - obs))),
                "observed_monotonic_increase": bool(np.all(np.diff(obs) > 0.0)),
                "model_monotonic_increase": bool(np.all(np.diff(feature) > 0.0)),
                "claim_boundary": (
                    "Figure-level drained CSS amplitude gate with affine rescaling; "
                    "not raw time-history validation and not a calibration of cyclic shear strain."
                ),
            }
        )
        grouped[f"{model}_feature_pct"] = feature
    return grouped, pd.DataFrame(rows)


def draw(grouped: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1600, 980
    left, right, top, bottom = 210, 90, 115, 190
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, tick_font = font(35, True), font(27, True), font(22)
    plot = (left, top, width - right, height - bottom)
    d.rectangle(plot, outline="#222222", width=3)
    xmax = 5.2
    ymax = 9.2

    def xp(x: float) -> float:
        return left + (x / xmax) * (width - right - left)

    def yp(y: float) -> float:
        return height - bottom - (y / ymax) * (height - bottom - top)

    for x in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        px = xp(x)
        d.line((px, top, px, height - bottom), fill="#eeeeee")
        d.text((px - 12, height - bottom + 16), f"{x:g}", font=tick_font, fill="#222222")
    for y in np.linspace(0, 9, 10):
        py = yp(float(y))
        d.line((left, py, width - right, py), fill="#eeeeee")
        d.text((left - 58, py - 12), f"{y:.0f}", font=tick_font, fill="#222222")

    d.text((left, 42), "Rong/McCartney drained CSS amplitude gate", font=title_font, fill="#111111")
    d.text((560, height - 80), "Cyclic shear strain amplitude (%)", font=label_font, fill="#111111")
    y_label = Image.new("RGBA", (430, 44), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Volumetric strain after 200 cycles (%)", font=label_font, fill="#111111")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label, (50, 300), y_label)

    g = grouped.sort_values("cyclic_shear_strain_amplitude_pct")
    band = []
    for row in g.itertuples():
        band.append((xp(float(row.cyclic_shear_strain_amplitude_pct)), yp(float(row.observed_min_pct))))
    for row in reversed(list(g.itertuples())):
        band.append((xp(float(row.cyclic_shear_strain_amplitude_pct)), yp(float(row.observed_max_pct))))
    d.polygon(band, fill="#d9e8f6")
    obs_pts = [(xp(float(row.cyclic_shear_strain_amplitude_pct)), yp(float(row.observed_median_pct))) for row in g.itertuples()]
    d.line(obs_pts, fill="#114c8d", width=5)
    for px, py in obs_pts:
        d.ellipse((px - 10, py - 10, px + 10, py + 10), fill="#114c8d")

    full = summary[summary["model"] == "full_hysteretic_damage"].iloc[0]
    d.text((left + 45, top + 35), "Blue band: digitized Figure 4.5 series range", font=tick_font, fill="#111111")
    d.text((left + 45, top + 70), "Blue line: median external figure response", font=tick_font, fill="#111111")
    d.text(
        (left, height - 135),
        f"Full-model rank trend: rho={full.spearman_rank_correlation:.2f}; monotone gate={full.model_monotonic_increase}.",
        font=tick_font,
        fill="#222222",
    )
    d.text(
        (left, height - 103),
        "Scope: external figure-level amplitude trend; no raw CSS time histories or design calibration are claimed.",
        font=tick_font,
        fill="#222222",
    )
    img.save(FIGS / "fig23_rong_mccartney_drained_amplitude_gate.png")


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing source PDF: {RAW}")
    points = transcribed_points()
    points.to_csv(DATA / "rong_mccartney_drained_amplitude_digitized_points.csv", index=False)
    grouped, summary = summarize(points)
    grouped.to_csv(DATA / "rong_mccartney_drained_amplitude_digitized_group_summary.csv", index=False)
    summary.to_csv(DATA / "rong_mccartney_drained_amplitude_digitized_gate_summary.csv", index=False)
    draw(grouped, summary)
    print(f"rong_mccartney_drained_amplitude_gate=ok rows={len(points)} groups={len(grouped)}")


if __name__ == "__main__":
    main()
