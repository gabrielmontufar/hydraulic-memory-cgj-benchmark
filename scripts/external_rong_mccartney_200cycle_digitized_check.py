from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_unsat_cyclic_benchmark import Params, simulate_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RAW = ROOT / "external_data" / "rong_mccartney_unsaturated_cyclic" / "2022_05_mccartney_final.pdf"


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def transcribed_points() -> pd.DataFrame:
    # Figure-level values transcribed from Rong/McCartney PEER 2022/05 figures 4.3-4.16 and
    # Section 5.4 text. They are used only as an independent 200-cycle trend gate, not as raw
    # time-history validation.
    rows = [
        (0.000, 2.60, "Figure 4.15(d)", "dry specimen, 200 cycles"),
        (0.117, 1.08, "Figure 4.14(d)", "lowest funicular saturation, 200 cycles"),
        (0.206, 1.32, "Figure 4.13(d)", "low funicular saturation, 200 cycles"),
        (0.300, 1.55, "Figure 4.11(d)", "intermediate funicular saturation, 200 cycles"),
        (0.400, 1.38, "Figure 4.9(d)", "intermediate-high funicular saturation, 200 cycles"),
        (0.560, 0.78, "Figure 4.7(d)", "higher funicular saturation, 200 cycles"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "initial_degree_saturation",
            "observed_volumetric_strain_200_cycles_pct",
            "source_figure",
            "transcription_note",
        ],
    )


def model_proxy(points: pd.DataFrame) -> pd.DataFrame:
    p = Params()

    def suction_for_initial_saturation(sr0: float) -> float:
        sr = max(sr0, p.sr_res + 1.0e-4)
        se = np.clip((sr - p.sr_res) / (p.sr_sat - p.sr_res), 1.0e-4, 0.9999)
        m = 1.0 - 1.0 / p.n_dry
        return float(((se ** (-1.0 / m) - 1.0) ** (1.0 / p.n_dry)) / p.alpha_dry)

    pred: list[float] = []
    paired_suction: list[float] = []
    paired_initial_sr: list[float] = []
    for sr0 in points["initial_degree_saturation"].to_numpy(float):
        suction = max(0.5, suction_for_initial_saturation(sr0))
        paired_suction.append(suction)
        df = simulate_case(
            "full_hysteretic_damage",
            suction_amp=min(100.0, suction),
            cyclic_amp=0.20,
            n_steps=720,
            suction_mean=suction,
        )
        paired_initial_sr.append(float(df["degree_saturation"].iloc[0]))
        pred.append(float(100.0 * df["plastic_strain_index"].iloc[-1]))
    out = points.copy()
    out["benchmark_proxy_source"] = "full_hysteretic_damage paired by initial saturation through the benchmark retention curve"
    out["paired_model_initial_degree_saturation"] = paired_initial_sr
    out["paired_model_suction_kpa"] = paired_suction
    out["benchmark_proxy_value_pct"] = pred
    out["absolute_error_pct"] = (out["benchmark_proxy_value_pct"] - out["observed_volumetric_strain_200_cycles_pct"]).abs()
    out["squared_error"] = out["absolute_error_pct"] ** 2
    out["claim_boundary"] = (
        "Independent Rong/McCartney 200-cycle figure-level trend gate; "
        "not raw cyclic time-history calibration or a soil-specific inverse fit."
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    obs = df["observed_volumetric_strain_200_cycles_pct"].to_numpy(float)
    pred = df["benchmark_proxy_value_pct"].to_numpy(float)
    if len(obs) >= 2:
        corr = float(pd.Series(obs).corr(pd.Series(pred), method="spearman"))
    else:
        corr = np.nan
    summary = pd.DataFrame(
        [
            {
                "source": "Rong/McCartney PEER 2022/05 Figure-level 200-cycle volumetric response",
                "validation_type": "independent_200_cycle_figure_trend_gate",
                "n_digitized_groups": int(len(df)),
                "rmse_pct": float(np.sqrt(df["squared_error"].mean())),
                "mae_pct": float(df["absolute_error_pct"].mean()),
                "max_abs_error_pct": float(df["absolute_error_pct"].max()),
                "spearman_rank_correlation": corr,
                "observed_range_pct": float(obs.max() - obs.min()),
                "model_proxy_range_pct": float(pred.max() - pred.min()),
                "interpretation": (
                    "Provides an independent Rong/McCartney 200-cycle figure-level check. "
                    "Agreement should be read as trend/envelope evidence only."
                ),
            }
        ]
    )
    return summary


def draw(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1650, 980
    left, right, top, bottom = 210, 90, 115, 190
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, tick_font = font(35, True), font(27, True), font(22)
    plot = (left, top, width - right, height - bottom)
    d.rectangle(plot, outline="#222222", width=3)

    ymax = max(3.0, float(df[["observed_volumetric_strain_200_cycles_pct", "benchmark_proxy_value_pct"]].max().max()) * 1.15)

    def xp(x: float) -> float:
        return left + (x / 0.6) * (width - right - left)

    def yp(y: float) -> float:
        return height - bottom - (y / ymax) * (height - bottom - top)

    for x in np.linspace(0, 0.6, 7):
        px = xp(float(x))
        d.line((px, top, px, height - bottom), fill="#eeeeee")
        d.text((px - 18, height - bottom + 16), f"{x:.1f}", font=tick_font, fill="#222222")
    for y in np.linspace(0, ymax, 7):
        py = yp(float(y))
        d.line((left, py, width - right, py), fill="#eeeeee")
        d.text((left - 72, py - 12), f"{y:.1f}", font=tick_font, fill="#222222")

    d.text((left, 42), "Independent Rong/McCartney 200-cycle figure-level validation gate", font=title_font, fill="#111111")
    d.text((585, height - 80), "Initial degree of saturation, S0", font=label_font, fill="#111111")
    y_label = Image.new("RGBA", (430, 44), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Volumetric strain after 200 cycles (%)", font=label_font, fill="#111111")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label, (50, 300), y_label)

    df = df.sort_values("initial_degree_saturation")
    obs_pts = [(xp(float(r.initial_degree_saturation)), yp(float(r.observed_volumetric_strain_200_cycles_pct))) for r in df.itertuples()]
    pred_pts = [(xp(float(r.initial_degree_saturation)), yp(float(r.benchmark_proxy_value_pct))) for r in df.itertuples()]
    d.line(obs_pts, fill="#114c8d", width=5)
    d.line(pred_pts, fill="#d94801", width=5)
    for (px, py) in obs_pts:
        d.ellipse((px - 10, py - 10, px + 10, py + 10), fill="#114c8d")
    for (px, py) in pred_pts:
        d.rectangle((px - 9, py - 9, px + 9, py + 9), fill="#d94801")

    sx = left + 45
    sy = top + 35
    d.rectangle((sx - 20, sy - 20, sx + 690, sy + 115), fill="#ffffff", outline="#777777")
    d.ellipse((sx, sy, sx + 20, sy + 20), fill="#114c8d")
    d.text((sx + 36, sy - 4), "Rong/McCartney figure-level transcription", font=tick_font, fill="#111111")
    d.rectangle((sx, sy + 42, sx + 20, sy + 62), fill="#d94801")
    d.text((sx + 36, sy + 37), "Benchmark full-model proxy, same ordered gate", font=tick_font, fill="#111111")

    s = summary.iloc[0]
    d.text(
        (left, height - 135),
        f"RMSE={s.rmse_pct:.3f}%, MAE={s.mae_pct:.3f}%, Spearman rho={s.spearman_rank_correlation:.2f}.",
        font=tick_font,
        fill="#222222",
    )
    d.text(
        (left, height - 103),
        "Scope: external figure-level 200-cycle trend gate; not raw time-history validation or design calibration.",
        font=tick_font,
        fill="#222222",
    )
    img.save(FIGS / "fig22_rong_mccartney_200cycle_digitized_check.png")


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing source PDF: {RAW}")
    points = transcribed_points()
    points.to_csv(DATA / "rong_mccartney_200cycle_transcribed_points.csv", index=False)
    check = model_proxy(points)
    check.to_csv(DATA / "rong_mccartney_digitized_200cycle_validation.csv", index=False)
    summary = summarize(check)
    summary.to_csv(DATA / "rong_mccartney_digitized_200cycle_validation_summary.csv", index=False)
    draw(check, summary)
    print(f"rong_mccartney_200cycle_digitized_check=ok rows={len(check)}")


if __name__ == "__main__":
    main()
