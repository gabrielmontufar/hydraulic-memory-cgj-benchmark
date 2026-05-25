from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"

SUMMARY = DATA / "benchmark_summary.csv"


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_model_metrics() -> pd.DataFrame:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing {SUMMARY}; run run_unsat_cyclic_benchmark.py first")
    df = pd.read_csv(SUMMARY)
    target = df[(df["suction_amp"] == 100.0) & (df["cyclic_amp"] == 0.20)].copy()
    if target.empty:
        raise ValueError("Required benchmark case suction=100 kPa, cyclic_strain=0.20 not found")
    return target


def layer_inputs() -> pd.DataFrame:
    depths = np.arange(0.5, 10.5, 1.0)
    rows = []
    for z in depths:
        rows.append(
            {
                "layer_id": int(z + 0.5),
                "mid_depth_m": float(z),
                "thickness_m": 1.0,
                "base_shear_modulus_mpa": float(85.0 + 4.0 * z),
                "cyclic_shear_strain_pct": 0.20,
                "hydraulic_history": "seasonal wetting-drying before cyclic loading",
                "claim_boundary": "1D proxy layer; not a calibrated site-response model",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "boundary_1d_layer_inputs.csv", index=False)
    return out


def case_summary(metrics: pd.DataFrame, layers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    profiles = []
    base_model = "constant_suction"
    base = metrics[metrics["model"] == base_model].iloc[0]
    base_mean_g = float(layers["base_shear_modulus_mpa"].mean())
    base_proxy_disp = float((layers["thickness_m"] / layers["base_shear_modulus_mpa"]).sum())

    for _, row in metrics.iterrows():
        model = row["model"]
        stiffness_loss = float(row["stiffness_loss_pct"])
        plastic_index = float(row["final_plastic_strain_index"])
        damage_factor = max(0.05, 1.0 - stiffness_loss / 100.0)
        degradation_multiplier = 1.0 + plastic_index / max(float(base["final_plastic_strain_index"]), 1e-9)
        model_g = layers["base_shear_modulus_mpa"].to_numpy(float) * damage_factor
        proxy_disp = float((layers["thickness_m"].to_numpy(float) / model_g).sum())
        eq_g = float(layers["thickness_m"].sum() / proxy_disp)
        relative_demand = float((proxy_disp / base_proxy_disp) * degradation_multiplier)
        threshold_depth = np.nan
        rel_diff = np.abs(model_g / layers["base_shear_modulus_mpa"].to_numpy(float) - 1.0)
        if np.any(rel_diff >= 0.10):
            threshold_depth = float(layers.loc[np.argmax(rel_diff >= 0.10), "mid_depth_m"])
        rows.append(
            {
                "model": model,
                "input_suction_kpa": float(row["suction_amp"]),
                "input_cyclic_strain_pct": float(row["cyclic_amp"]),
                "material_point_stiffness_loss_pct": stiffness_loss,
                "material_point_plastic_strain_index": plastic_index,
                "equivalent_1d_shear_modulus_mpa": eq_g,
                "equivalent_modulus_loss_vs_initial_pct": 100.0 * (1.0 - eq_g / base_mean_g),
                "proxy_flexibility_ratio_vs_constant_suction": proxy_disp / base_proxy_disp,
                "relative_deformation_demand_index": relative_demand,
                "first_depth_exceeding_10pct_modulus_change_m": threshold_depth,
                "claim_boundary": "Translates material-point outputs to an interpretable 1D metric; not site response, design approval, or FEM validation.",
            }
        )
        for layer, g in zip(layers.to_dict("records"), model_g):
            profiles.append(
                {
                    "model": model,
                    "mid_depth_m": layer["mid_depth_m"],
                    "thickness_m": layer["thickness_m"],
                    "initial_shear_modulus_mpa": layer["base_shear_modulus_mpa"],
                    "post_cyclic_proxy_shear_modulus_mpa": float(g),
                    "relative_modulus": float(g / layer["base_shear_modulus_mpa"]),
                }
            )
    summary = pd.DataFrame(rows)
    prof = pd.DataFrame(profiles)
    summary.to_csv(DATA / "boundary_1d_case_summary.csv", index=False)
    prof.to_csv(DATA / "boundary_1d_depth_profiles.csv", index=False)
    return summary, prof


def monte_carlo(summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(117)
    rows = []
    for _, row in summary.iterrows():
        for sample in range(250):
            stiffness = max(0.0, float(row["material_point_stiffness_loss_pct"]) * rng.lognormal(0.0, 0.12))
            plastic = max(0.0, float(row["material_point_plastic_strain_index"]) * rng.lognormal(0.0, 0.20))
            demand = (1.0 / max(0.05, 1.0 - stiffness / 100.0)) * (1.0 + 100.0 * plastic)
            rows.append(
                {
                    "model": row["model"],
                    "sample": sample,
                    "sampled_stiffness_loss_pct": stiffness,
                    "sampled_plastic_strain_index": plastic,
                    "sampled_relative_deformation_demand_index": demand,
                }
            )
    samples = pd.DataFrame(rows)
    samples.to_csv(DATA / "boundary_1d_monte_carlo_samples.csv", index=False)
    mc = (
        samples.groupby("model")["sampled_relative_deformation_demand_index"]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "p05", 0.5: "median", 0.95: "p95"})
    )
    mc.to_csv(DATA / "boundary_1d_monte_carlo_summary.csv", index=False)
    return mc


def claim_passport() -> pd.DataFrame:
    rows = [
        {
            "evidence_added": "1D hydraulic-memory proxy",
            "file_or_script": "scripts/boundary_1d_hydraulic_memory_proxy.py",
            "what_it_supports": "Shows how hydraulic-memory state assumptions change an equivalent 1D stiffness/deformation indicator under identical cyclic demand.",
            "what_it_does_not_support": "Not calibrated site response, not design approval, not FEM validation of a constitutive law.",
        },
        {
            "evidence_added": "Blind digitized Kinikles Fig. 12(a) holdout",
            "file_or_script": "scripts/external_kinikles2024_fig12a_blind_holdout.py",
            "what_it_supports": "Quantifies figure-level median and peak-envelope prediction errors under leave-one-saturation-group-out holdout.",
            "what_it_does_not_support": "Not raw cyclic time-history validation and not a soil-specific parameter calibration.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "table_boundary_value_claim_passport.csv", index=False)
    return out


def draw_profiles(profiles: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1500, 1050
    left, right, top, bottom = 170, 120, 105, 120
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(27, True)
    tick_font = font(22)
    colors = {
        "constant_suction": "#1b9e77",
        "no_hysteresis": "#7570b3",
        "hysteresis_only": "#d95f02",
        "full_hysteretic_damage": "#e7298a",
    }
    d.text((left, 42), "1D hydraulic-memory proxy: post-cyclic stiffness profiles", font=title_font, fill="#111111")
    d.rectangle((left, top, width - right, height - bottom), outline="#222222", width=3)

    max_g = float(profiles["post_cyclic_proxy_shear_modulus_mpa"].max()) * 1.08

    def xp(g):
        return left + (float(g) / max_g) * (width - right - left)

    def yp(z):
        return top + (float(z) / 10.0) * (height - bottom - top)

    for g in np.linspace(0, max_g, 6):
        x = xp(g)
        d.line((x, top, x, height - bottom), fill="#eeeeee")
        d.text((x - 25, height - bottom + 18), f"{g:.0f}", font=tick_font, fill="#222222")
    for z in range(0, 11, 2):
        y = yp(z)
        d.line((left, y, width - right, y), fill="#eeeeee")
        d.text((left - 55, y - 12), f"{z}", font=tick_font, fill="#222222")

    for model, group in profiles.groupby("model"):
        group = group.sort_values("mid_depth_m")
        pts = [(xp(r["post_cyclic_proxy_shear_modulus_mpa"]), yp(r["mid_depth_m"])) for _, r in group.iterrows()]
        d.line(pts, fill=colors.get(model, "#333333"), width=5)
        for x, y in pts:
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors.get(model, "#333333"))

    d.text((width // 2 - 160, height - 70), "Post-cyclic proxy shear modulus (MPa)", font=label_font, fill="#111111")
    d.text((35, height // 2), "Depth (m)", font=label_font, fill="#111111")
    y0 = top + 40
    for i, (model, color) in enumerate(colors.items()):
        yy = y0 + i * 42
        d.line((width - right - 430, yy, width - right - 370, yy), fill=color, width=6)
        d.text((width - right - 355, yy - 15), model, font=tick_font, fill="#111111")
    img.save(FIGS / "fig16_1d_hydraulic_memory_profiles.png")


def draw_sensitivity(mc: pd.DataFrame) -> None:
    width, height = 1500, 950
    left, right, top, bottom = 210, 80, 105, 170
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(27, True)
    tick_font = font(22)
    models = list(mc["model"])
    max_v = float(mc["p95"].max()) * 1.12
    d.text((left, 42), "Monte Carlo sensitivity of the 1D proxy demand metric", font=title_font, fill="#111111")
    d.rectangle((left, top, width - right, height - bottom), outline="#222222", width=3)

    def xp(v):
        return left + (float(v) / max_v) * (width - right - left)

    bar_h = 70
    gap = 45
    for i, row in mc.iterrows():
        y = top + 60 + i * (bar_h + gap)
        d.rectangle((xp(row["p05"]), y, xp(row["p95"]), y + bar_h), fill="#d9e8f6", outline="#4b6f91")
        d.rectangle((xp(row["median"]) - 4, y - 8, xp(row["median"]) + 4, y + bar_h + 8), fill="#d94801")
        d.text((35, y + 20), str(row["model"]), font=tick_font, fill="#111111")
    for v in np.linspace(0, max_v, 6):
        x = xp(v)
        d.line((x, top, x, height - bottom), fill="#eeeeee")
        d.text((x - 28, height - bottom + 18), f"{v:.1f}", font=tick_font, fill="#222222")
    d.text((left + 200, height - 80), "Relative deformation demand index, arbitrary but reproducible proxy", font=label_font, fill="#111111")
    d.text((left, height - 120), "Bars show 5th-95th percentiles; orange line is median. Scope is sensitivity, not site-response validation.", font=tick_font, fill="#333333")
    img.save(FIGS / "fig17_1d_decision_metric_sensitivity.png")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    metrics = load_model_metrics()
    layers = layer_inputs()
    summary, profiles = case_summary(metrics, layers)
    mc = monte_carlo(summary)
    claim_passport()
    draw_profiles(profiles)
    draw_sensitivity(mc)
    print(f"boundary_1d_hydraulic_memory_proxy=ok cases={len(summary)} layers={len(layers)}")


if __name__ == "__main__":
    main()
