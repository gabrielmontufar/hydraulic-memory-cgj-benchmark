from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RAW = ROOT / "external_data" / "kinikles_2024_unsaturated_seismic_compression"
SOURCE = RAW / "1-s2.0-S0266352X24000491-gr12_lrg.jpg"
PANEL = RAW / "kinikles_gr12_panel_a_fullcrop.png"

SOURCE_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0266352X24000491-gr12_lrg.jpg"
ARTICLE_URL = "https://www.sciencedirect.com/science/article/pii/S0266352X24000491"
DOI = "10.1016/j.compgeo.2024.106113"

# Manual digitization of Figure 12(a) from the downloaded open-access image.
# Coordinates are in pixels on PANEL, whose crop box is defined below. The
# points are not assigned to 15-cycle versus 200-cycle subsets because the
# grayscale open-diamond markers are not reliably distinguishable in the
# downloaded figure. The gate therefore uses the observed point envelope.
CROP_BOX = (220, 0, 1465, 1130)
AXIS = {
    "x0_px": 77.0,
    "x1_px": 1244.0,
    "y0_px": 893.0,
    "y1_px": 29.0,
    "x0_value": 0.0,
    "x1_value": 0.6,
    "y0_value": 0.0,
    "y1_value": 3.0,
}

DIGITIZED_POINTS = [
    # group_s0, pixel_x, pixel_y
    (0.00, 77, 574),
    (0.00, 77, 811),
    (0.12, 309, 575),
    (0.12, 310, 656),
    (0.12, 310, 767),
    (0.12, 310, 803),
    (0.20, 484, 519),
    (0.20, 484, 535),
    (0.20, 484, 719),
    (0.20, 484, 751),
    (0.30, 669, 422),
    (0.30, 669, 442),
    (0.30, 669, 694),
    (0.30, 669, 765),
    (0.40, 865, 354),
    (0.40, 866, 440),
    (0.40, 866, 676),
    (0.40, 866, 696),
    (0.56, 1178, 408),
    (0.56, 1176, 459),
    (0.56, 1176, 704),
    (0.56, 1182, 714),
]


def font(size: int):
    for candidate in [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\times.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def ensure_panel() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source image: {SOURCE}")
    image = Image.open(SOURCE).convert("RGB")
    panel = image.crop(CROP_BOX)
    panel.save(PANEL)


def pixel_to_data(px: float, py: float) -> tuple[float, float]:
    x = AXIS["x0_value"] + (px - AXIS["x0_px"]) * (AXIS["x1_value"] - AXIS["x0_value"]) / (
        AXIS["x1_px"] - AXIS["x0_px"]
    )
    y = AXIS["y0_value"] + (AXIS["y0_px"] - py) * (AXIS["y1_value"] - AXIS["y0_value"]) / (
        AXIS["y0_px"] - AXIS["y1_px"]
    )
    return float(x), float(y)


def digitized_dataframe() -> pd.DataFrame:
    rows = []
    x_unc_px = 15.0
    y_unc_px = 22.0
    x_unc = x_unc_px * (AXIS["x1_value"] - AXIS["x0_value"]) / (AXIS["x1_px"] - AXIS["x0_px"])
    y_unc = y_unc_px * (AXIS["y1_value"] - AXIS["y0_value"]) / (AXIS["y0_px"] - AXIS["y1_px"])
    for idx, (group_s0, px, py) in enumerate(DIGITIZED_POINTS, start=1):
        s0, eps = pixel_to_data(px, py)
        rows.append(
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "source_url": SOURCE_URL,
                "article_url": ARTICLE_URL,
                "doi": DOI,
                "digitization_type": "manual_pixel_envelope_from_open_access_figure",
                "point_id": f"K2024-F12a-{idx:02d}",
                "published_cycle_marker": "unassigned_15_or_200_cycles",
                "group_initial_degree_saturation": group_s0,
                "pixel_x": px,
                "pixel_y": py,
                "digitized_initial_degree_saturation": s0,
                "digitized_volumetric_strain_pct": eps,
                "x_uncertainty_abs": x_unc,
                "volumetric_strain_uncertainty_pct": y_unc,
                "axis_calibration": "x 0-0.6 S0; y 0-3% volumetric strain; calibrated on panel axes",
                "claim_boundary": "digitized figure evidence only; not raw experimental time history",
            }
        )
    return pd.DataFrame(rows)


def envelope_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_s0, group in points.groupby("group_initial_degree_saturation"):
        vals = group["digitized_volumetric_strain_pct"].to_numpy(float)
        rows.append(
            {
                "group_initial_degree_saturation": group_s0,
                "n_digitized_points": len(group),
                "volumetric_strain_min_pct": float(vals.min()),
                "volumetric_strain_median_pct": float(np.median(vals)),
                "volumetric_strain_max_pct": float(vals.max()),
                "volumetric_strain_range_pct": float(vals.max() - vals.min()),
            }
        )
    return pd.DataFrame(rows)


def draw_digitization(points: pd.DataFrame, envelope: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1600, 1050
    left, right, top, bottom = 180, 80, 100, 160
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = font(36)
    label_font = font(30)
    tick_font = font(24)
    small_font = font(22)
    plot = (left, top, width - right, height - bottom)

    def xp(x: float) -> float:
        return left + (x / 0.6) * (width - right - left)

    def yp(y: float) -> float:
        return height - bottom - (y / 3.0) * (height - bottom - top)

    d.rectangle(plot, outline="#222222", width=3)
    for i in range(7):
        x = i * 0.1
        px = xp(x)
        d.line([(px, top), (px, height - bottom)], fill="#eeeeee", width=1)
        d.text((px - 22, height - bottom + 18), f"{x:.1f}", font=tick_font, fill="#222222")
    for i in range(7):
        y = i * 0.5
        py = yp(y)
        d.line([(left, py), (width - right, py)], fill="#eeeeee", width=1)
        d.text((left - 70, py - 14), f"{y:.1f}", font=tick_font, fill="#222222")
    d.text((left, 35), "Digitized Kinikles et al. (2024) Fig. 12(a) point envelope", font=title_font, fill="#111111")
    d.text((560, height - 75), "Initial degree of saturation, S0", font=label_font, fill="#111111")
    ylab = Image.new("RGBA", (420, 45), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylab)
    yd.text((0, 0), "Volumetric strain (%)", font=label_font, fill="#111111")
    ylab = ylab.rotate(90, expand=True)
    img.paste(ylab, (55, 350), ylab)

    # Point cloud
    for _, row in points.iterrows():
        x = float(row["digitized_initial_degree_saturation"])
        y = float(row["digitized_volumetric_strain_pct"])
        px, py = xp(x), yp(y)
        d.ellipse((px - 7, py - 7, px + 7, py + 7), outline="#114c8d", fill="#d7e8ff", width=2)

    # Envelope bars
    for _, row in envelope.iterrows():
        x = float(row["group_initial_degree_saturation"])
        y0 = float(row["volumetric_strain_min_pct"])
        y1 = float(row["volumetric_strain_max_pct"])
        ym = float(row["volumetric_strain_median_pct"])
        px = xp(x)
        d.line((px, yp(y0), px, yp(y1)), fill="#c83f12", width=4)
        d.line((px - 18, yp(ym), px + 18, yp(ym)), fill="#c83f12", width=4)

    d.text(
        (left, height - 115),
        "Markers: manual pixel digitization. Red bars: min-max envelope; red ticks: median. This is digitized figure evidence, not raw time-history data.",
        font=small_font,
        fill="#333333",
    )
    img.save(FIGS / "fig14_kinikles2024_digitized_volumetric_envelope.png")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    ensure_panel()
    points = digitized_dataframe()
    envelope = envelope_summary(points)

    max_row = envelope.loc[envelope["volumetric_strain_max_pct"].idxmax()]
    trend = pd.DataFrame(
        [
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "digitized_points": len(points),
                "saturation_groups": len(envelope),
                "max_digitized_volumetric_strain_pct": float(max_row["volumetric_strain_max_pct"]),
                "saturation_at_max_group": float(max_row["group_initial_degree_saturation"]),
                "median_peak_group": float(
                    envelope.loc[envelope["volumetric_strain_median_pct"].idxmax(), "group_initial_degree_saturation"]
                ),
                "cycle_assignment_reliable": False,
                "raw_time_history_available": False,
                "validation_use": "partial digitized direct-cyclic envelope; supports comparison and future calibration target only",
                "prohibited_claim": "Do not claim blind calibrated prediction of Kinikles et al. response histories from this digitization alone.",
            }
        ]
    )

    points.to_csv(DATA / "kinikles2024_fig12a_digitized_points.csv", index=False)
    envelope.to_csv(DATA / "kinikles2024_fig12a_digitized_envelope_summary.csv", index=False)
    trend.to_csv(DATA / "kinikles2024_fig12a_digitized_gate_summary.csv", index=False)
    draw_digitization(points, envelope)
    print(f"kinikles2024_fig12a_digitized_envelope=ok points={len(points)} groups={len(envelope)}")


if __name__ == "__main__":
    main()
