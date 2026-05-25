from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf") if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf") if bold else Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def color(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    if value < 0.10:
        return blend((237, 248, 233), (186, 228, 179), max(0.0, value / 0.10))
    if value <= 0.30:
        return blend((254, 232, 200), (253, 141, 60), (value - 0.10) / 0.20)
    if t < 0.75:
        return blend((252, 141, 89), (215, 48, 31), t / 0.75)
    return blend((215, 48, 31), (103, 0, 13), (t - 0.75) / 0.25)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def phase(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value <= 0.30:
        return "relevant"
    return "dominant"


def main() -> int:
    rows = pd.read_csv(DATA / "hydraulic_memory_amplification_index.csv")
    pivot = rows.pivot_table(
        index="suction_amp_kpa",
        columns="cyclic_amp",
        values="hmai_composite" if "hmai_composite" in rows.columns else "hydraulic_memory_amplification_index",
        aggfunc="mean",
    ).sort_index(ascending=False)
    xvals = list(pivot.columns)
    yvals = list(pivot.index)
    values = pivot.to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    left, top = 305, 220
    cell_w, cell_h = 185, 110
    legend_w, bottom_pad = 430, 210
    w = left + len(xvals) * cell_w + legend_w
    h = top + len(yvals) * cell_h + bottom_pad
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(44, True)
    label_font = font(34, True)
    tick_font = font(30)
    val_font = font(31, True)
    note_font = font(25)

    title = "HMAI phase map with full benchmark grid"
    draw.text((left, 65), title, fill="#111111", font=title_font)
    subtitle = "Hydraulic-memory contrast: hysteresis-only vs no-hysteresis; classes are diagnostic, not design limits"
    draw.text((left, 125), subtitle, fill="#333333", font=note_font)

    for i, s in enumerate(yvals):
        label = f"{s:g}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - 35 - tw, top + i * cell_h + (cell_h - th) / 2), label, fill="#111111", font=tick_font)
    
    # Draw the y-axis label vertically to avoid overlap with tick labels.
    y_label = "Suction amplitude (kPa)"
    tw_y, th_y = text_size(draw, y_label, label_font)
    label_img = Image.new("RGBA", (tw_y + 20, th_y + 20), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((10, 10), y_label, fill="#111111", font=label_font)
    label_img = label_img.rotate(90, expand=True)
    img.paste(label_img, (45, int(top + len(yvals) * cell_h / 2 - label_img.height / 2)), label_img)

    for j, c in enumerate(xvals):
        label = f"{c:g}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left + j * cell_w + (cell_w - tw) / 2, top - 48), label, fill="#111111", font=tick_font)
    xlabel = "Cyclic amplitude ratio"
    tw, _ = text_size(draw, xlabel, label_font)
    draw.text((left + len(xvals) * cell_w / 2 - tw / 2, top + len(yvals) * cell_h + 55), xlabel, fill="#111111", font=label_font)

    for i, s in enumerate(yvals):
        for j, c in enumerate(xvals):
            val = float(pivot.loc[s, c])
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            fill = color(val, vmin, vmax)
            border = {"negligible": "#238b45", "relevant": "#d95f0e", "dominant": "#99000d"}[phase(val)]
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=fill, outline=border, width=5)
            txt = f"{val:.2f}"
            phase_txt = phase(val)
            tw, th = text_size(draw, txt, val_font)
            ptw, pth = text_size(draw, phase_txt, note_font)
            text_fill = "#ffffff" if sum(fill) < 330 else "#111111"
            draw.text((x0 + (cell_w - tw) / 2, y0 + cell_h / 2 - th - 3), txt, fill=text_fill, font=val_font)
            draw.text((x0 + (cell_w - ptw) / 2, y0 + cell_h / 2 + 10), phase_txt, fill=text_fill, font=note_font)

    overlay_path = DATA / "external_hmai_phase_overlay_points.csv"
    if overlay_path.exists():
        overlay = pd.read_csv(overlay_path)
        for _, point in overlay.iterrows():
            sx = float(point["suction_amp_kpa"])
            cx = float(point["cyclic_amp"])
            nearest_s = min(yvals, key=lambda v: abs(float(v) - sx))
            nearest_c = min(xvals, key=lambda v: abs(float(v) - cx))
            i = yvals.index(nearest_s)
            j = xvals.index(nearest_c)
            x = left + j * cell_w + cell_w - 32
            y = top + i * cell_h + 30
            fill = "#1f78b4" if bool(point["classification_match"]) else "#6a3d9a"
            draw.ellipse([x - 17, y - 17, x + 17, y + 17], fill=fill, outline="#ffffff", width=4)
            label = str(point["marker_label"])[:1]
            tw, th = text_size(draw, label, note_font)
            draw.text((x - tw / 2, y - th / 2 - 1), label, fill="#ffffff", font=note_font)

    draw.rectangle([left, top, left + len(xvals) * cell_w, top + len(yvals) * cell_h], outline="#222222", width=3)

    legend_x = left + len(xvals) * cell_w + 105
    legend_y = top
    legend_h = len(yvals) * cell_h
    for k in range(120):
        t = k / 119
        fill = color(vmin + (vmax - vmin) * (1 - t), vmin, vmax)
        y0 = legend_y + int(k * legend_h / 120)
        y1 = legend_y + int((k + 1) * legend_h / 120)
        draw.rectangle([legend_x, y0, legend_x + 50, y1], fill=fill, outline=fill)
    draw.rectangle([legend_x, legend_y, legend_x + 50, legend_y + legend_h], outline="#222222", width=2)
    for value, label in [(vmax, f"{vmax:.3f}"), ((vmin + vmax) / 2, f"{(vmin + vmax) / 2:.3f}"), (vmin, f"{vmin:.3f}")]:
        yy = legend_y + (1 - (value - vmin) / (vmax - vmin)) * legend_h
        draw.line([legend_x + 55, yy, legend_x + 75, yy], fill="#111111", width=2)
        draw.text((legend_x + 85, yy - 16), label, fill="#111111", font=tick_font)
    draw.text((legend_x, legend_y - 55), "HMAI", fill="#111111", font=label_font)

    class_y = legend_y + legend_h + 45
    for label, fill, outline in [
        ("<0.10 negligible", (186, 228, 179), "#238b45"),
        ("0.10-0.30 relevant", (253, 174, 107), "#d95f0e"),
        (">0.30 dominant", (215, 48, 31), "#99000d"),
    ]:
        draw.rectangle([legend_x, class_y, legend_x + 42, class_y + 28], fill=fill, outline=outline, width=3)
        draw.text((legend_x + 58, class_y - 4), label, fill="#111111", font=note_font)
        class_y += 42

    if overlay_path.exists():
        class_y += 10
        draw.ellipse([legend_x, class_y, legend_x + 30, class_y + 30], fill="#1f78b4", outline="#ffffff", width=3)
        draw.text((legend_x + 48, class_y - 3), "external match", fill="#111111", font=note_font)
        class_y += 42
        draw.ellipse([legend_x, class_y, legend_x + 30, class_y + 30], fill="#6a3d9a", outline="#ffffff", width=3)
        draw.text((legend_x + 48, class_y - 3), "external boundary", fill="#111111", font=note_font)

    # Thresholds and external markers are described in the manuscript caption; omit the
    # in-figure footnote to keep the embedded figure uncluttered.
    out = FIGS / "fig20_hmai_heatmap.png"
    img.save(out, dpi=(300, 300))
    print(f"hmai_phase_map=ok path={out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
