from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "times.ttf"]
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def audit_rows() -> pd.DataFrame:
    kin = pd.read_csv(DATA / "kinikles2024_fig12a_digitized_envelope_summary.csv")
    kin_transfer = pd.read_csv(DATA / "kinikles2024_model_output_transfer_holdout_summary.csv")
    kin_transfer_med = kin_transfer[kin_transfer["target"] == "volumetric_strain_median_pct"].iloc[0]
    rong_200 = pd.read_csv(DATA / "rong_mccartney_digitized_200cycle_validation_summary.csv").iloc[0]
    rong_amp = pd.read_csv(DATA / "rong_mccartney_drained_amplitude_digitized_gate_summary.csv")
    rong_amp_full = rong_amp[rong_amp["model"] == "full_hysteretic_damage"].iloc[0]
    gabr = pd.read_csv(DATA / "gabr2019_unsaturated_triaxial_holdout_summary.csv").iloc[0]
    ng = pd.read_csv(DATA / "ng2013_semicalibrated_envelope_summary.csv").iloc[0]
    zen = pd.read_csv(DATA / "zenodo_italian_clays_range_check.csv")

    rows = [
        {
            "cross_source_check": "direct_cyclic_figure_transfer",
            "external_sources": "Kinikles 2024 Figure 12(a)",
            "measured_statistic": "median-envelope model-output transfer RMSE",
            "value": float(kin_transfer_med.rmse_pct),
            "threshold_or_rule": "RMSE <= 0.50 percentage points",
            "status": "pass" if float(kin_transfer_med.rmse_pct) <= 0.50 else "review",
            "supports_claim": "benchmark-output transfer has useful figure-level predictive content",
            "prevents_overclaim": "does not prove raw cyclic time-history calibration",
        },
        {
            "cross_source_check": "saturation_rank_boundary",
            "external_sources": "Kinikles 2024 and Rong/McCartney 200-cycle figures",
            "measured_statistic": "Spearman signs across saturation-indexed cyclic responses",
            "value": float(np.sign(spearman(kin["group_initial_degree_saturation"], kin["volumetric_strain_median_pct"])) * np.sign(float(rong_200.spearman_rank_correlation))),
            "threshold_or_rule": "negative or mixed sign is retained as a boundary, not forced into agreement",
            "status": "boundary",
            "supports_claim": "external cyclic sources are not reduced to one universal saturation trend",
            "prevents_overclaim": "prevents claiming universal calibrated superiority over all cyclic data",
        },
        {
            "cross_source_check": "cyclic_amplitude_monotonicity",
            "external_sources": "Rong/McCartney drained CSS Figure 4.5",
            "measured_statistic": "full-model feature Spearman and monotonic trend",
            "value": float(rong_amp_full.spearman_rank_correlation),
            "threshold_or_rule": "Spearman >= 0.95 and monotone=True",
            "status": "pass" if float(rong_amp_full.spearman_rank_correlation) >= 0.95 and bool(rong_amp_full.model_monotonic_increase) else "review",
            "supports_claim": "model feature preserves observed amplitude-response ordering",
            "prevents_overclaim": "does not calibrate cyclic shear-strain scale",
        },
        {
            "cross_source_check": "static_unsaturated_strength_holdout",
            "external_sources": "Gabr 2019 unsaturated triaxial data",
            "measured_statistic": "held-out strength R2",
            "value": float(gabr["holdout_metrics.r2"]),
            "threshold_or_rule": "R2 >= 0.50",
            "status": "pass" if float(gabr["holdout_metrics.r2"]) >= 0.50 else "review",
            "supports_claim": "suction-strength trend plausibility",
            "prevents_overclaim": "does not validate cyclic plasticity",
        },
        {
            "cross_source_check": "resilient_modulus_direction",
            "external_sources": "Ng/Zhou unsaturated resilient-modulus evidence",
            "measured_statistic": "semi-calibrated suction ratio from 0 to 30 kPa",
            "value": float(ng.semicalibrated_ratio_0_to_30),
            "threshold_or_rule": "ratio > 1.0",
            "status": "pass" if float(ng.semicalibrated_ratio_0_to_30) > 1.0 else "review",
            "supports_claim": "suction stiffening direction is externally plausible",
            "prevents_overclaim": "does not provide high-suction resilient-modulus calibration",
        },
        {
            "cross_source_check": "dynamic_stiffness_magnitude_range",
            "external_sources": "Zenodo Italian clays dynamic-stiffness archive",
            "measured_statistic": "initial and final benchmark stiffness within external range",
            "value": float(zen["model_initial_inside_range"].dropna().mean() * zen["model_final_inside_range"].dropna().mean()),
            "threshold_or_rule": "all range checks inside",
            "status": "pass" if bool(zen["model_initial_inside_range"].dropna().all() and zen["model_final_inside_range"].dropna().all()) else "review",
            "supports_claim": "benchmark stiffness magnitudes are not outside broad dynamic-stiffness evidence",
            "prevents_overclaim": "does not establish soil-specific stiffness calibration",
        },
    ]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    pass_count = int((df["status"] == "pass").sum())
    boundary_count = int((df["status"] == "boundary").sum())
    review_count = int((df["status"] == "review").sum())
    score = (pass_count + 0.5 * boundary_count) / len(df)
    return pd.DataFrame(
        [
            {
                "n_cross_source_checks": int(len(df)),
                "pass_count": pass_count,
                "boundary_count": boundary_count,
                "review_count": review_count,
                "claim_consistency_score_0_to_1": float(score),
                "raw_time_history_gap_retained": True,
                "interpretation": (
                    "Cross-source audit supports a bounded plausibility-envelope claim while retaining adverse "
                    "saturation-rank evidence and the absence of public raw cyclic response histories."
                ),
            }
        ]
    )


def draw(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1800, 1050
    left, top, row_h = 90, 145, 120
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font, text_font, small_font = font(40, True), font(28), font(23)
    d.text((left, 45), "Cross-source external consistency audit", font=title_font, fill="#111111")
    d.text((left, 96), "Passes, boundaries and review flags across independent external evidence layers.", font=text_font, fill="#333333")
    colors = {"pass": "#c7e9c0", "boundary": "#fee391", "review": "#fcbba1"}
    for i, row in df.iterrows():
        y = top + i * row_h
        d.rectangle((left, y, width - 90, y + row_h - 14), fill="#ffffff", outline="#d0d0d0")
        d.rectangle((left, y, left + 150, y + row_h - 14), fill=colors[str(row.status)])
        d.text((left + 24, y + 34), str(row.status).upper(), font=text_font, fill="#111111")
        d.text((left + 180, y + 16), str(row.cross_source_check), font=text_font, fill="#111111")
        d.text((left + 180, y + 54), str(row.supports_claim)[:94], font=small_font, fill="#333333")
        d.text((left + 180, y + 84), str(row.prevents_overclaim)[:100], font=small_font, fill="#555555")
    s = summary.iloc[0]
    d.text(
        (left, height - 105),
        f"Consistency score={s.claim_consistency_score_0_to_1:.2f}; pass={int(s.pass_count)}, boundary={int(s.boundary_count)}, review={int(s.review_count)}.",
        font=text_font,
        fill="#111111",
    )
    d.text((left, height - 66), "The raw cyclic time-history gap is retained explicitly.", font=small_font, fill="#333333")
    img.save(FIGS / "fig24_cross_source_external_consistency_audit.png")


def main() -> None:
    df = audit_rows()
    summary = summarize(df)
    df.to_csv(DATA / "external_cross_source_consistency_audit.csv", index=False)
    summary.to_csv(DATA / "external_cross_source_consistency_summary.csv", index=False)
    draw(df, summary)
    print(
        "external_cross_source_consistency=ok "
        f"checks={len(df)} score={summary.iloc[0].claim_consistency_score_0_to_1:.3f}"
    )


if __name__ == "__main__":
    main()
