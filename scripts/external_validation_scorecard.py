from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"


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


def read_first(path: str) -> pd.Series:
    p = DATA / path
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"{p} is empty")
    return df.iloc[0]


def score_rows() -> pd.DataFrame:
    kin = pd.read_csv(DATA / "kinikles2024_fig12a_blind_holdout_summary.csv")
    kin_med = kin[kin["target"] == "volumetric_strain_median_pct"].iloc[0]
    kin_xfer = pd.read_csv(DATA / "kinikles2024_model_output_transfer_holdout_summary.csv")
    kin_xfer_med = kin_xfer[kin_xfer["target"] == "volumetric_strain_median_pct"].iloc[0]
    rong = read_first("rong_mccartney_digitized_200cycle_validation_summary.csv")
    rong_amp = pd.read_csv(DATA / "rong_mccartney_drained_amplitude_digitized_gate_summary.csv")
    rong_amp_full = rong_amp[rong_amp["model"] == "full_hysteretic_damage"].iloc[0]
    cross = pd.read_csv(DATA / "external_cross_source_consistency_summary.csv").iloc[0]
    gabr = read_first("gabr2019_unsaturated_triaxial_holdout_summary.csv")
    ng = read_first("ng2013_semicalibrated_envelope_summary.csv")
    zen = pd.read_csv(DATA / "zenodo_italian_clays_range_check.csv")
    zen_pass = bool(zen["model_initial_inside_range"].dropna().all() and zen["model_final_inside_range"].dropna().all())

    rows = [
        {
            "evidence_layer": "Kinikles 2024 digitized median envelope",
            "source_file": "kinikles2024_fig12a_blind_holdout_summary.csv",
            "metric": "leave-one-saturation-group-out RMSE/MAE/range coverage",
            "value": f"RMSE={kin_med.rmse_pct:.3f} pct, MAE={kin_med.mae_pct:.3f} pct, coverage={kin_med.range_coverage_fraction:.2f}",
            "score_0_to_1": min(1.0, max(0.0, 1.0 - float(kin_med.rmse_pct) / 1.0)),
            "claim_supported": "blind figure-level external envelope holdout",
            "claim_not_supported": "raw cyclic time-history validation",
        },
        {
            "evidence_layer": "Kinikles 2024 model-output transfer",
            "source_file": "kinikles2024_model_output_transfer_holdout_summary.csv",
            "metric": "benchmark-output transfer RMSE/MAE/range coverage",
            "value": f"RMSE={kin_xfer_med.rmse_pct:.3f} pct, MAE={kin_xfer_med.mae_pct:.3f} pct, coverage={kin_xfer_med.range_coverage_fraction:.2f}",
            "score_0_to_1": min(1.0, max(0.0, 1.0 - float(kin_xfer_med.rmse_pct) / 1.0)),
            "claim_supported": "benchmark output can be mapped to a recent direct-cyclic figure envelope",
            "claim_not_supported": "soil-specific calibrated constitutive prediction",
        },
        {
            "evidence_layer": "Rong/McCartney 200-cycle figure gate",
            "source_file": "rong_mccartney_digitized_200cycle_validation_summary.csv",
            "metric": "independent figure-level RMSE/MAE/rank agreement",
            "value": f"RMSE={rong.rmse_pct:.3f} pct, MAE={rong.mae_pct:.3f} pct, Spearman={rong.spearman_rank_correlation:.2f}",
            "score_0_to_1": min(1.0, max(0.0, (float(rong.spearman_rank_correlation) + 1.0) / 2.0)),
            "claim_supported": "independent 200-cycle trend/envelope gate",
            "claim_not_supported": "blind fit to raw Rong/McCartney time histories",
        },
        {
            "evidence_layer": "Rong/McCartney drained CSS amplitude gate",
            "source_file": "rong_mccartney_drained_amplitude_digitized_gate_summary.csv",
            "metric": "figure-level cyclic-amplitude trend/rank agreement",
            "value": f"Spearman={rong_amp_full.spearman_rank_correlation:.2f}; monotone={rong_amp_full.model_monotonic_increase}",
            "score_0_to_1": min(1.0, max(0.0, (float(rong_amp_full.spearman_rank_correlation) + 1.0) / 2.0)),
            "claim_supported": "independent drained CSS amplitude-response trend gate",
            "claim_not_supported": "raw drained CSS time-history validation",
        },
        {
            "evidence_layer": "Gabr 2019 static unsaturated triaxial holdout",
            "source_file": "gabr2019_unsaturated_triaxial_holdout_summary.csv",
            "metric": "held-out single-stage strength R2/MAE",
            "value": f"holdout R2={gabr['holdout_metrics.r2']:.3f}, MAE={gabr['holdout_metrics.mae_kpa']:.1f} kPa",
            "score_0_to_1": min(1.0, max(0.0, float(gabr["holdout_metrics.r2"]))),
            "claim_supported": "suction-strength trend plausibility",
            "claim_not_supported": "cyclic validation",
        },
        {
            "evidence_layer": "Ng/Zhou resilient-modulus gate",
            "source_file": "ng2013_semicalibrated_envelope_summary.csv",
            "metric": "suction and cyclic-stress semi-calibration direction",
            "value": f"0-30 kPa ratio={ng.semicalibrated_ratio_0_to_30:.2f}; 0-250 kPa ratio={ng.semicalibrated_ratio_0_to_250:.2f}",
            "score_0_to_1": 0.65,
            "claim_supported": "directional suction/cyclic-stress consistency",
            "claim_not_supported": "high-suction calibrated resilient-modulus prediction",
        },
        {
            "evidence_layer": "Zenodo Italian clays dynamic stiffness range",
            "source_file": "zenodo_italian_clays_range_check.csv",
            "metric": "model stiffness inside external dynamic-stiffness range",
            "value": "initial and final stiffness inside range" if zen_pass else "range check not fully passed",
            "score_0_to_1": 1.0 if zen_pass else 0.4,
            "claim_supported": "dynamic-stiffness magnitude plausibility",
            "claim_not_supported": "soil-specific calibration",
        },
        {
            "evidence_layer": "Cross-source claim-consistency audit",
            "source_file": "external_cross_source_consistency_summary.csv",
            "metric": "pass/boundary/review matrix across external layers",
            "value": f"score={cross.claim_consistency_score_0_to_1:.2f}; pass={int(cross.pass_count)}, boundary={int(cross.boundary_count)}, review={int(cross.review_count)}",
            "score_0_to_1": float(cross.claim_consistency_score_0_to_1),
            "claim_supported": "bounded external-evidence synthesis across sources",
            "claim_not_supported": "replacement for public raw cyclic response histories",
        },
    ]
    out = pd.DataFrame(rows)
    out["mean_score_0_to_1"] = float(out["score_0_to_1"].mean())
    return out


def draw(df: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1800, 1300
    left, right, top, row_h = 95, 85, 145, 122
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font, head_font, text_font, small_font = font(40, True), font(30, True), font(28), font(24)
    d.text((left, 44), "Bounded external-evidence scorecard", font=title_font, fill="#111111")
    d.text((left, 98), "Overall score is the mean of evidence-layer scores; row values remain individual gates.", font=text_font, fill="#333333")
    d.line((left, top - 18, width - right, top - 18), fill="#222222", width=3)

    colors = ["#f7fbff", "#ffffff"]
    for i, row in df.iterrows():
        y = top + i * row_h
        d.rectangle((left - 12, y - 8, width - right, y + row_h - 14), fill=colors[i % 2], outline="#dddddd")
        d.text((left, y), str(row["evidence_layer"])[:48], font=text_font, fill="#111111")
        d.text((left, y + 42), str(row["value"])[:76], font=small_font, fill="#333333")
        score = float(row["score_0_to_1"])
        bar_x, bar_y, bar_w, bar_h = 1365, y + 20, 285, 36
        d.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline="#555555")
        d.rectangle((bar_x, bar_y, bar_x + bar_w * score, bar_y + bar_h), fill="#2b8cbe")
        d.text((bar_x + bar_w + 18, y + 17), f"{score:.2f}", font=text_font, fill="#111111")

    mean_score = float(df["score_0_to_1"].mean())
    d.text((left, height - 95), f"Mean bounded-evidence score: {mean_score:.3f}. Residual gap: no public raw cyclic response histories.", font=head_font, fill="#111111")
    img.save(FIGS / "fig21_external_validation_scorecard.png")


def main() -> None:
    df = score_rows()
    df.to_csv(DATA / "external_cross_source_validation_scorecard.csv", index=False)
    draw(df)
    print(f"external_validation_scorecard=ok layers={len(df)} mean={df['score_0_to_1'].mean():.3f}")


if __name__ == "__main__":
    main()
