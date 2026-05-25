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


def residual_diagnostics() -> pd.DataFrame:
    df = pd.read_csv(DATA / "rong_mccartney_digitized_200cycle_validation.csv")
    df = df.sort_values("initial_degree_saturation").reset_index(drop=True)
    df["residual_pct"] = df["benchmark_proxy_value_pct"] - df["observed_volumetric_strain_200_cycles_pct"]
    df["observed_rank"] = df["observed_volumetric_strain_200_cycles_pct"].rank(method="dense")
    df["model_rank"] = df["benchmark_proxy_value_pct"].rank(method="dense")
    df["rank_residual"] = df["model_rank"] - df["observed_rank"]
    df["observed_step_delta_pct"] = df["observed_volumetric_strain_200_cycles_pct"].diff()
    df["model_step_delta_pct"] = df["benchmark_proxy_value_pct"].diff()
    df["step_direction_agreement"] = (
        np.sign(df["observed_step_delta_pct"].fillna(0.0)) == np.sign(df["model_step_delta_pct"].fillna(0.0))
    )
    df["diagnostic_interpretation"] = np.where(
        df["initial_degree_saturation"] == 0.0,
        "dry-point magnitude is outside the paired retention-curve proxy range",
        np.where(
            df["rank_residual"] > 0,
            "model proxy ranks this saturation higher than the figure-level observation",
            "model proxy ranks this saturation lower than or similar to the figure-level observation",
        ),
    )
    return df


def holdout_transfer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for degree, name in [(1, "affine"), (2, "quadratic")]:
        for holdout_idx, holdout in df.iterrows():
            train = df.drop(index=holdout_idx)
            x_train = train["benchmark_proxy_value_pct"].to_numpy(float)
            y_train = train["observed_volumetric_strain_200_cycles_pct"].to_numpy(float)
            coeff = np.polyfit(x_train, y_train, degree)
            pred = float(np.polyval(coeff, float(holdout["benchmark_proxy_value_pct"])))
            obs = float(holdout["observed_volumetric_strain_200_cycles_pct"])
            rows.append(
                {
                    "transfer_model": name,
                    "holdout_initial_degree_saturation": float(holdout["initial_degree_saturation"]),
                    "observed_pct": obs,
                    "predicted_pct": pred,
                    "absolute_error_pct": abs(pred - obs),
                    "squared_error": (pred - obs) ** 2,
                    "claim_boundary": "Leave-one-saturation-group-out transfer on figure-level data; not raw time-history calibration.",
                }
            )
    pred = pd.DataFrame(rows)
    summary_rows = []
    for model, group in pred.groupby("transfer_model", sort=False):
        summary_rows.append(
            {
                "transfer_model": model,
                "n_holdout_groups": int(len(group)),
                "rmse_pct": float(np.sqrt(group["squared_error"].mean())),
                "mae_pct": float(group["absolute_error_pct"].mean()),
                "max_abs_error_pct": float(group["absolute_error_pct"].max()),
                "interpretation": (
                    "Residual diagnostics are a boundary test: the transfer can reduce magnitude error, "
                    "but it does not remove the adverse rank behavior in the unscaled proxy."
                ),
            }
        )
    return pred, pd.DataFrame(summary_rows)


def failure_mode_summary(df: pd.DataFrame, transfer_summary: pd.DataFrame) -> pd.DataFrame:
    obs = df["observed_volumetric_strain_200_cycles_pct"]
    pred = df["benchmark_proxy_value_pct"]
    rho = float(obs.corr(pred, method="spearman"))
    observed_peak_sr = float(df.loc[obs.idxmax(), "initial_degree_saturation"])
    model_peak_sr = float(df.loc[pred.idxmax(), "initial_degree_saturation"])
    observed_range = float(obs.max() - obs.min())
    model_range = float(pred.max() - pred.min())
    direction_agreement = float(df["step_direction_agreement"].iloc[1:].mean())
    best_transfer = transfer_summary.sort_values(["rmse_pct", "mae_pct"]).iloc[0]
    return pd.DataFrame(
        [
            {
                "source": "Rong/McCartney 200-cycle figure-level gate",
                "spearman_rank_correlation": rho,
                "observed_peak_initial_degree_saturation": observed_peak_sr,
                "model_proxy_peak_initial_degree_saturation": model_peak_sr,
                "observed_range_pct": observed_range,
                "model_proxy_range_pct": model_range,
                "model_to_observed_range_ratio": float(model_range / observed_range) if observed_range else np.nan,
                "adjacent_step_direction_agreement_fraction": direction_agreement,
                "best_leave_one_group_transfer": str(best_transfer.transfer_model),
                "best_transfer_rmse_pct": float(best_transfer.rmse_pct),
                "failure_mode": (
                    "The observed 200-cycle response is nonmonotone and has a large dry-point magnitude, "
                    "whereas the paired benchmark proxy is compressed and increases toward higher initial saturation."
                ),
                "claim_boundary": (
                    "Use this result as an explicit long-cycle boundary for the benchmark, not as evidence of "
                    "calibrated soil-specific prediction."
                ),
            }
        ]
    )


def draw(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    width, height = 1700, 1030
    left, right, top, bottom = 185, 100, 120, 210
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, tick_font = font(38, True), font(27, True), font(22)
    d.text((left, 42), "Rong/McCartney 200-cycle boundary diagnostics", font=title_font, fill="#111111")
    d.text((left, 88), "Figure-level residuals explain the adverse rank result without hiding it.", font=tick_font, fill="#333333")
    plot = (left, top, width - right, height - bottom)
    d.rectangle(plot, outline="#222222", width=3)
    ymax = max(3.0, float(df[["observed_volumetric_strain_200_cycles_pct", "benchmark_proxy_value_pct"]].max().max()) * 1.12)

    def xp(x: float) -> float:
        return left + (x / 0.6) * (width - right - left)

    def yp(y: float) -> float:
        return height - bottom - (y / ymax) * (height - bottom - top)

    for x in np.linspace(0, 0.6, 7):
        px = xp(float(x))
        d.line((px, top, px, height - bottom), fill="#eeeeee")
        d.text((px - 18, height - bottom + 18), f"{x:.1f}", font=tick_font, fill="#222222")
    for y in np.linspace(0, ymax, 7):
        py = yp(float(y))
        d.line((left, py, width - right, py), fill="#eeeeee")
        d.text((left - 70, py - 12), f"{y:.1f}", font=tick_font, fill="#222222")

    obs_pts = [(xp(float(r.initial_degree_saturation)), yp(float(r.observed_volumetric_strain_200_cycles_pct))) for r in df.itertuples()]
    pred_pts = [(xp(float(r.initial_degree_saturation)), yp(float(r.benchmark_proxy_value_pct))) for r in df.itertuples()]
    d.line(obs_pts, fill="#114c8d", width=5)
    d.line(pred_pts, fill="#d94801", width=5)
    for px, py in obs_pts:
        d.ellipse((px - 10, py - 10, px + 10, py + 10), fill="#114c8d")
    for px, py in pred_pts:
        d.rectangle((px - 9, py - 9, px + 9, py + 9), fill="#d94801")
    d.text((610, height - 78), "Initial degree of saturation", font=label_font, fill="#111111")
    d.text((left, height - 145), f"Spearman={summary.iloc[0].spearman_rank_correlation:.2f}; model/observed range ratio={summary.iloc[0].model_to_observed_range_ratio:.2f}; best transfer RMSE={summary.iloc[0].best_transfer_rmse_pct:.3f}%.", font=tick_font, fill="#222222")
    d.text((left, height - 110), "Boundary: the unscaled proxy cannot be claimed as calibrated long-cycle seismic-compression prediction.", font=tick_font, fill="#222222")
    d.text((left + 35, top + 35), "blue: digitized observation", font=tick_font, fill="#114c8d")
    d.text((left + 35, top + 70), "orange: paired benchmark proxy", font=tick_font, fill="#d94801")
    img.save(FIGS / "fig25_rong_mccartney_200cycle_boundary_diagnostics.png")


def main() -> None:
    df = residual_diagnostics()
    holdouts, holdout_summary = holdout_transfer(df)
    summary = failure_mode_summary(df, holdout_summary)
    df.to_csv(DATA / "rong_mccartney_200cycle_residual_diagnostics.csv", index=False)
    holdouts.to_csv(DATA / "rong_mccartney_200cycle_holdout_transfer_predictions.csv", index=False)
    holdout_summary.to_csv(DATA / "rong_mccartney_200cycle_holdout_transfer_summary.csv", index=False)
    summary.to_csv(DATA / "rong_mccartney_200cycle_failure_mode_summary.csv", index=False)
    draw(df, summary)
    print(f"rong_mccartney_200cycle_residual_diagnostics=ok groups={len(df)} spearman={summary.iloc[0].spearman_rank_correlation:.3f}")


if __name__ == "__main__":
    main()
