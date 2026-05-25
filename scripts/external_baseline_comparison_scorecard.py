from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_unsat_cyclic_benchmark import Params, simulate_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ["constant_suction", "no_hysteresis", "hysteresis_only", "full_hysteretic_damage"]


def suction_for_initial_saturation(sr0: float) -> float:
    p = Params()
    sr = max(sr0, p.sr_res + 1.0e-4)
    se = np.clip((sr - p.sr_res) / (p.sr_sat - p.sr_res), 1.0e-4, 0.9999)
    m = 1.0 - 1.0 / p.n_dry
    return float(((se ** (-1.0 / m) - 1.0) ** (1.0 / p.n_dry)) / p.alpha_dry)


def model_feature(model: str, sr0: float, cyclic_amp: float = 0.20) -> float:
    suction = max(0.5, suction_for_initial_saturation(sr0))
    df = simulate_case(model, suction_amp=min(100.0, suction), cyclic_amp=cyclic_amp, n_steps=720, suction_mean=suction)
    return float(100.0 * df["plastic_strain_index"].iloc[-1])


def summarize_errors(source: str, target: str, rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    summaries = []
    for model, group in out.groupby("model", sort=False):
        err = group["predicted_pct"] - group["observed_pct"]
        summaries.append(
            {
                "source": source,
                "target": target,
                "model": model,
                "n": int(len(group)),
                "rmse_pct": float(np.sqrt(np.mean(err**2))),
                "mae_pct": float(np.mean(np.abs(err))),
                "bias_pct": float(np.mean(err)),
                "spearman_rank_correlation": float(pd.Series(group["observed_pct"]).corr(pd.Series(group["predicted_pct"]), method="spearman")),
            }
        )
    return pd.DataFrame(summaries)


def kinikles_baseline_comparison() -> pd.DataFrame:
    env = pd.read_csv(DATA / "kinikles2024_fig12a_digitized_envelope_summary.csv")
    rows: list[dict] = []
    for model in MODELS:
        features = {float(sr): model_feature(model, float(sr)) for sr in env["group_initial_degree_saturation"].to_numpy(float)}
        merged = env.copy()
        merged["model_feature_pct"] = merged["group_initial_degree_saturation"].map(features)
        for target in ["volumetric_strain_median_pct", "volumetric_strain_max_pct"]:
            groups = sorted(merged["group_initial_degree_saturation"].unique())
            for holdout in groups:
                train = merged[merged["group_initial_degree_saturation"] != holdout]
                test = merged[merged["group_initial_degree_saturation"] == holdout].iloc[0]
                x = train["model_feature_pct"].to_numpy(float)
                y = train[target].to_numpy(float)
                degree = 2 if len(np.unique(x)) >= 3 else 1
                coeff = np.polyfit(x, y, degree)
                pred = float(np.polyval(coeff, float(test["model_feature_pct"])))
                rows.append(
                    {
                        "source": "Kinikles et al. 2024 Figure 12(a)",
                        "target": target,
                        "model": model,
                        "holdout_initial_degree_saturation": float(holdout),
                        "observed_pct": float(test[target]),
                        "predicted_pct": pred,
                        "model_feature_pct": float(test["model_feature_pct"]),
                        "claim_boundary": "Baseline comparison uses digitized figure envelopes and scripted benchmark outputs only.",
                    }
                )
    return summarize_errors("Kinikles et al. 2024 Figure 12(a)", "leave_one_saturation_group_out_transfer", rows)


def rong_baseline_comparison() -> pd.DataFrame:
    obs = pd.read_csv(DATA / "rong_mccartney_digitized_200cycle_validation.csv")
    rows: list[dict] = []
    for model in MODELS:
        for _, row in obs.iterrows():
            sr = float(row["initial_degree_saturation"])
            rows.append(
                {
                    "source": "Rong/McCartney PEER 2022/05 200-cycle figure gate",
                    "target": "volumetric_strain_200_cycles_pct",
                    "model": model,
                    "holdout_initial_degree_saturation": sr,
                    "observed_pct": float(row["observed_volumetric_strain_200_cycles_pct"]),
                    "predicted_pct": model_feature(model, sr),
                    "claim_boundary": "Baseline comparison is a figure-level gate, not raw time-history validation.",
                }
            )
    return summarize_errors("Rong/McCartney PEER 2022/05 200-cycle figure gate", "volumetric_strain_200_cycles_pct", rows)


def main() -> None:
    out = pd.concat([kinikles_baseline_comparison(), rong_baseline_comparison()], ignore_index=True)
    best = []
    for (source, target), group in out.groupby(["source", "target"]):
        winner = group.sort_values(["rmse_pct", "mae_pct"]).iloc[0]
        full_rmse = float(group[group["model"] == "full_hysteretic_damage"]["rmse_pct"].iloc[0])
        best.append(
            {
                "source": source,
                "target": target,
                "best_rmse_model": winner["model"],
                "best_rmse_pct": float(winner["rmse_pct"]),
                "full_model_rmse_pct": full_rmse,
                "full_model_best": bool(winner["model"] == "full_hysteretic_damage"),
                "interpretation": "Do not claim calibrated superiority unless full_model_best is true.",
            }
        )
    out.to_csv(DATA / "external_baseline_comparison_scorecard.csv", index=False)
    pd.DataFrame(best).to_csv(DATA / "external_baseline_comparison_summary.csv", index=False)
    print(f"external_baseline_comparison=ok rows={len(out)} targets={len(best)}")


if __name__ == "__main__":
    main()
