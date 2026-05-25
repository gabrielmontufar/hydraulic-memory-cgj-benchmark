from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_unsat_cyclic_benchmark import Params, simulate_case


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ["constant_suction", "no_hysteresis", "hysteresis_only", "full_hysteretic_damage"]
WEIGHTS = {
    "base": (0.40, 0.30, 0.30),
    "equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "deformation_heavy": (0.60, 0.20, 0.20),
    "stiffness_heavy": (0.25, 0.50, 0.25),
    "damage_heavy": (0.25, 0.25, 0.50),
}


def clipped(value: float) -> float:
    return max(0.0, min(1.0, value))


def phase(value: float) -> str:
    if value < 0.10:
        return "negligible"
    if value <= 0.30:
        return "relevant"
    return "dominant"


def suction_for_initial_saturation(sr0: float) -> float:
    p = Params()
    se = np.clip((sr0 - p.sr_res) / (p.sr_sat - p.sr_res), 1.0e-4, 0.9999)
    m = 1.0 - 1.0 / p.n_dry
    return float(((se ** (-1.0 / m) - 1.0) ** (1.0 / p.n_dry)) / p.alpha_dry)


def response_summary(model: str, sr0: float, cyclic_amp: float = 0.20) -> dict:
    suction = max(0.5, suction_for_initial_saturation(sr0))
    df = simulate_case(
        model,
        suction_amp=min(100.0, suction),
        cyclic_amp=cyclic_amp,
        n_steps=720,
        suction_mean=suction,
    )
    return {
        "model": model,
        "initial_degree_saturation": sr0,
        "model_initial_degree_saturation": float(df["degree_saturation"].iloc[0]),
        "final_plastic_strain_index": float(df["plastic_strain_index"].iloc[-1]),
        "stiffness_loss_pct": float(100.0 * (1.0 - df["secant_stiffness_mpa"].iloc[-1] / df["secant_stiffness_mpa"].iloc[0])),
        "final_hydraulic_damage": float(df["hydraulic_damage"].iloc[-1]),
        "final_seismic_damage": float(df["seismic_damage"].iloc[-1]),
    }


def hmai_components(group: pd.DataFrame) -> dict:
    p = Params()
    by_model = {row["model"]: row for _, row in group.iterrows()}
    base = by_model["constant_suction"]
    full = by_model["full_hysteretic_damage"]
    plastic_scale = max(float(group["final_plastic_strain_index"].max()), 1.0e-9)
    stiffness_scale = max(float(group["stiffness_loss_pct"].max()), 1.0e-9)
    plastic = clipped((float(full["final_plastic_strain_index"]) - float(base["final_plastic_strain_index"])) / plastic_scale)
    stiffness = clipped((float(full["stiffness_loss_pct"]) - float(base["stiffness_loss_pct"])) / stiffness_scale)
    damage = clipped(
        0.5 * float(full["final_hydraulic_damage"]) / max(p.beta_h, 1.0e-9)
        + 0.5 * float(full["final_seismic_damage"]) / max(p.beta_c, 1.0e-9)
    )
    return {"hmai_plastic": plastic, "hmai_stiffness": stiffness, "hmai_damage": damage}


def external_hmai_table() -> pd.DataFrame:
    env = pd.read_csv(DATA / "kinikles2024_fig12a_digitized_envelope_summary.csv")
    rows = []
    for sr0 in env["group_initial_degree_saturation"].to_numpy(float):
        responses = pd.DataFrame([response_summary(model, sr0) for model in MODELS])
        comp = hmai_components(responses)
        base_hmai = sum(w * comp[name] for w, name in zip(WEIGHTS["base"], ["hmai_plastic", "hmai_stiffness", "hmai_damage"]))
        obs = env[np.isclose(env["group_initial_degree_saturation"].to_numpy(float), sr0)].iloc[0]
        rows.append(
            {
                "source": "Kinikles et al. 2024 Figure 12(a)",
                "initial_degree_saturation": sr0,
                **comp,
                "hmai_composite": clipped(base_hmai),
                "hmai_phase_class": phase(clipped(base_hmai)),
                "external_volumetric_strain_median_pct": float(obs["volumetric_strain_median_pct"]),
                "external_volumetric_strain_max_pct": float(obs["volumetric_strain_max_pct"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hmai_external_paired_cases.csv", index=False)
    return out


def correlation_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["external_volumetric_strain_median_pct", "external_volumetric_strain_max_pct"]:
        for predictor in ["hmai_composite", "hmai_plastic", "hmai_stiffness", "hmai_damage"]:
            x = pd.Series(paired[predictor])
            y = pd.Series(paired[target])
            if float(x.std()) == 0.0 or float(y.std()) == 0.0:
                rho = ""
                pearson = ""
            else:
                rho = float(x.corr(y, method="spearman"))
                pearson = float(x.corr(y, method="pearson"))
            rows.append(
                {
                    "source": "Kinikles et al. 2024 Figure 12(a)",
                    "predictor": predictor,
                    "external_target": target,
                    "spearman_rank_correlation": rho,
                    "pearson_correlation": pearson,
                    "interpretation": "Direct HMAI check against digitized figure-level cyclic compression evidence; weak or undefined correlations bound the HMAI claim.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hmai_external_correlation.csv", index=False)
    return out


def ablation_scorecard() -> pd.DataFrame:
    baseline = pd.read_csv(DATA / "external_baseline_comparison_scorecard.csv")
    rows = []
    for (source, target), group in baseline.groupby(["source", "target"]):
        winner = group.sort_values(["rmse_pct", "mae_pct"]).iloc[0]
        full = group[group["model"] == "full_hysteretic_damage"].iloc[0]
        constant = group[group["model"] == "constant_suction"].iloc[0]
        rows.append(
            {
                "source": source,
                "target": target,
                "best_model": winner["model"],
                "best_rmse_pct": float(winner["rmse_pct"]),
                "full_model_rmse_pct": float(full["rmse_pct"]),
                "constant_suction_rmse_pct": float(constant["rmse_pct"]),
                "full_beats_constant_suction": bool(float(full["rmse_pct"]) < float(constant["rmse_pct"])),
                "full_model_best": bool(winner["model"] == "full_hysteretic_damage"),
                "interpretation": "Ablation evidence supports HMAI only where removing hydraulic memory or damage worsens external fit; mixed outcomes are retained as boundaries.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hmai_external_ablation_scorecard.csv", index=False)
    return out


def weight_sensitivity(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_classes: dict[float, str] = {}
    for _, row in paired.iterrows():
        sr0 = float(row["initial_degree_saturation"])
        comp = [float(row["hmai_plastic"]), float(row["hmai_stiffness"]), float(row["hmai_damage"])]
        for name, weights in WEIGHTS.items():
            value = clipped(float(np.dot(weights, comp)))
            class_name = phase(value)
            if name == "base":
                base_classes[sr0] = class_name
            rows.append(
                {
                    "initial_degree_saturation": sr0,
                    "weight_variant": name,
                    "plastic_weight": weights[0],
                    "stiffness_weight": weights[1],
                    "damage_weight": weights[2],
                    "hmai_value": value,
                    "phase_class": class_name,
                }
            )
    out = pd.DataFrame(rows)
    out["matches_base_phase_class"] = out.apply(
        lambda r: bool(r["phase_class"] == base_classes[float(r["initial_degree_saturation"])]), axis=1
    )
    out.to_csv(DATA / "hmai_weight_sensitivity.csv", index=False)
    return out


def summary(corr: pd.DataFrame, ablation: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    rho_values = pd.to_numeric(corr["spearman_rank_correlation"], errors="coerce").dropna()
    out = pd.DataFrame(
        [
            {
                "hmai_external_correlation_rows": len(corr),
                "max_abs_spearman": float(rho_values.abs().max()) if len(rho_values) else "",
                "positive_spearman_count": int((rho_values > 0).sum()),
                "ablation_targets": len(ablation),
                "full_beats_constant_suction_count": int(ablation["full_beats_constant_suction"].sum()),
                "full_model_best_count": int(ablation["full_model_best"].sum()),
                "weight_sensitivity_rows": len(sensitivity),
                "phase_class_agreement_fraction": float(sensitivity["matches_base_phase_class"].mean()),
                "claim_boundary": "HMAI is externally constrained by figure-level trends, ablation outcomes and weight sensitivity; it is not an experimentally calibrated design index.",
            }
        ]
    )
    out.to_csv(DATA / "hmai_validation_summary.csv", index=False)


def main() -> None:
    paired = external_hmai_table()
    corr = correlation_table(paired)
    ablation = ablation_scorecard()
    sensitivity = weight_sensitivity(paired)
    summary(corr, ablation, sensitivity)
    print(f"hmai_external_validation_and_sensitivity=ok paired={len(paired)} weights={sensitivity['weight_variant'].nunique()}")


if __name__ == "__main__":
    main()
