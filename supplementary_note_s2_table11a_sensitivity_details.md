# Supplementary Note S2 - Detailed sensitivity and uncertainty checks

This note preserves the detailed sensitivity and uncertainty checks that were condensed in the main manuscript because the former detailed sensitivity table was too large for journal submission.

| Check | Reproducible evidence | Observed range or error | Interpretation boundary |
| --- | --- | --- | --- |
| Monte Carlo sensitivity | data/monte_carlo_sensitivity_summary.csv | Final plastic-strain index 0.00304-0.00678; mean 0.00461; std 0.00096. | Bounds output variability for the benchmark parameters; not identifiability or calibration. |
| Stiffness-loss sensitivity | data/monte_carlo_sensitivity_summary.csv | Stiffness loss 17.95-19.05%; mean 18.42%; std 0.284%. | Shows the reference stiffness-loss magnitude is not a single-point numerical artifact. |
| Time-step convergence | data/timestep_convergence.csv | Plastic-strain error falls from 1.23% at 360 steps to 0.059% at 1440 steps and 0 at 2880 steps; stiffness-loss error <=0.0119%. | Supports numerical robustness of the demanding full-model case. |
| Component verification | data/component_verification_summary.csv | Closed-form checks pass; maximum absolute error <=2.842e-14. | Supports implementation verification, not external validation. |
| Preconsolidation-pressure claim boundary | data/pc_consistency_audit_summary.csv | Damage-sensitive pc audit would differ by up to 27.44 kPa, so the primary benchmark uses suction-only pc_s(s). | Prevents overclaiming damage-coupled hardening as a calibrated law. |

The main manuscript now retains a compact three-row sensitivity table and points the reader to this supplementary note for the full details.
