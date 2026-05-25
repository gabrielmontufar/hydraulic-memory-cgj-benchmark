# Supplementary Note S1 - moved detailed tables and claim-boundary prose

Date: 2026-05-23

This note preserves detailed material moved from the main manuscript to improve editorial layout while keeping traceability.

## Supplementary trend-envelope outputs. Quantitative plausibility-envelope metrics used to bound the external evidence claim.

| External source/context | Benchmark metric | Reported benchmark value | Quantitative check | Allowed interpretation |
| --- | --- | --- | --- | --- |
| Dai and Zhou (2025) suction trend | Plastic-strain index at 0, 10 and 30 kPa | 0.007764; 0.005775; 0.002688 | Monotonic decrease with suction; normalized directional RMSE 0.0624 | Trend/plausibility only; not calibrated to raw data |
| Dai/Zhou and Ng et al. stiffness trend | Mean stiffness at 0, 10 and 30 kPa | 41.95; 43.41; 46.36 MPa | Monotonic increase with suction; +10.5% at 30 kPa relative to 0 kPa | Trend/plausibility only; not a fitted resilient-modulus curve |
| Howard (2021) path-dependence observation | Ablation at 100 kPa suction amplitude and 0.20 cyclic amplitude | constant suction 0.000165; no hysteresis 0.001936; hysteresis-only 0.004234; full 0.004521 | Path-dependent models differ strongly from constant-suction baseline | Mechanism check only; no dissertation figure digitization |
| Zenodo Italian clays RC/CTS context | Bounded stiffness degradation in reference benchmark | initial 44.10 MPa; final 35.99 MPa; loss 18.39% | Positive bounded stiffness retained under declared damage limits | Dynamic-envelope plausibility only; not clay-specific calibration |
| UNSODA/GSHP SWCC context | Physical-bounds and branch-order checks | physical_bounds_pass true; hysteresis_order_pass true | Retention branches remain ordered and physically bounded | Hydraulic envelope only; not cyclic mechanical calibration |
| Ng et al. resilient-modulus cyclic triaxial evidence | Semi-quantitative suction and cyclic-stress calibration stress test | Raw model ratio 0-30 kPa = 1.13 vs external about 2.0; two-point secondary suction correction gives 2.0 at 30 kPa and 2.2 at 60 kPa, but only 2.46 at 250 kPa against an external order-of-magnitude target. Cyclic-stress correction gives q70/q30 = 0.60 by calibration. | Partial semi-calibration succeeds at fitted low-suction and qcyc points; high-suction holdout fails. | Calibration-path evidence only; not blind cyclic-response validation or final MR calibration. |
| Gabr et al. unsaturated residual-soil triaxial holdout | Static suction-strength surrogate trained on multistage tests and held out on all single-stage tests | 86 transcribed triaxial stages from CC0 Mendeley Data; train n=66, holdout n=20; suction coefficient positive; holdout RMSE 53.3 kPa, MAPE 12.7%, R2 0.645. | Supports suction-strength direction and external holdout discipline, but only for static peak strength. | Auxiliary evidence only; not cyclic MR/permanent-strain validation. |


## Supplementary stiffness-range outputs. Raw Zenodo Italian clays range check used as an external stiffness-envelope control.

| Check | External data extracted | Model value checked | Interpretation boundary |
| --- | --- | --- | --- |
| Source and license | Zenodo record 3600964; Italian_Clays_Archive.xlsx; CC BY 4.0; raw workbook bundled in external_data/zenodo_italian_clays_3600964. | Not a fitted source. | Redistributable with attribution; included for transparent external range checking. |
| Workbook coverage used by script | Dataset sheet parsed into 2999 dynamic-data rows and 95 samples with dynamic rows. | No sample-specific fitting. | Counts describe the parsed dynamic rows used by the supplementary script, not a new experimental campaign. |
| Small-strain stiffness range | G0,c: n=1262; range 6.94-243.03 MPa; median 53.08 MPa; p05-p95 16.24-147.00 MPa. | Model stiffness endpoints 44.10 and 35.99 MPa. | Both endpoints lie within the raw external small-strain stiffness range. |
| Dynamic RC/CTS stiffness range | Dynamic G: n=3260; range 0.045-292.02 MPa; median 67.42 MPa; p05-p95 15.45-196.01 MPa. | Model stiffness endpoints 44.10 and 35.99 MPa. | Both endpoints lie within the raw external dynamic-stiffness range. |
| Damage-envelope magnitude | The external workbook supplies dynamic stiffness and damping ranges across RC/CTS steps, not matched suction paths. | Model stiffness loss is 18.39%. | This is a plausibility-envelope check only; it is not calibration to Italian clays. |
| Claim boundary | The script writes data/zenodo_italian_clays_range_check.csv and is run by run_all.py. | No calibrated prediction or blind validation is claimed. | Use as external range evidence, not as proof of a universal constitutive law. |


## Supplementary novelty and evidence outputs after the external-review benchmark

Supplementary novelty and evidence outputs after the external-review benchmark.

Evidence addedFile or scriptWhat it supportsWhat it does not support

Blind digitized-envelope Kinikles Fig. 12(a) holdoutscripts/external_kinikles2024_fig12a_blind_holdout.py; data/kinikles2024_fig12a_blind_holdout_summary.csvQuantifies leave-one-saturation-group-out error and coverage for an external cyclic figure envelope.Not raw cyclic time-history validation; not soil-specific calibration; not proof of peak-envelope robustness at all saturation states.

1D hydraulic-memory consequence proxyscripts/boundary_1d_hydraulic_memory_proxy.py; data/boundary_1d_case_summary.csv; data/boundary_1d_monte_carlo_summary.csvShows how hydraulic-memory assumptions change an equivalent stiffness/flexibility/deformation-demand indicator for a layered column.Not a site-response analysis; not design approval; not FEM validation of a new constitutive law.

Updated claim passportdata/table_boundary_value_claim_passport.csvConnects each new evidence item to allowed and blocked claims for reviewers.Does not expand the claim beyond reproducible benchmark, blind figure-level holdout and consequence proxy.


## Moved non-reference prose originally after References

Plausibility-Envelope and claim-boundary update. The revised supplementary package adds run_all.py, software validation, pc consistency auditing, external_data_manifest.csv, third_party_raw_data_note.md, monte_carlo_sensitivity_summary.csv, claim_boundary_negative_tests.csv and openseespy_cyclic_crosscheck.csv. These files support a reproducible benchmark, blind figure-level holdout and 1D consequence-proxy claim only. They do not support soil-specific calibration, full site-response analysis, 3D FEM validation, professional approval or seismic safety certification.

Hydraulic Memory Amplification Index update. The supplementary package includes `scripts/hydraulic_memory_amplification_index.py`, `data/hydraulic_memory_amplification_index.csv`, `data/hydraulic_memory_amplification_index_summary.csv` and `figures/fig20_hmai_heatmap.png`. HMAI is now reported as a composite phase-map contrast between the full hysteretic-damage model and the constant-suction baseline under the same scripted suction-amplitude/cyclic-amplitude case. The composite uses plastic-strain contrast, stiffness-loss contrast and normalized internal-damage contrast with weights 0.40, 0.30 and 0.30. The former plastic-only contrast is retained as a component column. Diagnostic phase classes are negligible below 0.10, relevant from 0.10 to 0.30 and dominant above 0.30. These classes are reproducible benchmark labels only; they are not calibrated design limits, raw cyclic time-history validation or site-response metrics.

External locked-validation protocol update. The supplementary package now includes `scripts/external_locked_validation_protocol.py`, `data/external_validation_protocol_registry.csv`, `data/external_locked_holdout_predictions.csv`, `data/external_locked_holdout_metrics.csv` and `data/external_validation_acceptance_matrix.csv`. The Kinikles Figure 12(a) envelope is split by initial saturation into dry/intermediate/wet training groups (S0 = 0.00, 0.20 and 0.56) and withheld validation groups (S0 = 0.12, 0.30 and 0.40). The median-envelope locked holdout passes the quantitative screen with RMSE = 0.218 percentage points, MAE = 0.215 percentage points, normalized RMSE = 0.188, normalized MAE = 0.185, Spearman rho = 1.00 and coverage = 1.00. The max-envelope holdout remains a boundary because coverage is 0.667 despite normalized RMSE = 0.172 and Spearman rho = 1.00. Rong/McCartney 200-cycle remains an explicit failure-domain boundary with Spearman rho = -0.486.

HMAI external-validation and sensitivity update. The package includes `scripts/hmai_external_validation_and_sensitivity.py`, `data/hmai_external_paired_cases.csv`, `data/hmai_external_correlation.csv`, `data/hmai_external_ablation_scorecard.csv`, `data/hmai_weight_sensitivity.csv` and `data/hmai_validation_summary.csv`. The strongest HMAI-related external correlation is the damage component against the Kinikles max envelope (Spearman rho = 0.880). The full model beats the constant-suction baseline in both external ablation targets, is the best model for the Rong/McCartney 200-cycle gate, and is not the best model for the Kinikles transfer gate where the no-hysteresis comparator has slightly lower RMSE. HMAI phase classes are unchanged under the tested base, equal, deformation-heavy, stiffness-heavy and damage-heavy weights.


## Moved non-reference prose originally after References

The effective preconsolidation pressure used in the cyclic threshold is reported separately from the suction-only hardening pressure. The benchmark stores pc_suction_only_kpa, damage_factor_pc, preconsolidation_effective_kpa, yield_strength_kpa and overstress so that readers can audit the threshold calculation directly.


## Moved non-reference prose originally after References

External evidence is treated by level. Dai and Zhou (2025) and Howard (2021) support unsaturated cyclic trend and plausibility-envelope checks; UNSODA and GSHP support hydraulic-envelope context; resonant-column and cyclic torsional-shear datasets support dynamic-stiffness context. These comparisons bound plausibility and reproducibility but do not constitute full calibration.
