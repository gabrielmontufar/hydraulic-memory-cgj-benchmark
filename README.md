# Hydraulic-memory benchmark for cyclically loaded unsaturated soils

This repository contains the public reproducibility materials for the Canadian Geotechnical Journal manuscript, "Hydraulic-memory amplification as a regime variable in cyclically loaded unsaturated soils: a reproducible geotechnical phase-map benchmark."

It includes the benchmark scripts, generated CSV tables, generated figures, audit logs, validation scorecards, HMAI/HMN outputs, metadata and supplementary notes. The complete submission package also contains a local supplementary archive submitted with the manuscript.

Run `python run_all.py --continue-on-error` from the repository root to regenerate the benchmark outputs when all optional third-party raw-source files are available. The repository keeps generated evidence and source metadata public, but it does not redistribute third-party raw PDFs, figures, workbooks or supplementary files unless reuse rights are explicit. Those source records are documented in `data/external_data_manifest.csv` and `third_party_raw_data_note.md`.

Models compared: constant suction baseline, no-hysteresis suction model, hysteresis-only model, and full hysteretic-damage model.

The file `data/external_validation_trend_checks.csv` records external trend-consistency sources used in the manuscript. It is a qualitative plausibility table, not a calibrated experimental data set.

If `external_data/zenodo_italian_clays_3600964/Italian_Clays_Archive.xlsx` is present in a local working copy, `scripts/external_zenodo_clay_range_check.py` regenerates `data/zenodo_italian_clays_range_check.csv` as a CC BY 4.0 external dynamic-stiffness range check. This check is not a calibration or blind validation.

`scripts/external_kinikles2024_cyclic_simple_shear_gate.py` records a recent direct-cyclic validation target from Kinikles, Rong and McCartney (2024) and Chen et al. (2024). The generated CSV files are a state-of-art comparison gate and a test-condition matrix, not a blind calibration against raw cyclic time histories.

`scripts/external_kinikles2024_fig12a_digitized_envelope.py` digitizes Kinikles et al. (2024) Figure 12(a) with explicit pixel coordinates, axis calibration and uncertainty. The generated files provide a partial direct-cyclic figure envelope, not raw response histories.

`scripts/external_chen2024_supplementary_calibration_audit.py` parses the public Chen et al. (2024) supplementary Word file with standard-library DOCX XML parsing. It extracts the supplementary-table inventory, SPARC sand properties and monotonic test conditions, and records that raw cyclic time histories were not found in the supplementary file.

`scripts/external_rong_mccartney_cyclic_program_audit.py` records exact drained and undrained cyclic simple-shear test-program tables from Rong/McCartney open sources. These CSV files document direct experimental state-space coverage and trend gates; they are not raw response time histories.

`scripts/external_rong_mccartney_drained_amplitude_digitized_gate.py` adds a second Rong/McCartney figure-level digitization from drained CSS Figure 4.5. It checks whether the benchmark response preserves the observed increase in 200-cycle volumetric strain with cyclic shear-strain amplitude. The generated files are an external-evidence amplitude gate, not raw time-history validation or cyclic-shear-strain calibration.

`scripts/external_rong_mccartney_200cycle_residual_diagnostics.py` diagnoses the weak Rong/McCartney 200-cycle result. It reports residuals, leave-one-saturation-group-out transfer errors, rank behavior and the specific failure mode behind the adverse rank agreement. This is a boundary diagnostic, not a calibrated long-cycle validation.

`scripts/external_cross_source_consistency_audit.py` aggregates independent external-evidence layers into a claim-consistency matrix. It records pass, boundary and review outcomes across Kinikles, Rong/McCartney, Gabr, Ng/Zhou and Zenodo evidence, explicitly retaining the absence of public raw cyclic response histories.

`scripts/external_locked_validation_protocol.py` formalizes the external-validation protocol into a source registry, a locked 50/50 Kinikles saturation-group holdout, acceptance metrics and explicit boundary rows for Rong/McCartney, Ng/Zhou and Dai/Zhou. The Kinikles median-envelope holdout passes the predeclared quantitative screen; the max-envelope and Rong 200-cycle checks remain boundaries.

`scripts/boundary_1d_hydraulic_memory_proxy.py` adds a reproducible one-dimensional boundary-value proxy. It translates material-point hydraulic-memory outcomes into equivalent stiffness, flexibility and relative deformation-demand indicators for a layered column. This is a practical geotechnical consequence metric, not a calibrated site-response analysis or FEM validation of a constitutive law.

`scripts/parameter_identifiability_sensitivity.py` converts the Monte Carlo perturbation set into a local identifiability screen with standardized coefficients, Pearson/Spearman correlations and permutation importance. It is a local sensitivity audit around the declared benchmark range, not a global calibration or formal inverse analysis.

`scripts/hydraulic_memory_amplification_index.py` computes the Hydraulic Memory Amplification Index (HMAI), a composite bounded phase-map contrast between the hysteresis-only hydraulic-memory branch and the no-hysteresis suction branch under the same scripted demand. The reported composite combines plastic-strain contrast (0.40), stiffness-loss contrast (0.30), and normalized hydraulic-path intensity (0.30), while retaining the former full-vs-constant contrast as an audit column. HMAI is a diagnostic benchmark index, not a calibrated design factor or raw-data validation.

`scripts/hydraulic_memory_number_transition.py` computes the Hydraulic Memory Number (HMN), diagnostic HMAI/HMN transition tables, claim-boundary checks, and a originality matrix. HMN is a reproducible benchmark number, not a universal constitutive parameter or design coefficient.

`scripts/hmai_external_validation_and_sensitivity.py` pairs HMAI with Kinikles figure-level external response groups, reports HMAI-external correlations, compares ablated model variants against external gates and checks whether HMAI phase classes persist under alternative component weights.

Additional QA files include `data/timestep_convergence.csv` and `data/model_parameters.csv`.
