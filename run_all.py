from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def run(script: str) -> dict:
    start = time.perf_counter()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SCRIPTS) if not existing else str(SCRIPTS) + os.pathsep + existing
    proc = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=str(ROOT), text=True, capture_output=True, env=env)
    elapsed = time.perf_counter() - start
    return {
        "script": script,
        "returncode": proc.returncode,
        "elapsed_s": f"{elapsed:.3f}",
        "stdout": proc.stdout.strip().replace("\n", " | "),
        "stderr": proc.stderr.strip().replace("\n", " | "),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run the minimum extracted-package validation path.")
    parser.add_argument("--continue-on-error", action="store_true", help="Run every script and report failures at the end.")
    args = parser.parse_args()

    scripts = [
        "run_unsat_cyclic_benchmark.py",
        "hydro_validation_envelope.py",
        "external_zenodo_clay_range_check.py",
        "external_ng2013_resilient_modulus_check.py",
        "external_ng2013_semicalibrated_envelope.py",
        "external_gabr2019_unsaturated_triaxial_holdout.py",
        "external_kinikles2024_cyclic_simple_shear_gate.py",
        "external_kinikles2024_fig12a_digitized_envelope.py",
        "external_kinikles2024_fig12a_blind_holdout.py",
        "external_kinikles2024_model_output_transfer_holdout.py",
        "external_chen2024_supplementary_calibration_audit.py",
        "external_rong_mccartney_cyclic_program_audit.py",
        "external_rong_mccartney_200cycle_digitized_check.py",
        "external_rong_mccartney_drained_amplitude_digitized_gate.py",
        "external_rong_mccartney_200cycle_residual_diagnostics.py",
        "boundary_1d_hydraulic_memory_proxy.py",
        "parameter_identifiability_sensitivity.py",
        "hydraulic_memory_amplification_index.py",
        "hydraulic_memory_number_transition.py",
        "plot_hmai_heatmap.py",
        "mathematical_invariants_audit.py",
        "algorithmic_dissipation_audit.py",
        "theoretical_admissibility_tests.py",
        "external_cross_source_consistency_audit.py",
        "external_validation_scorecard.py",
        "external_baseline_comparison_scorecard.py",
        "external_locked_validation_protocol.py",
        "external_validation_protocol.py",
        "hmai_external_validation_and_sensitivity.py",
        "refresh_manifest_sha256.py",
        "software_validation.py",
    ]
    if args.quick:
        scripts = ["hydro_validation_envelope.py", "mathematical_invariants_audit.py", "software_validation.py"]

    rows = [run(script) for script in scripts]
    log = DATA / "run_all_log.csv"
    with log.open("w", encoding="utf-8", newline="") as f:
        f.write("script,returncode,elapsed_s,stdout,stderr\n")
        for row in rows:
            f.write(
                '"{script}",{returncode},{elapsed_s},"{stdout}","{stderr}"\n'.format(
                    **{k: str(v).replace('"', '""') for k, v in row.items()}
                )
            )
    failures = [row for row in rows if row["returncode"] != 0]
    print(f"scripts_run={len(rows)}")
    print(f"failures={len(failures)}")
    print(f"log={log}")
    if failures and not args.continue_on_error:
        return 1
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
