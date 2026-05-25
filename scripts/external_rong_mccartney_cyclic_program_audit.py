from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "external_data" / "rong_mccartney_unsaturated_cyclic"

THESIS = RAW / "qt9wx3t712_noSplash_44f8caf288d18fc6f196549851c054b7.pdf"
PEER = RAW / "2022_05_mccartney_final.pdf"
E3S = RAW / "e3sconf_isg2019_08004.pdf"

THESIS_URL = "https://escholarship.org/content/qt9wx3t712/qt9wx3t712_noSplash_44f8caf288d18fc6f196549851c054b7.pdf"
PEER_URL = "https://peer.berkeley.edu/sites/default/files/2022_05_mccartney_final.pdf"
E3S_URL = "https://www.e3s-conferences.org/articles/e3sconf/pdf/2019/18/e3sconf_isg2019_08004.pdf"


def undrained_program() -> pd.DataFrame:
    # Rong dissertation Table 5.1: undrained cyclic simple-shear program.
    rows = [
        ("A-1", 20.04, 0.04, 0.940, 0.236, 0.379, 0.238),
        ("B-1", 20.03, 2.05, 0.558, 0.135, 0.214, 0.136),
        ("B-2", 19.85, 2.09, 0.549, 0.133, 0.211, 0.133),
        ("C-1", 19.95, 3.01, 0.400, 0.098, 0.156, 0.097),
        ("C-2", 19.39, 2.99, 0.401, 0.098, 0.157, 0.097),
        ("D-1", 19.96, 3.99, 0.308, 0.075, 0.119, 0.075),
        ("D-2", 19.76, 3.96, 0.310, 0.075, 0.120, 0.074),
        ("E-1", 19.89, 5.98, 0.204, 0.049, 0.079, 0.047),
        ("E-2", 19.78, 6.03, 0.200, 0.049, 0.078, 0.049),
        ("F-1", 19.92, 9.98, 0.117, 0.029, 0.046, 0.027),
        ("F-2", 19.78, 10.02, 0.117, 0.029, 0.046, 0.030),
        ("G-1", 19.94, 100.00, 0.000, 0.000, 0.000, 0.000),
    ]
    out = []
    for specimen, h0, psi0, sr0, w0, theta0, wf in rows:
        out.append(
            {
                "source": "Rong UCSD dissertation Table 5.1",
                "source_url": THESIS_URL,
                "local_file": str(THESIS.relative_to(ROOT)).replace("\\", "/"),
                "drainage_condition": "undrained",
                "specimen": specimen,
                "initial_height_mm": h0,
                "initial_matric_suction_kpa": psi0,
                "initial_degree_saturation": sr0,
                "initial_gravimetric_water_content": w0,
                "initial_volumetric_water_content": theta0,
                "final_gravimetric_water_content": wf,
                "vertical_total_stress_kpa": 50.0,
                "cyclic_shear_strain_amplitude_pct": 1.0,
                "cycles": 200,
                "time_history_public_as_table": False,
            }
        )
    return pd.DataFrame(out)


def drained_program() -> pd.DataFrame:
    # Rong and McCartney (2019) E3S Table 2: drained cyclic simple-shear program.
    rows = [
        ("A-1", 3.92, 0.31, 0.076, 0.121, 0.3),
        ("A-2", 3.87, 0.31, 0.077, 0.123, 1.0),
        ("A-3", 3.96, 0.31, 0.075, 0.120, 3.0),
        ("A-4", 4.02, 0.30, 0.074, 0.118, 5.0),
        ("B-1", 6.03, 0.20, 0.049, 0.078, 0.3),
        ("B-2", 5.93, 0.20, 0.050, 0.079, 1.0),
        ("B-3", 5.95, 0.20, 0.050, 0.079, 3.0),
        ("B-4", 5.88, 0.21, 0.050, 0.080, 5.0),
        ("C-1", 8.10, 0.15, 0.036, 0.057, 0.3),
        ("C-2", 8.01, 0.15, 0.036, 0.058, 1.0),
        ("C-3", 7.92, 0.15, 0.037, 0.058, 3.0),
        ("C-4", 7.98, 0.15, 0.036, 0.058, 5.0),
        ("D-1", 10.12, 0.12, 0.028, 0.045, 0.3),
        ("D-2", 10.15, 0.11, 0.028, 0.045, 1.0),
        ("D-3", 10.03, 0.12, 0.028, 0.045, 3.0),
        ("D-4", 9.94, 0.12, 0.029, 0.046, 5.0),
    ]
    out = []
    for specimen, psi0, sr0, w0, theta0, gamma in rows:
        out.append(
            {
                "source": "Rong and McCartney 2019 E3S Table 2",
                "source_url": E3S_URL,
                "local_file": str(E3S.relative_to(ROOT)).replace("\\", "/"),
                "drainage_condition": "drained_constant_suction",
                "specimen": specimen,
                "initial_matric_suction_kpa": psi0,
                "initial_degree_saturation": sr0,
                "initial_gravimetric_water_content": w0,
                "initial_volumetric_water_content": theta0,
                "vertical_total_stress_kpa": 50.0,
                "cyclic_shear_strain_amplitude_pct": gamma,
                "cycles": 200,
                "time_history_public_as_table": False,
            }
        )
    return pd.DataFrame(out)


def trend_claims() -> pd.DataFrame:
    rows = [
        {
            "source": "Rong UCSD dissertation / undrained CSS",
            "source_url": THESIS_URL,
            "evidence_type": "text plus figures; exact test program table",
            "reported_trend": "after 200 undrained cycles, volumetric strain has a nonlinear dependence on initial saturation/suction, with greatest seismic compression at intermediate degrees of saturation or suction",
            "article117_gate": "The current benchmark can cite this as the strongest direct cyclic target, but cannot claim a blind fit without digitizing or obtaining time histories.",
        },
        {
            "source": "Rong and McCartney 2019 E3S / drained CSS",
            "source_url": E3S_URL,
            "evidence_type": "open PDF text plus test-program table",
            "reported_trend": "larger cyclic shear strains give larger volumetric contractions, and higher suction specimens show lower final volumetric strain after 200 drained cycles",
            "article117_gate": "Supports suction-direction and cyclic-amplitude trend checks, not calibrated validation.",
        },
        {
            "source": "PEER 2022/05 McCartney report",
            "source_url": PEER_URL,
            "evidence_type": "open PDF report with model/data comparison figures",
            "reported_trend": "model/data comparisons show good early-cycle behaviour but deviations at 200 cycles, especially for volumetric strain and mean effective stress",
            "article117_gate": "Use as a digitization target for a future blind figure-based check with uncertainty bands.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    missing = [path for path in (THESIS, PEER, E3S) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Rong/McCartney PDF(s): " + ", ".join(map(str, missing)))
    DATA.mkdir(exist_ok=True)

    undrained = undrained_program()
    drained = drained_program()
    claims = trend_claims()

    undrained.to_csv(DATA / "rong2021_undrained_css_test_program.csv", index=False)
    drained.to_csv(DATA / "rong2019_drained_css_test_program.csv", index=False)
    claims.to_csv(DATA / "rong_mccartney_cyclic_trend_claims.csv", index=False)

    all_conditions = pd.concat(
        [
            undrained.assign(program="undrained"),
            drained.assign(program="drained"),
        ],
        ignore_index=True,
        sort=False,
    )
    summary = pd.DataFrame(
        [
            {
                "source_family": "Rong/McCartney unsaturated cyclic simple shear",
                "undrained_test_program_rows": len(undrained),
                "drained_test_program_rows": len(drained),
                "suction_min_kpa": float(all_conditions["initial_matric_suction_kpa"].min()),
                "suction_max_kpa": float(all_conditions["initial_matric_suction_kpa"].max()),
                "saturation_min": float(all_conditions["initial_degree_saturation"].min()),
                "saturation_max": float(all_conditions["initial_degree_saturation"].max()),
                "cyclic_strain_min_pct": float(all_conditions["cyclic_shear_strain_amplitude_pct"].min()),
                "cyclic_strain_max_pct": float(all_conditions["cyclic_shear_strain_amplitude_pct"].max()),
                "cycles": 200,
                "direct_cyclic_experiments_identified": True,
                "public_raw_time_histories_in_downloaded_sources": False,
                "mrnb_interpretation": "direct cyclic experimental state-space coverage is documented; a 95-100 validation score still requires digitized or raw response histories",
            }
        ]
    )
    summary.to_csv(DATA / "rong_mccartney_cyclic_program_summary.csv", index=False)
    print(f"rong_mccartney_cyclic_program_audit=ok undrained={len(undrained)} drained={len(drained)}")


if __name__ == "__main__":
    main()
