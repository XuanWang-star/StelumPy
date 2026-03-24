"""
StelumPy Test Helpers
=====================
Helper functions for creating test data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_test_model_file(directory: Path, filename: str = "md001.txt") -> Path:
    """Create a minimal valid stellar model file for testing (54 columns)."""
    filepath = directory / filename

    # 54 columns across 5 lines per mesh point: 11+11+11+11+10 = 54
    # Header line 2 format: mesh_num  ?  ?  log_g  ?  T_eff
    # positions:            0        1  2  3      4  5

    lines = ["PROFILE", "  10   1.0000E+00   0.0000E+00   4.5000E+00   0.0000E+00   5.7700E+03"]

    for i in range(1, 11):
        # Line 1: 11 values (n, r, m_r, rho, P, T, chi_p, chi_T, Del, Del_ad, Y)
        line1 = f"  {i}  {i*1.0E+10:.6E}  {i*1.0E+00:.6E}  {i*1.0E+02:.6E}  {i*1.0E+15:.6E}  {i*1.0E+07:.6E}  0.500  0.300  0.700  0.001  0.002"
        # Line 2: 11 values (B, log_q, flag14, flag15, unknown, L, eta, eta_r, kappa, kappa_rho, kappa_T)
        line2 = f"  0.003  {i*1.0E+01:.6E}  1.000  2.000  3.000  4.000  5.000  6.000  7.000  8.000  9.000"
        # Line 3: 11 values (Del_rad, zeta, epsilon_N, epsilon_rho, epsilon_T, tau, w, wtau, Del_r, Del_P, Del_T)
        line3 = f"  1.000  2.000  3.000  4.000  5.000  6.000  7.000  8.000  9.000  1.000  2.000"
        # Line 4: 11 values (Del_tau, Del_L, Del_ad_P, Del_ad_T, Del_ad_minus_Del, U, dU_dP, dU_dT, C_p, C_V, eta_e)
        line4 = f"  3.000  4.000  5.000  6.000  7.000  8.000  9.000  1.000  2.000  3.000  4.000"
        # Line 5: 10 values (Z_moy, Gamma, X_H, X_He, X_C, X_O, log_P_gas, unknown1, unknown2, unknown3)
        # X_He decreases from 0.98 at center to 0.28 at surface
        x_he = 0.98 - (i - 1) * 0.07
        line5 = f"  5.000  6.000  0.700  {x_he:.3f}  0.003  0.009  {i*1.0E+15:.6E}  0.001  0.002  0.003"

        lines.extend([line1, line2, line3, line4, line5])

    filepath.write_text("\n".join(lines))
    return filepath


def create_test_seq_file(directory: Path, num_models: int = 3) -> Path:
    """Create a minimal seq.txt file for testing (5 header lines like real data)."""
    filepath = directory / "seq.txt"

    # 5 header lines + data lines (matches real seq.txt format)
    # skiprows=5 will skip these 5 lines, then read data with column names

    lines = [
        "SEQ",
        "  STELLAR EVOLUTION SEQUENCE",
        "  INITIAL MASS: 5.0 MSUN",
        "  METALLICITY: Z=0.02",
        "  DATE: 2024-01-01",
    ]

    for i in range(1, num_models + 1):
        age = 1.0e6 * i
        teff = 5000 + i * 100
        log_g = 4.5 - i * 0.1
        lines.append(
            f"{i:3d}  {teff:8.1f}  {log_g:7.2f}  {1.0+i*0.1:7.2f}  "
            f"{age:10.2e}  {100+i:7.1f}  {7.0:7.2f}  {17.0:7.2f}  "
            f"{5.0:9.2f}  {5.0:7.2f}  {-2.0:7.2f}  {0.0:7.2f}  "
            f"{0.0:7.2f}  {3.5:7.2f}  {1.0:7.2f}  {3.0:7.2f}"
        )

    filepath.write_text("\n".join(lines))
    return filepath


def create_test_sequence_directory(tmp_path: Path, num_models: int = 3) -> Path:
    """Create a complete test sequence directory structure."""
    seq_dir = tmp_path / "test_seq"
    models_dir = seq_dir / "5mext"
    models_dir.mkdir(parents=True)

    create_test_seq_file(seq_dir, num_models)

    for i in range(1, num_models + 1):
        create_test_model_file(models_dir, f"md{i:03d}.txt")

    return seq_dir
