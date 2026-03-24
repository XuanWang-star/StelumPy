"""
stellarmod.io.model
-------------------
Reads and parses a single stellar model profile file.
"""

import logging

import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class Model:
    """
    Read and hold the data from one stellar model profile file.

    Attributes
    ----------
    file_path : Path
    data_type : str
    mesh_number : int
    T_eff : float
    log_g : float
    df : pd.DataFrame   All mesh-point data (columns defined by COLUMN_NAMES).
    """

    COLUMN_NAMES = [
        'n', 'r', 'm_r', 'rho', 'P', 'T', 'chi_p', 'chi_T', 'Del', 'Del_ad',
        'Y', 'B', 'log_q', 'flag14', 'flag15', 'unknown',
        'L', 'eta', 'eta_r', 'kappa', 'kappa_rho', 'kappa_T', 'Del_rad',
        'zeta', 'epsilon_N', 'epsilon_rho', 'epsilon_T',
        'tau', 'w', 'wtau', 'Del_r', 'Del_P', 'Del_T', 'Del_tau', 'Del_L',
        'Del_ad_P', 'Del_ad_T', 'Del_ad_minus_Del',
        'U', 'dU_dP', 'dU_dT', 'C_p', 'C_V', 'eta_e', 'Z_moy', 'Gamma',
        'X_H', 'X_He', 'X_C', 'X_O', 'log_P_gas', 'unknown1', 'unknown2', 'unknown3',
    ]
    
    OPTIONAL_COLUMNS = ['unknown1', 'unknown2', 'unknown3']

    BLOCK_SIZE = 5  # data lines per mesh point

    # ------------------------------------------------------------------
    # Construction / I/O
    # ------------------------------------------------------------------

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.data_type: str | None = None
        self.mesh_number: int | None = None
        self.T_eff: float | None = None
        self.log_g: float | None = None
        self.df: pd.DataFrame | None = None
        self._read_file()

    def _read_file(self) -> None:
        with open(self.file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        self._parse_header(lines)
        self._parse_data(lines[2:])

    def _parse_header(self, lines: list[str]) -> None:
        self.data_type = lines[0]
        parts = lines[1].split()
        self.mesh_number = int(parts[0])
        self.log_g = float(parts[3])
        self.T_eff = float(parts[5])

    def _parse_data(self, data_lines):
        all_steps_data = []

        for step in range(self.mesh_number):
            start_idx = step * self.BLOCK_SIZE
            block = data_lines[start_idx:start_idx + self.BLOCK_SIZE]

            all_values = []
            for line in block:
                all_values.extend(map(float, line.split()))

            all_steps_data.append(all_values)

    
        actual_col_count = len(all_steps_data[0]) if all_steps_data else 0
        expected_new = len(self.COLUMN_NAMES)                        # 54列（新版）
        expected_old = expected_new - len(self.OPTIONAL_COLUMNS)     # 51列（老版）
    
        if actual_col_count == expected_new:
            columns = self.COLUMN_NAMES
        elif actual_col_count == expected_old:
            columns = self.COLUMN_NAMES[:-len(self.OPTIONAL_COLUMNS)]
        else:
            raise ValueError(
                f"Unexpected number of columns: {actual_col_count}. "
                f"Expected {expected_old} (old) or {expected_new} (new)."
            )
    
        self.df = pd.DataFrame(all_steps_data, columns=columns)
    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def get_column(self, column_name: str) -> pd.Series:
        """Return a single column from the profile DataFrame."""
        if self.df is None:
            raise ValueError("No data loaded.")
        return self.df[column_name]

    def he_core_he(self, n_points: int = 10) -> float:
        """
        Mean X_He over the *n* innermost mesh points (index 0 = centre).
        Used as a proxy for the core helium abundance.
        """
        if self.df is None:
            raise ValueError("No profile data loaded.")
        return float(self.df['X_He'].iloc[:n_points].mean())

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a summary of the model to stdout."""
        sep = '=' * 50
        lines = [
            sep,
            "Model Summary",
            sep,
            f"File:        {self.file_path.name}",
            f"Data type:   {self.data_type}",
            f"Mesh points: {self.mesh_number}",
            f"T_eff:       {self.T_eff:.2f} K",
            f"log(g):      {self.log_g:.2f}",
            sep,
        ]
        if self.df is not None:
            lines.append(f"DataFrame shape: {self.df.shape}")
        logger.info("\n".join(lines))
        # Also print for backward compatibility
        for line in lines:
            print(line)
        if self.df is not None:
            print(self.df.head())

    def __repr__(self) -> str:
        return (
            f"Model('{self.file_path.name}', "
            f"mesh={self.mesh_number}, "
            f"T_eff={self.T_eff}, "
            f"log_g={self.log_g})"
        )
