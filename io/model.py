"""
stellarmod.io.model
-------------------
Reads and parses a single stellar model profile file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..exceptions import ModelFileError, ProfileColumnError, ValidationError

logger = logging.getLogger(__name__)


class Model:
    """
    Read and hold the data from one stellar model profile file.

    Parameters
    ----------
    file_path : str or Path
        Path to the stellar model file.

    Attributes
    ----------
    file_path : Path
        Path to the model file.
    data_type : str or None
        Data type identifier from file header.
    mesh_number : int or None
        Number of mesh points in the model.
    T_eff : float or None
        Effective temperature in Kelvin.
    log_g : float or None
        Surface gravity (log g).
    df : pd.DataFrame or None
        DataFrame containing all mesh-point data.

    Examples
    --------
    >>> model = Model("path/to/model.txt")  # doctest: +SKIP
    >>> model.T_eff  # doctest: +SKIP
    5770.0
    >>> model.he_core_he(n_points=5)  # doctest: +SKIP
    0.85
    """

    COLUMN_NAMES: list[str] = [
        'n', 'r', 'm_r', 'rho', 'P', 'T', 'chi_p', 'chi_T', 'Del', 'Del_ad',
        'Y', 'B', 'log_q', 'flag14', 'flag15', 'unknown',
        'L', 'eta', 'eta_r', 'kappa', 'kappa_rho', 'kappa_T', 'Del_rad',
        'zeta', 'epsilon_N', 'epsilon_rho', 'epsilon_T',
        'tau', 'w', 'wtau', 'Del_r', 'Del_P', 'Del_T', 'Del_tau', 'Del_L',
        'Del_ad_P', 'Del_ad_T', 'Del_ad_minus_Del',
        'U', 'dU_dP', 'dU_dT', 'C_p', 'C_V', 'eta_e', 'Z_moy', 'Gamma',
        'X_H', 'X_He', 'X_C', 'X_O', 'log_P_gas', 'unknown1', 'unknown2', 'unknown3',
    ]

    OPTIONAL_COLUMNS: set[str] = {'unknown1', 'unknown2', 'unknown3'}

    BLOCK_SIZE: int = 5  # data lines per mesh point

    def __init__(self, file_path: str | Path) -> None:
        self.file_path: Path = Path(file_path)
        self.data_type: str | None = None
        self.mesh_number: int | None = None
        self.T_eff: float | None = None
        self.log_g: float | None = None
        self.df: pd.DataFrame | None = None
        self._read_file()

    def _read_file(self) -> None:
        """Read and parse the model file."""
        try:
            with open(self.file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError as exc:
            raise ModelFileError(f"Model file not found: {self.file_path}") from exc
        except IOError as exc:
            raise ModelFileError(f"Error reading model file: {self.file_path}") from exc

        self._parse_header(lines)
        self._parse_data(lines[2:])

    def _parse_header(self, lines: list[str]) -> None:
        """Parse the header lines of the model file."""
        if len(lines) < 2:
            raise ModelFileError(
                f"Model file {self.file_path} has insufficient header lines"
            )

        self.data_type = lines[0]
        parts = lines[1].split()

        if len(parts) < 6:
            raise ModelFileError(
                f"Model file {self.file_path} has invalid header format"
            )

        try:
            self.mesh_number = int(parts[0])
            self.log_g = float(parts[3])
            self.T_eff = float(parts[5])
        except ValueError as exc:
            raise ModelFileError(
                f"Model file {self.file_path} has invalid header values"
            ) from exc

    def _parse_data(self, data_lines: list[str]) -> None:
        """Parse the data lines of the model file."""
        if self.mesh_number is None or self.mesh_number == 0:
            self.df = pd.DataFrame(columns=self.COLUMN_NAMES[:-len(self.OPTIONAL_COLUMNS)])
            return

        all_steps_data: list[list[float]] = []

        for step in range(self.mesh_number):
            start_idx = step * self.BLOCK_SIZE
            block = data_lines[start_idx:start_idx + self.BLOCK_SIZE]

            if len(block) < self.BLOCK_SIZE:
                raise ModelFileError(
                    f"Insufficient data lines for mesh point {step} in {self.file_path}"
                )

            all_values: list[float] = []
            for line in block:
                try:
                    values = [float(v) for v in line.split()]
                    all_values.extend(values)
                except ValueError as exc:
                    raise ModelFileError(
                        f"Invalid data format at mesh point {step} in {self.file_path}"
                    ) from exc

            all_steps_data.append(all_values)

        if not all_steps_data:
            raise ModelFileError(f"No data parsed from {self.file_path}")

        actual_col_count = len(all_steps_data[0])
        expected_new = len(self.COLUMN_NAMES)                        # 54 columns (new)
        expected_old = expected_new - len(self.OPTIONAL_COLUMNS)     # 51 columns (old)

        if actual_col_count == expected_new:
            columns = self.COLUMN_NAMES
        elif actual_col_count == expected_old:
            columns = self.COLUMN_NAMES[:-len(self.OPTIONAL_COLUMNS)]
        else:
            raise ModelFileError(
                f"Unexpected number of columns in {self.file_path}: {actual_col_count}. "
                f"Expected {expected_old} (old format) or {expected_new} (new format)."
            )

        self.df = pd.DataFrame(all_steps_data, columns=columns)

    def get_column(self, column_name: str) -> pd.Series:
        """
        Return a single column from the profile DataFrame.

        Parameters
        ----------
        column_name : str
            Name of the column to retrieve.

        Returns
        -------
        pd.Series
            The requested column data.

        Raises
        ------
        ProfileColumnError
            If the column does not exist or data is not loaded.
        """
        if self.df is None:
            raise ProfileColumnError("No profile data loaded")
        if column_name not in self.df.columns:
            raise ProfileColumnError(
                f"Column '{column_name}' not found. "
                f"Available columns: {list(self.df.columns)}"
            )
        return self.df[column_name]

    def he_core_he(self, n_points: int = 10) -> float:
        """
        Calculate mean X_He over the n innermost mesh points.

        Parameters
        ----------
        n_points : int, optional
            Number of innermost mesh points to average (default: 10).

        Returns
        -------
        float
            Mean helium abundance in the core.

        Raises
        ------
        ValidationError
            If n_points is invalid.
        ProfileColumnError
            If profile data is not loaded.
        """
        if self.df is None:
            raise ProfileColumnError("No profile data loaded")
        if n_points < 1:
            raise ValidationError(f"n_points must be >= 1, got {n_points}")
        if n_points > len(self.df):
            raise ValidationError(
                f"n_points ({n_points}) exceeds mesh points ({len(self.df)})"
            )
        return float(self.df['X_He'].iloc[:n_points].mean())

    @property
    def center_X_He(self) -> float:
        """Convenience property for central helium abundance."""
        return self.he_core_he(n_points=1)

    @property
    def core_X_He(self) -> float:
        """Convenience property for core helium abundance (10 points)."""
        return self.he_core_he(n_points=10)

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
