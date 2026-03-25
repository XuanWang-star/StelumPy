"""
stellarmod.analysis.matching
-----------------------------
SequenceAnalyzer — all matching, statistics, and data-extraction methods
for a Sequence of stellar models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import MatchingError, ProfileColumnError, ValidationError
from ..io.model import Model
from ..io.sequence import Sequence

logger = logging.getLogger(__name__)


class SequenceAnalyzer:
    """
    Perform matching and data-extraction operations on a Sequence.

    Parameters
    ----------
    sequence : Sequence
        The loaded model sequence to analyse.

    Attributes
    ----------
    seq : Sequence
        The model sequence being analysed.

    Examples
    --------
    >>> analyzer = SequenceAnalyzer(seq)  # doctest: +SKIP
    >>> result = analyzer.find_model_by_he_core(0.5)  # doctest: +SKIP
    >>> metrics = analyzer.he_profile_match(target, result['model'])  # doctest: +SKIP
    """

    def __init__(self, sequence: Sequence) -> None:
        self.seq: Sequence = sequence

    def get_evolution_data(self, parameter: str) -> np.ndarray:
        """
        Return an array of parameter values, one per model.

        For header parameters ('T_eff', 'log_g', 'mesh_number', 'data_type')
        the attribute is read directly. For profile columns the central
        value (row 0) is used.

        Parameters
        ----------
        parameter : str
            Name of the parameter to extract.

        Returns
        -------
        np.ndarray
            Array of parameter values.
        """
        header_params: set[str] = {'T_eff', 'log_g', 'mesh_number', 'data_type'}
        values: list[float] = []

        for model in self.seq.models:
            if parameter in header_params:
                attr_value = getattr(model, parameter, None)
                values.append(float(attr_value) if attr_value is not None else np.nan)
            elif model.df is not None and parameter in model.df.columns:
                values.append(float(model.df[parameter].iloc[0]))
            else:
                values.append(np.nan)

        return np.array(values)

    def get_profile_evolution(
        self,
        column_name: str,
        mesh_point: str | int = 'center',
    ) -> np.ndarray:
        """
        Return a profile column value at a fixed mesh point across all models.

        Parameters
        ----------
        column_name : str
            Name of the profile column.
        mesh_point : 'center' | 'surface' | int
            Mesh point specification:
            - 'center': use row 0 (innermost point)
            - 'surface': use last row
            - int: use specific row index (0-based)

        Returns
        -------
        np.ndarray
            Array of column values at the specified mesh point.
        """
        values: list[float] = []

        for model in self.seq.models:
            if model.df is None or column_name not in model.df.columns:
                values.append(np.nan)
                continue

            if mesh_point == 'center':
                values.append(float(model.df[column_name].iloc[0]))
            elif mesh_point == 'surface':
                values.append(float(model.df[column_name].iloc[-1]))
            elif isinstance(mesh_point, int):
                if 0 <= mesh_point < len(model.df):
                    values.append(float(model.df[column_name].iloc[mesh_point]))
                else:
                    values.append(np.nan)
            else:
                values.append(np.nan)

        return np.array(values)

    def get_he_core_evolution(self, n_points: int = 5) -> np.ndarray:
        """
        Calculate mean X_He over the n innermost mesh points for every model.

        Parameters
        ----------
        n_points : int, optional
            Number of innermost mesh points to average (default: 5).

        Returns
        -------
        np.ndarray
            Array of core helium abundances.
        """
        return np.array([m.he_core_he(n_points) for m in self.seq.models])

    def create_evolution_dataframe(self, parameters: list[str]) -> pd.DataFrame:
        """
        Build a DataFrame with the Age column plus one column per parameter.

        Parameters
        ----------
        parameters : list[str]
            List of parameter names to include.

        Returns
        -------
        pd.DataFrame
            DataFrame with Age and requested parameters.
        """
        data: dict[str, np.ndarray] = {}
        if self.seq.age_sequence is not None:
            data['Age'] = self.seq.age_sequence
        for p in parameters:
            data[p] = self.get_evolution_data(p)
        return pd.DataFrame(data)

    def find_model_by_he_core(
        self,
        target_he: float,
        n_points: int = 10,
    ) -> dict:
        """
        Find the single model whose core He value is closest to target_he.

        Parameters
        ----------
        target_he : float
            Target core helium abundance (0-1).
        n_points : int, optional
            Number of innermost mesh points for core He calculation (default: 10).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'index': 0-based position in sequence
            - 'model': the matched Model object
            - 'he_core': actual core He value
            - 'delta': signed difference (he_core - target_he)
            - 'age': stellar age (or NaN if unavailable)
            - 'T_eff': effective temperature
            - 'log_g': surface gravity

        Raises
        ------
        MatchingError
            If no models are loaded.
        """
        if not self.seq.models:
            raise MatchingError("No models loaded in sequence")

        he = self.get_he_core_evolution(n_points)
        idx = int(np.argmin(np.abs(he - target_he)))
        m = self.seq.models[idx]

        return {
            'index':   idx,
            'model':   m,
            'he_core': float(he[idx]),
            'delta':   float(he[idx] - target_he),
            'age':     float(self.seq.age_sequence[idx]) if self.seq.age_sequence is not None else np.nan,
            'T_eff':   m.T_eff,
            'log_g':   m.log_g,
        }

    def find_models_by_he_core(
        self,
        target_he: float,
        n_models: int = 5,
        n_points: int = 10,
    ) -> list[dict]:
        """
        Return the n models whose core He value is closest to target_he.

        Parameters
        ----------
        target_he : float
            Target core helium abundance (0-1).
        n_models : int, optional
            Number of models to return (default: 5).
        n_points : int, optional
            Number of innermost mesh points for core He calculation (default: 10).

        Returns
        -------
        list[dict]
            List of dictionaries (same structure as find_model_by_he_core),
            sorted by |delta| ascending.

        Raises
        ------
        MatchingError
            If no models are loaded.
        """
        if not self.seq.models:
            raise MatchingError("No models loaded in sequence")

        he = self.get_he_core_evolution(n_points)
        n = min(n_models, len(self.seq.models))
        results: list[dict] = []

        for idx in np.argsort(np.abs(he - target_he))[:n]:
            m = self.seq.models[int(idx)]
            results.append({
                'index':   int(idx),
                'model':   m,
                'he_core': float(he[idx]),
                'delta':   float(he[idx] - target_he),
                'age':     float(self.seq.age_sequence[idx]) if self.seq.age_sequence is not None else np.nan,
                'T_eff':   m.T_eff,
                'log_g':   m.log_g,
            })

        return results

    def he_profile_match(
        self,
        target_model: Model,
        snapshot_model: Model,
        n_points: int = 200,
        plot: bool = False,
        ax=None,
    ) -> dict:
        """
        Quantify the X_He profile match between target and snapshot models.

        Both profiles are interpolated onto a common log_q grid so that
        different mesh resolutions are handled correctly.

        Parameters
        ----------
        target_model : Model
            The reference (target) model.
        snapshot_model : Model
            The snapshot model to compare.
        n_points : int, optional
            Number of interpolation points (default: 200).
        plot : bool, optional
            Whether to create a plot (default: False).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (created if None).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'rmse': root-mean-square error of ΔX_He
            - 'mae': mean absolute error
            - 'r2': coefficient of determination (1 = perfect match)
            - 'max_dev': maximum absolute deviation

        Raises
        ------
        ProfileColumnError
            If models don't have required columns.
        """
        for label, m in [('target_model', target_model),
                          ('snapshot_model', snapshot_model)]:
            if m.df is None:
                raise ProfileColumnError(f"{label} has no profile data loaded")
            for col in ('X_He', 'log_q'):
                if col not in m.df.columns:
                    raise ProfileColumnError(
                        f"{label} is missing column '{col}'"
                    )

        def _prepare(model: Model) -> tuple[np.ndarray, np.ndarray]:
            df = model.df[['log_q', 'X_He']].copy()
            df = df.sort_values('log_q').drop_duplicates(subset='log_q', keep='last')
            return df['log_q'].values, df['X_He'].values

        q_tgt,  y_tgt  = _prepare(target_model)
        q_snap, y_snap = _prepare(snapshot_model)

        q_min  = max(q_tgt.min(),  q_snap.min())
        q_max  = min(q_tgt.max(),  q_snap.max())
        q_grid = np.linspace(q_min, q_max, n_points)

        y1 = np.interp(q_grid, q_tgt,  y_tgt)
        y2 = np.interp(q_grid, q_snap, y_snap)

        diff    = y1 - y2
        rmse    = float(np.sqrt(np.mean(diff ** 2)))
        mae     = float(np.mean(np.abs(diff)))
        max_dev = float(np.max(np.abs(diff)))
        ss_res  = float(np.sum(diff ** 2))
        ss_tot  = float(np.sum((y1 - y1.mean()) ** 2))
        r2      = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

        if plot:
            import matplotlib.pyplot as plt
            from ..plotting.plots import SequencePlotter
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=(9, 5))
            SequencePlotter._draw_he_profile_match(
                ax, q_grid, y1, y2, rmse,
                Path(target_model.file_path).name,
                Path(snapshot_model.file_path).name,
            )
            if created:
                plt.tight_layout()
                plt.show()

        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'max_dev': max_dev}

    def _normalize_teff_logg(
        self,
        T_eff_target: float,
        log_g_target: float,
        weight_T_eff: float = 1.0,
        weight_log_g: float = 1.0,
    ) -> np.ndarray:
        """
        Return normalised weighted Euclidean distances.

        Parameters
        ----------
        T_eff_target : float
            Target effective temperature.
        log_g_target : float
            Target surface gravity.
        weight_T_eff : float, optional
            Weight for T_eff term (default: 1.0).
        weight_log_g : float, optional
            Weight for log_g term (default: 1.0).

        Returns
        -------
        np.ndarray
            Array of normalised distances.
        """
        T_arr = np.array([m.T_eff for m in self.seq.models], dtype=float)
        g_arr = np.array([m.log_g for m in self.seq.models], dtype=float)

        # Use ddof=1 for sample std and add small epsilon to avoid division by zero
        T_std = T_arr.std(ddof=1) if len(T_arr) > 1 else 1.0
        g_std = g_arr.std(ddof=1) if len(g_arr) > 1 else 1.0
        T_mean = T_arr.mean()
        g_mean = g_arr.mean()

        # Avoid division by zero
        T_std = max(T_std, 1e-10)
        g_std = max(g_std, 1e-10)

        T_n = (T_arr - T_mean) / T_std
        g_n = (g_arr - g_mean) / g_std
        T_t = (T_eff_target - T_mean) / T_std
        g_t = (log_g_target - g_mean) / g_std

        return np.sqrt(weight_T_eff * (T_n - T_t) ** 2 + weight_log_g * (g_n - g_t) ** 2)

    def _teff_logg_dict(
        self,
        idx: int,
        distances: np.ndarray,
        T_eff_target: float,
        log_g_target: float,
    ) -> dict:
        """Build result dictionary for T_eff/log_g matching."""
        m = self.seq.models[idx]
        return {
            'index':    int(idx),
            'model':    m,
            'T_eff':    m.T_eff,
            'log_g':    m.log_g,
            'age':      float(self.seq.age_sequence[idx]) if self.seq.age_sequence is not None else np.nan,
            'distance': float(distances[idx]),
            'dT_eff':   m.T_eff - T_eff_target,
            'dlog_g':   m.log_g - log_g_target,
        }

    def find_closest_model(
        self,
        T_eff_target: float,
        log_g_target: float,
        weight_T_eff: float = 1.0,
        weight_log_g: float = 1.0,
    ) -> dict:
        """
        Find the single model closest to (T_eff_target, log_g_target).

        Parameters
        ----------
        T_eff_target : float
            Target effective temperature.
        log_g_target : float
            Target surface gravity.
        weight_T_eff : float, optional
            Weight for T_eff term (default: 1.0).
        weight_log_g : float, optional
            Weight for log_g term (default: 1.0).

        Returns
        -------
        dict
            Dictionary with model info and distance metric.

        Raises
        ------
        MatchingError
            If no models are loaded.
        """
        if not self.seq.models:
            raise MatchingError("No models loaded in sequence")

        d = self._normalize_teff_logg(
            T_eff_target, log_g_target, weight_T_eff, weight_log_g
        )
        return self._teff_logg_dict(int(np.argmin(d)), d, T_eff_target, log_g_target)

    def find_closest_models_around(
        self,
        T_eff_target: float,
        log_g_target: float,
        n_models: int = 5,
        weight_T_eff: float = 1.0,
        weight_log_g: float = 1.0,
    ) -> list[dict]:
        """
        Return the n models closest to (T_eff_target, log_g_target).

        Parameters
        ----------
        T_eff_target : float
            Target effective temperature.
        log_g_target : float
            Target surface gravity.
        n_models : int, optional
            Number of models to return (default: 5).
        weight_T_eff : float, optional
            Weight for T_eff term (default: 1.0).
        weight_log_g : float, optional
            Weight for log_g term (default: 1.0).

        Returns
        -------
        list[dict]
            List of dictionaries sorted by distance ascending.

        Raises
        ------
        MatchingError
            If no models are loaded.
        """
        if not self.seq.models:
            raise MatchingError("No models loaded in sequence")

        d = self._normalize_teff_logg(
            T_eff_target, log_g_target, weight_T_eff, weight_log_g
        )
        n = min(n_models, len(self.seq.models))

        return [
            self._teff_logg_dict(int(i), d, T_eff_target, log_g_target)
            for i in np.argsort(d)[:n]
        ]
