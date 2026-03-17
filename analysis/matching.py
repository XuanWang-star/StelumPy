"""
stellarmod.analysis.matching
-----------------------------
SequenceAnalyzer — all matching, statistics, and data-extraction methods
for a Sequence of stellar models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ..io.model import Model
from ..io.sequence import Sequence


class SequenceAnalyzer:
    """
    Perform matching and data-extraction operations on a :class:`Sequence`.

    Parameters
    ----------
    sequence : Sequence
        The loaded model sequence to analyse.
    """

    def __init__(self, sequence: Sequence):
        self.seq = sequence

    # ------------------------------------------------------------------
    # Evolution data extraction
    # ------------------------------------------------------------------

    def get_evolution_data(self, parameter: str) -> np.ndarray:
        """
        Return an array of *parameter* values, one per model.

        For header parameters ('T_eff', 'log_g', 'mesh_number', 'data_type')
        the attribute is read directly.  For profile columns the central
        value (row 0) is used.
        """
        header_params = {'T_eff', 'log_g', 'mesh_number', 'data_type'}
        values = []
        for model in self.seq.models:
            if parameter in header_params:
                values.append(getattr(model, parameter))
            elif model.df is not None and parameter in model.df.columns:
                values.append(model.df[parameter].iloc[0])
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
        mesh_point  : 'center' | 'surface' | int (0-based index)
        """
        values = []
        for model in self.seq.models:
            if model.df is not None and column_name in model.df.columns:
                if mesh_point == 'center':
                    values.append(model.df[column_name].iloc[0])
                elif mesh_point == 'surface':
                    values.append(model.df[column_name].iloc[-1])
                elif isinstance(mesh_point, int) and mesh_point < len(model.df):
                    values.append(model.df[column_name].iloc[mesh_point])
                else:
                    values.append(np.nan)
            else:
                values.append(np.nan)
        return np.array(values)

    def get_he_core_evolution(self, n_points: int = 5) -> np.ndarray:
        """Mean X_He over the *n* innermost mesh points for every model."""
        return np.array([m.he_core_he(n_points) for m in self.seq.models])

    def create_evolution_dataframe(self, parameters: list[str]) -> pd.DataFrame:
        """
        Build a DataFrame with the Age column plus one column per parameter.
        """
        data: dict[str, np.ndarray] = {'Age': self.seq.age_sequence}
        for p in parameters:
            data[p] = self.get_evolution_data(p)
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Matching by core He abundance
    # ------------------------------------------------------------------

    def find_model_by_he_core(
        self,
        target_he: float,
        n_points: int = 10,
    ) -> dict:
        """
        Find the single model whose core He value is closest to *target_he*.
        """
        if not self.seq.models:
            raise ValueError("No models loaded in sequence.")
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
        Return the *n* models whose core He value is closest to *target_he*,
        sorted by |delta| ascending.
        """
        if not self.seq.models:
            raise ValueError("No models loaded in sequence.")
        he = self.get_he_core_evolution(n_points)
        n = min(n_models, len(self.seq.models))
        results = []
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

    # ------------------------------------------------------------------
    # He profile matching (core → surface)
    # ------------------------------------------------------------------

    def he_profile_match(
        self,
        target_model: Model,
        snapshot_model: Model,
        n_points: int = 200,
        plot: bool = False,
        ax=None,
    ) -> dict:
        """
        Quantify the X_He profile match between a static *target_model* and
        an evolution *snapshot_model* from this sequence.

        Both profiles are interpolated onto a common log_q grid so that
        different mesh resolutions are handled correctly.

        Returns
        -------
        dict with keys:
            rmse     – root-mean-square error of ΔX_He
            mae      – mean absolute error
            r2       – coefficient of determination  (1 = perfect match)
            max_dev  – maximum absolute deviation at any grid point
        """
        for label, m in [('target_model', target_model),
                          ('snapshot_model', snapshot_model)]:
            if m.df is None:
                raise ValueError(f"{label} has no profile data loaded.")
            for col in ('X_He', 'log_q'):
                if col not in m.df.columns:
                    raise KeyError(f"{label} is missing column '{col}'.")

        def _prepare(model: Model):
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


    
    
    
    
    
    
    # ------------------------------------------------------------------
    # Matching by T_eff / log_g
    # ------------------------------------------------------------------

    def _normalize_teff_logg(
        self,
        T_eff_target: float,
        log_g_target: float,
        weight_T_eff: float = 1.0,
        weight_log_g: float = 1.0,
    ) -> np.ndarray:
        """Return normalised weighted Euclidean distances to (T_eff_target, log_g_target)."""
        T_arr = np.array([m.T_eff for m in self.seq.models])
        g_arr = np.array([m.log_g for m in self.seq.models])
        T_n = (T_arr - T_arr.mean()) / T_arr.std()
        g_n = (g_arr - g_arr.mean()) / g_arr.std()
        T_t = (T_eff_target - T_arr.mean()) / T_arr.std()
        g_t = (log_g_target - g_arr.mean()) / g_arr.std()
        return np.sqrt(weight_T_eff * (T_n - T_t) ** 2 + weight_log_g * (g_n - g_t) ** 2)

    def _teff_logg_dict(self, idx: int, distances: np.ndarray,
                        T_eff_target: float, log_g_target: float) -> dict:
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
        Find the single model in the sequence closest to *(T_eff_target, log_g_target)*
        in normalised parameter space.
        """
        if not self.seq.models:
            raise ValueError("No models loaded in sequence.")
        d = self._normalize_teff_logg(T_eff_target, log_g_target, weight_T_eff, weight_log_g)
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
        Return the *n* models closest to *(T_eff_target, log_g_target)*,
        sorted by distance ascending.
        """
        if not self.seq.models:
            raise ValueError("No models loaded in sequence.")
        d = self._normalize_teff_logg(T_eff_target, log_g_target, weight_T_eff, weight_log_g)
        n = min(n_models, len(self.seq.models))
        return [self._teff_logg_dict(int(i), d, T_eff_target, log_g_target)
                for i in np.argsort(d)[:n]]

