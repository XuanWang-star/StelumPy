"""
stellarmod.plotting.plots
--------------------------
SequencePlotter — all visualisation for a Sequence of stellar models.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..io.sequence import Sequence
from ..analysis.matching import SequenceAnalyzer


class SequencePlotter:
    """
    Generate plots for a :class:`Sequence` of stellar models.

    Parameters
    ----------
    sequence : Sequence
        The loaded model sequence to visualise.
    """

    def __init__(self, sequence: Sequence):
        self.seq = sequence
        self._analyzer = SequenceAnalyzer(sequence)

    # ------------------------------------------------------------------
    # Evolution plots
    # ------------------------------------------------------------------

    def plot_evolution(
        self,
        parameter: str,
        xlabel: str = 'Age',
        ylabel: str | None = None,
        use_log_time: bool = False,
        ax=None,
        **kwargs,
    ):
        """
        Plot the evolution of *parameter* over the model sequence.

        Returns
        -------
        (fig, ax)
        """
        values = self._analyzer.get_evolution_data(parameter)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()

        ax.plot(self.seq.age_sequence, values, marker='o', **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or parameter)
        ax.set_title(f'Evolution of {parameter}')
        ax.grid(True, alpha=0.3)
        if use_log_time:
            ax.set_xscale('log')

        if created:
            fig.tight_layout()
        return fig, ax

    def plot_hr_diagram(self, ax=None, **kwargs):
        """
        Hertzsprung-Russell diagram: log(L) vs T_eff (hot stars on the left).

        Returns
        -------
        (fig, ax)
        """
        if self.seq.seq_data is None:
            raise ValueError("seq_data not available; cannot build HR diagram.")

        T_eff = self.seq.seq_data['Teff'].values
        lum   = self.seq.seq_data['Lum'].values

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()

        ax.plot(T_eff, np.log10(lum), marker='o', **kwargs)
        ax.invert_xaxis()
        ax.set_xlabel('T_eff (K)')
        ax.set_ylabel('log(L/L☉)')
        ax.set_title('Hertzsprung-Russell Diagram')
        ax.grid(True, alpha=0.3)

        if created:
            fig.tight_layout()
        return fig, ax

    def compare_profiles(
        self,
        column_name: str,
        model_indices: list[int],
        x_column: str = 'r',
        ax=None,
        **kwargs,
    ):
        """
        Overlay a profile column for several models.

        Parameters
        ----------
        column_name  : y-axis column (e.g. 'X_He')
        model_indices: list of model indices to include
        x_column     : x-axis column (default 'r')

        Returns
        -------
        (fig, ax)
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.get_figure()

        for idx in model_indices:
            if idx >= len(self.seq.models):
                continue
            model = self.seq.models[idx]
            if model.df is None or column_name not in model.df.columns:
                continue
            age = (float(self.seq.age_sequence[idx])
                   if self.seq.age_sequence is not None else idx)
            label = f"t={age:.2e}  T_eff={model.T_eff:.0f} K"
            ax.plot(model.df[x_column], model.df[column_name], label=label, **kwargs)

        ax.set_xlabel(x_column)
        ax.set_ylabel(column_name)
        ax.set_title(f'Profile comparison: {column_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created:
            fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # He profile match plot
    # ------------------------------------------------------------------

    def plot_he_profile_match(
        self,
        target_model,
        snapshot_model,
        n_points: int = 200,
        ax=None,
    ):
        """
        Plot the X_He profile comparison between *target_model* and
        *snapshot_model*, and return the match metrics.

        Returns
        -------
        (fig, ax), metrics_dict
        """
        metrics = self._analyzer.he_profile_match(
            target_model, snapshot_model, n_points=n_points
        )

        from ..io.model import Model
        from ..analysis.matching import SequenceAnalyzer

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

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.get_figure()

        self._draw_he_profile_match(
            ax, q_grid, y1, y2, metrics['rmse'],
            Path(target_model.file_path).name,
            Path(snapshot_model.file_path).name,
        )

        if created:
            fig.tight_layout()
        return (fig, ax), metrics

    # ------------------------------------------------------------------
    # Internal drawing helper (also called by SequenceAnalyzer)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_he_profile_match(
        ax,
        q_grid: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        rmse: float,
        tgt_name: str,
        snap_name: str,
    ) -> None:
        ax.plot(q_grid, y1, label=f'Target ({tgt_name})', color='steelblue', lw=2)
        ax.plot(q_grid, y2, label=f'Snapshot ({snap_name})', color='tomato', lw=2, ls='--')
        ax.fill_between(q_grid, y1, y2, alpha=0.15, color='gray',
                        label=f'|ΔX_He|  RMSE={rmse:.4f}')
        ax.set_xlabel('log q  (centre → surface)')
        ax.set_ylabel('X_He')
        ax.set_title('He abundance profile match')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
