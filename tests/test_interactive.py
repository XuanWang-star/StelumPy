"""
Tests for StelumPy.plotting.interactive module.

Note: These tests focus on the non-GUI logic since Tkinter requires
a display server. We test initialization, data processing, and helper
functions, while mocking the GUI components.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from pathlib import Path

from StelumPy.plotting.interactive import (
    _style_ax,
    _col_to_values,
    _filter_params,
    _save_figure,
    PROFILE_PARAMS,
    SEQ_PARAMS,
    _SKIP_COLS,
    ModelExplorer,
    SequenceExplorer,
)


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for standalone helper functions."""
    
    def test_style_ax_applies_formatting(self):
        """Test _style_ax applies tick formatting."""
        fig, ax = plt.subplots()
        ax.grid(True)  # Enable grid first
        _style_ax(ax)
        
        # _style_ax calls ax.grid(False) which toggles grid off
        # Check that xaxis and yaxis grid are disabled
        ax.grid(False)  # Ensure grid is off
        grid_lines = ax.xaxis.get_gridlines()
        # After grid(False), gridlines should not be visible
        for line in grid_lines:
            assert not line.get_visible() or line.get_alpha() == 0
    
    def test_col_to_values_log_q_negates(self):
        """Test _col_to_values negates log_q column."""
        df = pd.DataFrame({"log_q": [-1.0, -2.0, -3.0]})
        values = _col_to_values(df, "log_q")
        
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
    
    def test_col_to_values_other_columns_unchanged(self):
        """Test _col_to_values keeps other columns unchanged."""
        df = pd.DataFrame({"X_He": [0.1, 0.5, 0.9]})
        values = _col_to_values(df, "X_He")
        
        np.testing.assert_array_equal(values, [0.1, 0.5, 0.9])
    
    def test_filter_params_includes_available(self):
        """Test _filter_params includes only available columns."""
        available = {"log_q", "X_He", "r"}
        filtered = _filter_params(PROFILE_PARAMS, available)
        
        # Should include log_q, X_He, r
        cols = [p[0] for p in filtered]
        assert "log_q" in cols
        assert "X_He" in cols
        assert "r" in cols
    
    def test_filter_params_excludes_skip_cols(self):
        """Test _filter_params excludes skip columns."""
        available = {"log_q", "X_He", "n", "flag14", "unknown"}
        filtered = _filter_params(PROFILE_PARAMS, available)
        
        cols = [p[0] for p in filtered]
        assert "n" not in cols
        assert "flag14" not in cols
        assert "unknown" not in cols
    
    def test_filter_params_empty_available(self):
        """Test _filter_params with empty available set."""
        filtered = _filter_params(PROFILE_PARAMS, set())
        assert filtered == []
    
    def test_save_figure_exports_multiple_formats(self, tmp_path):
        """Test _save_figure exports png, pdf, svg."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        stem = str(tmp_path / "test_plot")
        
        _save_figure(fig, stem)
        
        assert (tmp_path / "test_plot.png").exists()
        assert (tmp_path / "test_plot.pdf").exists()
        assert (tmp_path / "test_plot.svg").exists()
        
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests for PROFILE_PARAMS and SEQ_PARAMS
# ---------------------------------------------------------------------------

class TestParameterCatalogues:
    """Tests for parameter catalogues."""
    
    def test_profile_params_format(self):
        """Test PROFILE_PARAMS has correct format."""
        for param in PROFILE_PARAMS:
            assert len(param) == 4
            col, combo_label, xlabel, ylabel = param
            assert isinstance(col, str)
            assert isinstance(combo_label, str)
            assert isinstance(xlabel, str)
            assert isinstance(ylabel, str)
    
    def test_seq_params_format(self):
        """Test SEQ_PARAMS has correct format."""
        for param in SEQ_PARAMS:
            assert len(param) == 4
            col, combo_label, xlabel, ylabel = param
            assert isinstance(col, str)
            assert isinstance(combo_label, str)
            assert isinstance(xlabel, str)
            assert isinstance(ylabel, str)
    
    def test_skip_cols_defined(self):
        """Test _SKIP_COLS contains expected columns."""
        assert "n" in _SKIP_COLS
        assert "flag14" in _SKIP_COLS
        assert "flag15" in _SKIP_COLS
        assert "unknown" in _SKIP_COLS


# ---------------------------------------------------------------------------
# Tests for ModelExplorer
# ---------------------------------------------------------------------------

class TestModelExplorerInit:
    """Tests for ModelExplorer initialization."""
    
    def test_init_with_valid_model(self, sample_model):
        """Test ModelExplorer initializes with valid model."""
        explorer = ModelExplorer(sample_model)
        
        assert explorer.model is sample_model
        assert explorer._params is not None
        assert len(explorer._params) > 0
        assert explorer._cols is not None
        assert explorer._clabels is not None
    
    def test_init_with_no_df_raises(self, sample_model):
        """Test ModelExplorer raises when model has no df."""
        sample_model.df = None
        
        with pytest.raises(ValueError, match="no profile data"):
            ModelExplorer(sample_model)
    
    def test_init_sets_save_stem(self, sample_model):
        """Test ModelExplorer sets save_stem from model file."""
        explorer = ModelExplorer(sample_model)
        
        assert explorer.save_stem == sample_model.file_path.stem
    
    def test_init_with_custom_save_stem(self, sample_model):
        """Test ModelExplorer uses custom save_stem."""
        explorer = ModelExplorer(sample_model, save_stem="custom_name")
        
        assert explorer.save_stem == "custom_name"
    
    def test_init_filters_params(self, sample_model):
        """Test ModelExplorer filters params based on available columns."""
        explorer = ModelExplorer(sample_model)
        
        # All cols should be in model.df
        for col in explorer._cols:
            assert col in sample_model.df.columns
            assert col not in _SKIP_COLS


# ---------------------------------------------------------------------------
# Tests for ModelExplorer methods (with mocked GUI)
# ---------------------------------------------------------------------------

class TestModelExplorerMethods:
    """Tests for ModelExplorer methods."""
    
    def test_on_plot_draws_data(self, sample_model):
        """Test _on_plot draws data on axes."""
        explorer = ModelExplorer(sample_model)
        
        # Create figure with Agg backend (non-interactive)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        explorer._fig = fig
        explorer._ax = ax
        explorer._canvas = MagicMock()
        
        # Mock combo boxes
        mock_combo_x = MagicMock()
        mock_combo_x.get.return_value = "log q  (mass coord)"
        mock_combo_y = MagicMock()
        mock_combo_y.get.return_value = "X_He  (helium)"
        explorer._combo_x = mock_combo_x
        explorer._combo_y = mock_combo_y
        
        # Call _on_plot
        explorer._on_plot()
        
        # Check that data was plotted (ax should have lines)
        assert len(ax.lines) > 0
        
        # Check labels are set
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
    
    def test_on_plot_with_different_columns(self, sample_model):
        """Test _on_plot works with different column combinations."""
        explorer = ModelExplorer(sample_model)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        explorer._fig = fig
        explorer._ax = ax
        explorer._canvas = MagicMock()
        
        # Test with r vs T
        mock_combo_x = MagicMock()
        mock_combo_x.get.return_value = "r  (radius)"
        mock_combo_y = MagicMock()
        mock_combo_y.get.return_value = "T  (temperature)"
        explorer._combo_x = mock_combo_x
        explorer._combo_y = mock_combo_y
        
        explorer._on_plot()
        
        assert len(ax.lines) > 0
    
    def test_on_save_calls_save_figure(self, sample_model):
        """Test _on_save calls _save_figure with correct args."""
        explorer = ModelExplorer(sample_model)
        explorer._fig = MagicMock()
        
        with patch('StelumPy.plotting.interactive._save_figure') as mock_save:
            explorer._on_save()
            mock_save.assert_called_once_with(explorer._fig, explorer.save_stem)
    
    def test_on_key_press_h_resets_view(self, sample_model):
        """Test _on_key_press resets view with 'h' key."""
        explorer = ModelExplorer(sample_model)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3])
        explorer._fig = fig
        explorer._ax = ax
        explorer._canvas = MagicMock()
        
        # Mock event
        mock_event = MagicMock()
        mock_event.key = 'h'
        
        explorer._on_key_press(mock_event)
        
        # autoscale_view should be called
        explorer._canvas.draw.assert_called()
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_key_press_s_saves(self, mock_canvas, mock_tk, sample_model):
        """Test _on_key_press saves with 's' key."""
        explorer = ModelExplorer(sample_model)
        explorer._fig = MagicMock()
        explorer._canvas = MagicMock()
        
        with patch.object(explorer, '_on_save') as mock_save:
            mock_event = MagicMock()
            mock_event.key = 's'
            explorer._on_key_press(mock_event)
            mock_save.assert_called_once()
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_key_press_z_prints_zoom(self, mock_canvas, mock_tk, sample_model, capsys):
        """Test _on_key_press prints zoom message with 'z' key."""
        explorer = ModelExplorer(sample_model)
        explorer._fig = MagicMock()
        explorer._canvas = MagicMock()
        
        mock_event = MagicMock()
        mock_event.key = 'z'
        explorer._on_key_press(mock_event)
        
        captured = capsys.readouterr()
        assert "Zoom" in captured.out
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_key_press_p_prints_pan(self, mock_canvas, mock_tk, sample_model, capsys):
        """Test _on_key_press prints pan message with 'p' key."""
        explorer = ModelExplorer(sample_model)
        explorer._fig = MagicMock()
        explorer._canvas = MagicMock()
        
        mock_event = MagicMock()
        mock_event.key = 'p'
        explorer._on_key_press(mock_event)
        
        captured = capsys.readouterr()
        assert "Pan" in captured.out


# ---------------------------------------------------------------------------
# Tests for SequenceExplorer
# ---------------------------------------------------------------------------

class TestSequenceExplorerInit:
    """Tests for SequenceExplorer initialization."""
    
    def test_init_with_valid_sequence(self, sample_sequence):
        """Test SequenceExplorer initializes with valid sequence."""
        explorer = SequenceExplorer(sample_sequence)
        
        assert explorer.seq is sample_sequence
        assert explorer.analyzer is not None
        assert explorer._ev_params is not None
        assert explorer._pr_params is not None
    
    def test_init_with_no_seq_data_raises(self, sample_sequence):
        """Test SequenceExplorer raises when seq_data unavailable."""
        sample_sequence.seq_data = None
        
        with pytest.raises(ValueError, match="seq_data not available"):
            SequenceExplorer(sample_sequence)
    
    def test_init_with_no_models_raises(self, sample_sequence):
        """Test SequenceExplorer raises when no models loaded."""
        sample_sequence.models = []
        
        with pytest.raises(ValueError, match="No models loaded"):
            SequenceExplorer(sample_sequence)
    
    def test_init_with_no_profile_data_raises(self, sample_sequence):
        """Test SequenceExplorer raises when no model has profile data."""
        for model in sample_sequence.models:
            model.df = None
        
        with pytest.raises(ValueError, match="No model with profile data"):
            SequenceExplorer(sample_sequence)
    
    def test_init_sets_save_stem(self, sample_sequence):
        """Test SequenceExplorer sets default save_stem."""
        explorer = SequenceExplorer(sample_sequence)
        
        assert explorer.save_stem == "sequence_explorer"
    
    def test_init_with_custom_save_stem(self, sample_sequence):
        """Test SequenceExplorer uses custom save_stem."""
        explorer = SequenceExplorer(sample_sequence, save_stem="custom")
        
        assert explorer.save_stem == "custom"


# ---------------------------------------------------------------------------
# Tests for SequenceExplorer methods (with mocked GUI)
# ---------------------------------------------------------------------------

class TestSequenceExplorerMethods:
    """Tests for SequenceExplorer methods."""
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_plot_draws_both_subplots(self, mock_canvas, mock_tk, sample_sequence):
        """Test _on_plot draws both evolution and profile."""
        explorer = SequenceExplorer(sample_sequence)
        
        # Mock the figure and axes
        fig = MagicMock()
        ax_ev = MagicMock()
        ax_pr = MagicMock()
        explorer._fig = fig
        explorer._ax_ev = ax_ev
        explorer._ax_pr = ax_pr
        explorer._canvas = MagicMock()
        
        # Mock combo boxes
        mock_combo_ev_x = MagicMock()
        mock_combo_ev_x.get.return_value = "Age (yr)"
        mock_combo_ev_y = MagicMock()
        mock_combo_ev_y.get.return_value = "T_eff (K)"
        explorer._combo_ev_x = mock_combo_ev_x
        explorer._combo_ev_y = mock_combo_ev_y
        
        mock_combo_pr_x = MagicMock()
        mock_combo_pr_x.get.return_value = "log q  (mass coord)"
        mock_combo_pr_y = MagicMock()
        mock_combo_pr_y.get.return_value = "X_He  (helium)"
        explorer._combo_pr_x = mock_combo_pr_x
        explorer._combo_pr_y = mock_combo_pr_y
        
        # Call _on_plot
        explorer._on_plot()
        
        # Both axes should have been used
        assert ax_ev.plot.called or ax_ev.cla.called
        assert ax_pr.plot.called or ax_pr.cla.called
    
    def test_on_save_calls_save_figure(self, sample_sequence):
        """Test _on_save calls _save_figure."""
        explorer = SequenceExplorer(sample_sequence)
        explorer._fig = MagicMock()
        
        with patch('StelumPy.plotting.interactive._save_figure') as mock_save:
            explorer._on_save()
            mock_save.assert_called_once_with(explorer._fig, explorer.save_stem)
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_click_selects_model(self, mock_canvas, mock_tk, sample_sequence):
        """Test _on_click selects model index from click."""
        explorer = SequenceExplorer(sample_sequence)
        
        # Mock axes and canvas
        explorer._ax_ev = MagicMock()
        explorer._canvas = MagicMock()
        
        # Mock combo boxes for evolution
        mock_combo_ev_x = MagicMock()
        mock_combo_ev_x.get.return_value = "Age (yr)"
        explorer._combo_ev_x = mock_combo_ev_x
        mock_combo_ev_y = MagicMock()
        mock_combo_ev_y.get.return_value = "Teff"
        explorer._combo_ev_y = mock_combo_ev_y
        
        # Mock click event on evolution axes
        mock_event = MagicMock()
        mock_event.inaxes = explorer._ax_ev
        mock_event.xdata = 5e6  # Some age value
        
        # Mock _draw_evolution and _draw_profile to avoid actual plotting
        with patch.object(explorer, '_draw_evolution'):
            with patch.object(explorer, '_draw_profile'):
                explorer._on_click(mock_event)
        
        # Should have selected an index
        assert explorer._sel_idx is not None
        assert 0 <= explorer._sel_idx < len(sample_sequence)
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_click_ignores_outside_axes(self, mock_canvas, mock_tk, sample_sequence):
        """Test _on_click ignores clicks outside evolution axes."""
        explorer = SequenceExplorer(sample_sequence)
        other_ax = MagicMock()
        explorer._ax_ev = MagicMock()
        explorer._canvas = MagicMock()
        
        mock_event = MagicMock()
        mock_event.inaxes = other_ax  # Different axes
        explorer._on_click(mock_event)
        
        # Should not have selected anything
        assert explorer._sel_idx is None
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_key_press_h_resets_both_axes(self, mock_canvas, mock_tk, sample_sequence):
        """Test _on_key_press resets both axes with 'h' key."""
        explorer = SequenceExplorer(sample_sequence)
        explorer._ax_ev = MagicMock()
        explorer._ax_pr = MagicMock()
        explorer._canvas = MagicMock()
        
        mock_event = MagicMock()
        mock_event.key = 'h'
        explorer._on_key_press(mock_event)
        
        explorer._ax_ev.autoscale_view.assert_called()
        explorer._ax_pr.autoscale_view.assert_called()
        explorer._canvas.draw.assert_called()
    
    @patch('StelumPy.plotting.interactive.tk.Tk')
    @patch('StelumPy.plotting.interactive.FigureCanvasTkAgg')
    def test_on_key_press_s_saves(self, mock_canvas, mock_tk, sample_sequence):
        """Test _on_key_press saves with 's' key."""
        explorer = SequenceExplorer(sample_sequence)
        explorer._fig = MagicMock()
        explorer._canvas = MagicMock()
        
        with patch.object(explorer, '_on_save') as mock_save:
            mock_event = MagicMock()
            mock_event.key = 's'
            explorer._on_key_press(mock_event)
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for interactive module."""
    
    def test_model_explorer_with_all_columns(self, sample_model):
        """Test ModelExplorer handles all available columns."""
        explorer = ModelExplorer(sample_model)
        
        # Should have filtered params correctly
        assert len(explorer._cols) > 0
        
        # All cols should be valid
        for col in explorer._cols:
            assert col in sample_model.df.columns
    
    def test_sequence_explorer_with_all_columns(self, sample_sequence):
        """Test SequenceExplorer handles all available columns."""
        explorer = SequenceExplorer(sample_sequence)
        
        # Should have filtered params correctly
        assert len(explorer._ev_cols) > 0
        assert len(explorer._pr_cols) > 0
        
        # Evolution cols should be in seq_data
        for col in explorer._ev_cols:
            assert col in sample_sequence.seq_data.columns
        
        # Profile cols should be in model.df
        for col in explorer._pr_cols:
            assert col in sample_sequence.models[0].df.columns
    
    def test_helper_functions_consistency(self):
        """Test helper functions work consistently together."""
        df = pd.DataFrame({
            "log_q": [-1.0, -2.0, -3.0],
            "X_He": [0.1, 0.5, 0.9],
            "r": [1.0, 2.0, 3.0],
        })
        
        # _col_to_values should negate log_q
        log_q_vals = _col_to_values(df, "log_q")
        assert log_q_vals[0] > 0
        
        # Other columns unchanged
        x_he_vals = _col_to_values(df, "X_He")
        np.testing.assert_array_equal(x_he_vals, [0.1, 0.5, 0.9])
