"""
Tests for StelumPy.plotting.plots module.
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from StelumPy.plotting.plots import SequencePlotter


class TestSequencePlotterInit:
    """Tests for SequencePlotter initialization."""

    def test_init_with_valid_sequence(self, sample_sequence):
        """Test SequencePlotter initializes correctly."""
        plotter = SequencePlotter(sample_sequence)
        assert plotter.seq is sample_sequence
        assert plotter._analyzer is not None


class TestPlotEvolution:
    """Tests for plot_evolution method."""

    def test_plot_evolution_basic(self, sample_sequence):
        """Test plot_evolution creates figure and axes."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.plot_evolution('T_eff')
        
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)

    def test_plot_evolution_with_ax(self, sample_sequence):
        """Test plot_evolution with provided axes."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plt.subplots()
        result_fig, result_ax = plotter.plot_evolution('T_eff', ax=ax)
        
        assert result_fig is fig
        assert result_ax is ax
        
        plt.close(fig)

    def test_plot_evolution_labels(self, sample_sequence):
        """Test plot_evolution sets correct labels."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.plot_evolution('T_eff', xlabel='Time', ylabel='Temperature')
        
        assert ax.get_xlabel() == 'Time'
        assert ax.get_ylabel() == 'Temperature'
        
        plt.close(fig)

    def test_plot_evolution_log_time(self, sample_sequence):
        """Test plot_evolution with log time scale."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.plot_evolution('T_eff', use_log_time=True)
        
        assert ax.get_xscale() == 'log'
        
        plt.close(fig)


class TestPlotHRDiagram:
    """Tests for plot_hr_diagram method."""

    def test_plot_hr_diagram_basic(self, sample_sequence):
        """Test plot_hr_diagram creates figure and axes."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.plot_hr_diagram()
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_plot_hr_diagram_inverted_xaxis(self, sample_sequence):
        """Test plot_hr_diagram inverts x-axis (hot stars on left)."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.plot_hr_diagram()
        
        # X-axis should be inverted
        xlim = ax.get_xlim()
        assert xlim[0] > xlim[1]
        
        plt.close(fig)

    def test_plot_hr_diagram_labels(self, sample_sequence):
        """Test plot_hr_diagram sets correct labels."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.plot_hr_diagram()
        
        assert 'T_eff' in ax.get_xlabel()
        assert 'log' in ax.get_ylabel().lower()
        
        plt.close(fig)

    def test_plot_hr_diagram_no_seq_data(self, temp_dir):
        """Test plot_hr_diagram raises when seq_data unavailable."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=2)
        
        # Remove seq.txt content to simulate missing data
        seq = SequencePlotter.__module__.replace('plotting.plots', 'io.sequence')
        from StelumPy.io.sequence import Sequence
        sequence = Sequence(seq_dir, verbose=False)
        sequence.seq_data = None
        
        plotter = SequencePlotter(sequence)
        
        with pytest.raises(ValueError, match="seq_data not available"):
            plotter.plot_hr_diagram()


class TestCompareProfiles:
    """Tests for compare_profiles method."""

    def test_compare_profiles_basic(self, sample_sequence):
        """Test compare_profiles creates figure and axes."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.compare_profiles('X_He', model_indices=[0, 1])
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_compare_profiles_with_ax(self, sample_sequence):
        """Test compare_profiles with provided axes."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plt.subplots()
        result_fig, result_ax = plotter.compare_profiles(
            'X_He', model_indices=[0, 1], ax=ax
        )
        
        assert result_fig is fig
        assert result_ax is ax
        
        plt.close(fig)

    def test_compare_profiles_labels(self, sample_sequence):
        """Test compare_profiles sets correct labels."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.compare_profiles('X_He', model_indices=[0, 1], x_column='r')
        
        assert ax.get_xlabel() == 'r'
        assert ax.get_ylabel() == 'X_He'
        
        plt.close(fig)

    def test_compare_profiles_legend(self, sample_sequence):
        """Test compare_profiles creates legend."""
        plotter = SequencePlotter(sample_sequence)
        fig, ax = plotter.compare_profiles('X_He', model_indices=[0, 1])
        
        legend = ax.get_legend()
        assert legend is not None
        
        plt.close(fig)

    def test_compare_profiles_invalid_index(self, sample_sequence):
        """Test compare_profiles handles invalid model index."""
        plotter = SequencePlotter(sample_sequence)
        # Should not raise, just skip invalid indices
        fig, ax = plotter.compare_profiles('X_He', model_indices=[0, 100])
        
        assert fig is not None
        plt.close(fig)


class TestPlotHeProfileMatch:
    """Tests for plot_he_profile_match method."""

    def test_plot_he_profile_match_basic(self, sample_sequence):
        """Test plot_he_profile_match creates figure and returns metrics."""
        plotter = SequencePlotter(sample_sequence)
        target = sample_sequence.models[0]
        snapshot = sample_sequence.models[1]
        
        (fig, ax), metrics = plotter.plot_he_profile_match(target, snapshot)
        
        assert fig is not None
        assert ax is not None
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'max_dev' in metrics
        
        plt.close(fig)

    def test_plot_he_profile_match_with_ax(self, sample_sequence):
        """Test plot_he_profile_match with provided axes."""
        plotter = SequencePlotter(sample_sequence)
        target = sample_sequence.models[0]
        snapshot = sample_sequence.models[1]
        
        fig, ax = plt.subplots()
        (result_fig, result_ax), metrics = plotter.plot_he_profile_match(
            target, snapshot, ax=ax
        )
        
        assert result_fig is fig
        assert result_ax is ax
        
        plt.close(fig)

    def test_plot_he_profile_match_title(self, sample_sequence):
        """Test plot_he_profile_match sets correct title."""
        plotter = SequencePlotter(sample_sequence)
        target = sample_sequence.models[0]
        snapshot = sample_sequence.models[1]

        (fig, ax), metrics = plotter.plot_he_profile_match(target, snapshot)

        assert 'He' in ax.get_title() or 'profile' in ax.get_title().lower()

        plt.close(fig)

    def test_plot_he_profile_match_legend(self, sample_sequence):
        """Test plot_he_profile_match creates legend."""
        plotter = SequencePlotter(sample_sequence)
        target = sample_sequence.models[0]
        snapshot = sample_sequence.models[1]

        (fig, ax), metrics = plotter.plot_he_profile_match(target, snapshot)

        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)


class TestDrawHeProfileMatch:
    """Tests for static _draw_he_profile_match method."""

    def test_draw_he_profile_match_static(self, sample_data):
        """Test _draw_he_profile_match as static method."""
        fig, ax = plt.subplots()
        
        SequencePlotter._draw_he_profile_match(
            ax=ax,
            q_grid=sample_data['log_q'],
            y1=sample_data['X_He'],
            y2=sample_data['X_He'] * 0.95,
            rmse=0.01,
            tgt_name='target.txt',
            snap_name='snapshot.txt',
        )
        
        # Should have 2 lines (target + snapshot) and 1 fill_between
        assert len(ax.lines) >= 2
        assert len(ax.collections) >= 1  # fill_between creates collection
        
        plt.close(fig)

    def test_draw_he_profile_match_labels(self, sample_data):
        """Test _draw_he_profile_match sets correct labels."""
        fig, ax = plt.subplots()
        
        SequencePlotter._draw_he_profile_match(
            ax=ax,
            q_grid=sample_data['log_q'],
            y1=sample_data['X_He'],
            y2=sample_data['X_He'] * 0.95,
            rmse=0.01,
            tgt_name='target.txt',
            snap_name='snapshot.txt',
        )
        
        assert 'log q' in ax.get_xlabel().lower()
        assert 'X_He' in ax.get_ylabel()
        
        plt.close(fig)
