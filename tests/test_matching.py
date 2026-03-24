"""
Tests for StelumPy.analysis.matching module.
"""

import pytest
import numpy as np

from StelumPy.analysis.matching import SequenceAnalyzer
from StelumPy import Model, Sequence


class TestSequenceAnalyzerInit:
    """Tests for SequenceAnalyzer initialization."""

    def test_init_with_valid_sequence(self, sample_sequence):
        """Test SequenceAnalyzer initializes correctly."""
        analyzer = SequenceAnalyzer(sample_sequence)
        assert analyzer.seq is sample_sequence


class TestEvolutionData:
    """Tests for evolution data extraction methods."""

    def test_get_evolution_data_teff(self, sample_sequence):
        """Test get_evolution_data for T_eff."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_evolution_data('T_eff')
        assert isinstance(values, np.ndarray)
        assert len(values) == 3

    def test_get_evolution_data_log_g(self, sample_sequence):
        """Test get_evolution_data for log_g."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_evolution_data('log_g')
        assert isinstance(values, np.ndarray)
        assert len(values) == 3

    def test_get_evolution_data_profile_column(self, sample_sequence):
        """Test get_evolution_data for profile column (X_He)."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_evolution_data('X_He')
        assert isinstance(values, np.ndarray)
        assert len(values) == 3
        # Central values should be in [0, 1]
        assert np.all((values >= 0) & (values <= 1))

    def test_get_evolution_data_invalid_column(self, sample_sequence):
        """Test get_evolution_data returns NaN for invalid column."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_evolution_data('nonexistent')
        assert isinstance(values, np.ndarray)
        assert np.all(np.isnan(values))

    def test_get_profile_evolution_center(self, sample_sequence):
        """Test get_profile_evolution with center mesh point."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_profile_evolution('X_He', mesh_point='center')
        assert len(values) == 3

    def test_get_profile_evolution_surface(self, sample_sequence):
        """Test get_profile_evolution with surface mesh point."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_profile_evolution('X_He', mesh_point='surface')
        assert len(values) == 3

    def test_get_profile_evolution_index(self, sample_sequence):
        """Test get_profile_evolution with integer mesh point."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_profile_evolution('X_He', mesh_point=0)
        assert len(values) == 3

    def test_get_he_core_evolution(self, sample_sequence):
        """Test get_he_core_evolution."""
        analyzer = SequenceAnalyzer(sample_sequence)
        values = analyzer.get_he_core_evolution(n_points=5)
        assert isinstance(values, np.ndarray)
        assert len(values) == 3
        assert np.all((values >= 0) & (values <= 1))

    def test_create_evolution_dataframe(self, sample_sequence):
        """Test create_evolution_dataframe."""
        analyzer = SequenceAnalyzer(sample_sequence)
        df = analyzer.create_evolution_dataframe(['T_eff', 'log_g'])
        assert 'Age' in df.columns
        assert 'T_eff' in df.columns
        assert 'log_g' in df.columns
        assert len(df) == 3


class TestHeCoreMatching:
    """Tests for core He abundance matching methods."""

    def test_find_model_by_he_core(self, sample_sequence):
        """Test find_model_by_he_core returns correct structure."""
        analyzer = SequenceAnalyzer(sample_sequence)
        result = analyzer.find_model_by_he_core(target_he=0.5, n_points=5)
        
        assert 'index' in result
        assert 'model' in result
        assert 'he_core' in result
        assert 'delta' in result
        assert 'age' in result
        assert 'T_eff' in result
        assert 'log_g' in result
        
        assert isinstance(result['index'], int)
        assert isinstance(result['model'], Model)
        assert isinstance(result['he_core'], float)
        assert isinstance(result['delta'], float)

    def test_find_model_by_he_core_exact_match(self, sample_sequence):
        """Test find_model_by_he_core with exact match."""
        analyzer = SequenceAnalyzer(sample_sequence)
        # Get actual he_core value of first model
        first_model_he = sample_sequence.models[0].he_core_he(n_points=5)
        result = analyzer.find_model_by_he_core(target_he=first_model_he, n_points=5)
        
        assert result['index'] == 0
        assert abs(result['delta']) < 1e-10

    def test_find_models_by_he_core(self, sample_sequence):
        """Test find_models_by_he_core returns multiple results."""
        analyzer = SequenceAnalyzer(sample_sequence)
        results = analyzer.find_models_by_he_core(target_he=0.5, n_models=2, n_points=5)
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert 'index' in result
            assert 'model' in result
            assert 'he_core' in result
            assert 'delta' in result

    def test_find_models_by_he_core_sorted(self, sample_sequence):
        """Test find_models_by_he_core returns sorted results."""
        analyzer = SequenceAnalyzer(sample_sequence)
        results = analyzer.find_models_by_he_core(target_he=0.5, n_models=3, n_points=5)
        
        # Results should be sorted by |delta|
        deltas = [abs(r['delta']) for r in results]
        assert deltas == sorted(deltas)

    def test_find_model_by_he_core_no_models(self, temp_dir):
        """Test find_model_by_he_core raises with no models."""
        from .helpers import create_test_seq_file
        seq_dir = temp_dir / "empty_seq"
        seq_dir.mkdir()
        create_test_seq_file(seq_dir, num_models=0)
        
        # Create empty 5mext directory
        models_dir = seq_dir / "5mext"
        models_dir.mkdir()
        
        # Sequence will fail to load models, so we test the ValueError path
        with pytest.raises((ValueError, FileNotFoundError)):
            seq = Sequence(seq_dir, verbose=False)
            if seq.models:  # Only test if models were loaded
                analyzer = SequenceAnalyzer(seq)
                analyzer.find_model_by_he_core(target_he=0.5)


class TestHeProfileMatch:
    """Tests for He profile matching method."""

    def test_he_profile_match(self, sample_sequence):
        """Test he_profile_match returns correct metrics."""
        analyzer = SequenceAnalyzer(sample_sequence)
        target = sample_sequence.models[0]
        snapshot = sample_sequence.models[1]
        
        metrics = analyzer.he_profile_match(target, snapshot, n_points=50)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'max_dev' in metrics
        
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['r2'], float)
        assert isinstance(metrics['max_dev'], float)
        
        # RMSE and MAE should be non-negative
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['max_dev'] >= 0

    def test_he_profile_match_perfect(self, sample_sequence):
        """Test he_profile_match with identical profiles."""
        analyzer = SequenceAnalyzer(sample_sequence)
        target = sample_sequence.models[0]
        
        metrics = analyzer.he_profile_match(target, target, n_points=50)
        
        # Perfect match should have near-zero errors
        assert metrics['rmse'] < 1e-10
        assert metrics['mae'] < 1e-10
        assert metrics['max_dev'] < 1e-10

    def test_he_profile_match_no_data(self, sample_sequence):
        """Test he_profile_match raises when model has no data."""
        analyzer = SequenceAnalyzer(sample_sequence)
        target = sample_sequence.models[0]
        target.df = None
        
        with pytest.raises(ValueError, match="no profile data"):
            analyzer.he_profile_match(target, sample_sequence.models[1])

    def test_he_profile_match_missing_column(self, sample_sequence):
        """Test he_profile_match raises when column missing."""
        analyzer = SequenceAnalyzer(sample_sequence)
        target = sample_sequence.models[0]
        # Remove X_He column
        target.df = target.df.drop(columns=['X_He'])
        
        with pytest.raises(KeyError, match="X_He"):
            analyzer.he_profile_match(target, sample_sequence.models[1])


class TestTeffLogGMatching:
    """Tests for T_eff/log_g matching methods."""

    def test_find_closest_model(self, sample_sequence):
        """Test find_closest_model returns correct structure."""
        analyzer = SequenceAnalyzer(sample_sequence)
        result = analyzer.find_closest_model(T_eff_target=5100, log_g_target=4.4)
        
        assert 'index' in result
        assert 'model' in result
        assert 'T_eff' in result
        assert 'log_g' in result
        assert 'age' in result
        assert 'distance' in result
        assert 'dT_eff' in result
        assert 'dlog_g' in result

    def test_find_closest_models_around(self, sample_sequence):
        """Test find_closest_models_around returns multiple results."""
        analyzer = SequenceAnalyzer(sample_sequence)
        results = analyzer.find_closest_models_around(
            T_eff_target=5100, log_g_target=4.4, n_models=2
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert 'index' in result
            assert 'model' in result
            assert 'distance' in result

    def test_find_closest_models_sorted(self, sample_sequence):
        """Test find_closest_models_around returns sorted results."""
        analyzer = SequenceAnalyzer(sample_sequence)
        results = analyzer.find_closest_models_around(
            T_eff_target=5100, log_g_target=4.4, n_models=3
        )
        
        # Results should be sorted by distance
        distances = [r['distance'] for r in results]
        assert distances == sorted(distances)

    def test_find_closest_model_weights(self, sample_sequence):
        """Test find_closest_model with custom weights."""
        analyzer = SequenceAnalyzer(sample_sequence)
        result1 = analyzer.find_closest_model(
            T_eff_target=5100, log_g_target=4.4,
            weight_T_eff=1.0, weight_log_g=1.0
        )
        result2 = analyzer.find_closest_model(
            T_eff_target=5100, log_g_target=4.4,
            weight_T_eff=10.0, weight_log_g=1.0
        )
        
        # Both should return valid results
        assert 'model' in result1
        assert 'model' in result2
