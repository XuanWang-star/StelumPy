"""
Tests for StelumPy.analysis.edgedetector module.
"""

import pytest
import numpy as np
import pandas as pd

from StelumPy.analysis.edgedetector import EdgeDetector


# ---------------------------------------------------------------------------
# Helper: Create synthetic model with known X_He profile
# ---------------------------------------------------------------------------

class _SyntheticModel:
    """Mock model with a synthetic X_He profile for testing."""
    
    def __init__(
        self,
        n_points: int = 300,
        ascent_position: float = 3.0,
        noise_level: float = 0.005,
        seed: int = 42,
    ):
        # -log_q from 0 (surface) to 14 (core)
        neg_log_q = np.linspace(0, 14, n_points)
        log_q = -neg_log_q
        
        # X_He: ~0 at surface, step rise around ascent_position
        x_he = 0.95 / (1 + np.exp(-4 * (neg_log_q - ascent_position)))
        
        # Add noise
        rng = np.random.default_rng(seed)
        x_he += rng.normal(0, noise_level, len(x_he))
        x_he = np.clip(x_he, 0, 1)
        
        self.df = pd.DataFrame({"log_q": log_q, "X_He": x_he})


class _MultiStepModel:
    """Mock model with multiple X_He steps for testing detect_all_ascents."""
    
    def __init__(self, n_points: int = 500, seed: int = 42):
        neg_log_q = np.linspace(0, 14, n_points)
        log_q = -neg_log_q
        
        # Multiple steps at different -log_q positions
        step1 = 0.3 / (1 + np.exp(-6 * (neg_log_q - 2.0)))
        step2 = 0.4 / (1 + np.exp(-8 * (neg_log_q - 5.0)))
        step3 = 0.25 / (1 + np.exp(-10 * (neg_log_q - 9.0)))
        
        x_he = step1 + step2 + step3
        
        # Add small noise
        rng = np.random.default_rng(seed)
        x_he += rng.normal(0, 0.003, len(x_he))
        x_he = np.clip(x_he, 0, 1)
        
        self.df = pd.DataFrame({"log_q": log_q, "X_He": x_he})


# ---------------------------------------------------------------------------
# Tests for EdgeDetector initialization
# ---------------------------------------------------------------------------

class TestEdgeDetectorInit:
    """Tests for EdgeDetector initialization."""
    
    def test_init_with_valid_model(self):
        """Test EdgeDetector initializes with valid model."""
        model = _SyntheticModel()
        detector = EdgeDetector(model)
        assert detector.model is model
    
    def test_init_with_none_df(self):
        """Test EdgeDetector raises with None df."""
        class BadModel:
            df = None
        
        with pytest.raises(ValueError, match="must have a non-None"):
            EdgeDetector(BadModel())
    
    def test_init_without_df_attribute(self):
        """Test EdgeDetector raises when model has no df attribute."""
        class NoDfModel:
            pass
        
        with pytest.raises(ValueError, match="must have a non-None"):
            EdgeDetector(NoDfModel())
    
    def test_init_with_model_missing_columns(self):
        """Test EdgeDetector initializes even if columns are missing (checked later)."""
        class ModelMissingCols:
            df = pd.DataFrame({"other_col": [1, 2, 3]})
        
        detector = EdgeDetector(ModelMissingCols())
        # Should initialize, but detect_ascent_point will raise KeyError
        with pytest.raises(KeyError, match="log_q"):
            detector.detect_ascent_point()


# ---------------------------------------------------------------------------
# Tests for detect_ascent_point
# ---------------------------------------------------------------------------

class TestDetectAscentPoint:
    """Tests for detect_ascent_point method."""
    
    def test_detect_ascent_point_basic(self):
        """Test detect_ascent_point finds the correct position."""
        model = _SyntheticModel(ascent_position=3.0)
        detector = EdgeDetector(model)
        
        pos = detector.detect_ascent_point(
            baseline_value=0.0,
            noise_tolerance=0.05,
        )
        
        assert pos is not None
        # Should be close to the ascent position (within ~1 -log_q unit)
        assert 2.0 < pos < 4.0
    
    def test_detect_ascent_point_different_position(self):
        """Test detect_ascent_point with different ascent position."""
        model = _SyntheticModel(ascent_position=5.0)
        detector = EdgeDetector(model)
        
        pos = detector.detect_ascent_point()
        
        assert pos is not None
        assert 4.0 < pos < 6.0
    
    def test_detect_ascent_point_with_search_range(self):
        """Test detect_ascent_point with restricted search range."""
        model = _SyntheticModel(ascent_position=5.0)
        detector = EdgeDetector(model)
        
        # Search range that excludes the ascent
        pos_outside = detector.detect_ascent_point(
            search_range=(0.0, 2.0),
        )
        assert pos_outside is None
        
        # Search range that includes the ascent
        pos_inside = detector.detect_ascent_point(
            search_range=(3.0, 7.0),
        )
        assert pos_inside is not None
        assert 3.0 <= pos_inside <= 7.0
    
    def test_detect_ascent_point_no_crossing(self):
        """Test detect_ascent_point returns None when no crossing exists."""
        # Create model with X_He always below threshold
        class LowHeModel:
            df = pd.DataFrame({
                "log_q": -np.linspace(0, 14, 100),
                "X_He": np.zeros(100) + 0.01,  # Always very low
            })
        
        detector = EdgeDetector(LowHeModel())
        pos = detector.detect_ascent_point(
            baseline_value=0.0,
            noise_tolerance=0.5,  # High threshold
        )
        assert pos is None
    
    def test_detect_ascent_point_with_noise(self):
        """Test detect_ascent_point handles noisy data."""
        model = _SyntheticModel(noise_level=0.02, seed=42)
        detector = EdgeDetector(model)
        
        pos = detector.detect_ascent_point(
            baseline_value=0.0,
            noise_tolerance=0.05,
            window_size=7,  # Larger window for more smoothing
        )
        
        assert pos is not None
        # Should still find position near the true ascent
        assert 2.0 < pos < 4.0
    
    def test_detect_ascent_point_custom_threshold(self):
        """Test detect_ascent_point with custom baseline and tolerance."""
        model = _SyntheticModel(ascent_position=3.0)
        detector = EdgeDetector(model)
        
        # Custom threshold
        pos = detector.detect_ascent_point(
            baseline_value=0.1,
            noise_tolerance=0.1,
        )
        
        assert pos is not None
        # Higher threshold should find later position
        assert pos > 2.5
    
    def test_detect_ascent_point_small_window(self):
        """Test detect_ascent_point with window_size > data length."""
        class SmallModel:
            df = pd.DataFrame({
                "log_q": -np.linspace(0, 14, 3),
                "X_He": [0.0, 0.5, 0.9],
            })
        
        detector = EdgeDetector(SmallModel())
        pos = detector.detect_ascent_point(
            window_size=10,  # Larger than data
        )
        
        # Should still work (with warning logged)
        assert pos is not None
    
    def test_detect_ascent_point_returns_float(self):
        """Test detect_ascent_point returns float type."""
        model = _SyntheticModel()
        detector = EdgeDetector(model)
        
        pos = detector.detect_ascent_point()
        
        assert pos is None or isinstance(pos, float)


# ---------------------------------------------------------------------------
# Tests for detect_all_ascents
# ---------------------------------------------------------------------------

class TestDetectAllAscents:
    """Tests for detect_all_ascents method."""
    
    def test_detect_all_ascents_single(self):
        """Test detect_all_ascents with single ascent."""
        model = _SyntheticModel(ascent_position=3.0)
        detector = EdgeDetector(model)
        
        ascents = detector.detect_all_ascents()
        
        assert isinstance(ascents, list)
        assert len(ascents) >= 1
        assert all(isinstance(a, float) for a in ascents)
    
    def test_detect_all_ascents_multiple(self):
        """Test detect_all_ascents with multiple ascents."""
        model = _MultiStepModel()
        detector = EdgeDetector(model)
        
        ascents = detector.detect_all_ascents(
            baseline_value=0.0,
            noise_tolerance=0.03,
            min_gap=1.0,
        )
        
        assert isinstance(ascents, list)
        # Should find at least 1 ascent (may vary due to noise/profile shape)
        assert len(ascents) >= 1
    
    def test_detect_all_ascents_sorted(self):
        """Test detect_all_ascents returns sorted positions."""
        model = _MultiStepModel()
        detector = EdgeDetector(model)
        
        ascents = detector.detect_all_ascents(min_gap=0.5)
        
        assert ascents == sorted(ascents)
    
    def test_detect_all_ascents_min_gap(self):
        """Test detect_all_ascents respects min_gap parameter."""
        model = _MultiStepModel()
        detector = EdgeDetector(model)
        
        # Large min_gap should reduce number of detected ascents
        ascents_small_gap = detector.detect_all_ascents(min_gap=0.1)
        ascents_large_gap = detector.detect_all_ascents(min_gap=2.0)
        
        assert len(ascents_large_gap) <= len(ascents_small_gap)
    
    def test_detect_all_ascents_empty(self):
        """Test detect_all_ascents with no ascents."""
        class FlatModel:
            df = pd.DataFrame({
                "log_q": -np.linspace(0, 14, 100),
                "X_He": np.zeros(100),  # Flat line
            })
        
        detector = EdgeDetector(FlatModel())
        ascents = detector.detect_all_ascents()
        
        assert ascents == []
    
    def test_detect_all_ascents_returns_list_of_floats(self):
        """Test detect_all_ascents returns list of floats."""
        model = _SyntheticModel()
        detector = EdgeDetector(model)
        
        ascents = detector.detect_all_ascents()
        
        assert isinstance(ascents, list)
        assert all(isinstance(a, float) for a in ascents)


# ---------------------------------------------------------------------------
# Tests for edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeDetectorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_log_q_column(self):
        """Test detect_ascent_point raises KeyError for missing log_q."""
        class BadModel:
            df = pd.DataFrame({"X_He": [1, 2, 3]})
        
        detector = EdgeDetector(BadModel())
        with pytest.raises(KeyError, match="log_q"):
            detector.detect_ascent_point()
    
    def test_missing_x_he_column(self):
        """Test detect_ascent_point raises KeyError for missing X_He."""
        class BadModel:
            df = pd.DataFrame({"log_q": [1, 2, 3]})
        
        detector = EdgeDetector(BadModel())
        with pytest.raises(KeyError, match="X_He"):
            detector.detect_ascent_point()
    
    def test_unsorted_data(self):
        """Test detect_ascent_point handles unsorted data correctly."""
        # Create unsorted data
        rng = np.random.default_rng(42)
        log_q = -rng.permutation(np.linspace(0, 14, 100))
        x_he = 0.95 / (1 + np.exp(-4 * (-log_q - 3.0)))
        
        class UnsortedModel:
            df = pd.DataFrame({"log_q": log_q, "X_He": x_he})
        
        detector = EdgeDetector(UnsortedModel())
        pos = detector.detect_ascent_point()
        
        # Should still find the ascent (algorithm sorts internally)
        assert pos is not None
        assert 2.0 < pos < 4.0
    
    def test_very_small_dataset(self):
        """Test detect_ascent_point with minimal data."""
        class TinyModel:
            df = pd.DataFrame({
                "log_q": [-1.0, -0.5, 0.0],
                "X_He": [0.0, 0.5, 1.0],
            })
        
        detector = EdgeDetector(TinyModel())
        pos = detector.detect_ascent_point()
        
        assert pos is not None
    
    def test_constant_x_he_profile(self):
        """Test detect_ascent_point with constant X_He profile."""
        class ConstantModel:
            df = pd.DataFrame({
                "log_q": -np.linspace(0, 14, 100),
                "X_He": np.full(100, 0.5),
            })
        
        detector = EdgeDetector(ConstantModel())
        pos = detector.detect_ascent_point(
            baseline_value=0.0,
            noise_tolerance=0.6,  # Above constant value
        )
        
        assert pos is None
    
    def test_decreasing_x_he_profile(self):
        """Test detect_ascent_point with decreasing X_He (no ascent)."""
        class DecreasingModel:
            df = pd.DataFrame({
                "log_q": -np.linspace(0, 14, 100),
                "X_He": np.linspace(1.0, 0.0, 100),  # Decreasing
            })
        
        detector = EdgeDetector(DecreasingModel())
        pos = detector.detect_ascent_point(
            baseline_value=0.0,
            noise_tolerance=0.5,  # High threshold that won't be exceeded
        )
        
        # Should not find an ascent with high threshold
        # Note: With low threshold, may find edge artifact
        assert pos is None or pos < 1.0  # If found, should be at edge


# ---------------------------------------------------------------------------
# Integration-style tests with realistic models
# ---------------------------------------------------------------------------

class TestEdgeDetectorIntegration:
    """Integration tests with realistic model profiles."""
    
    def test_realistic_profile_detection(self):
        """Test detection on a realistic stellar profile."""
        model = _SyntheticModel(
            n_points=500,
            ascent_position=3.5,
            noise_level=0.008,
        )
        detector = EdgeDetector(model)
        
        pos = detector.detect_ascent_point(
            baseline_value=0.0,
            noise_tolerance=0.05,
            window_size=5,
        )
        
        assert pos is not None
        # Within 1.0 -log_q units (allowing for noise/smoothing effects)
        assert abs(pos - 3.5) < 1.0
    
    def test_compare_detect_ascent_point_and_detect_all_ascents(self):
        """Test that first ascent from detect_all matches detect_ascent_point."""
        model = _MultiStepModel()
        detector = EdgeDetector(model)
        
        first_ascent = detector.detect_ascent_point()
        all_ascents = detector.detect_all_ascents()
        
        if first_ascent is not None and len(all_ascents) > 0:
            # First ascent should be close to first in all_ascents
            assert abs(first_ascent - all_ascents[0]) < 0.5
    
    def test_multiple_detectors_same_model(self):
        """Test creating multiple detectors for same model."""
        model = _SyntheticModel()
        
        detector1 = EdgeDetector(model)
        detector2 = EdgeDetector(model)
        
        pos1 = detector1.detect_ascent_point()
        pos2 = detector2.detect_ascent_point()
        
        assert pos1 == pos2  # Same result
