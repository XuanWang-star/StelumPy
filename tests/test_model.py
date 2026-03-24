"""
Tests for StelumPy.io.model module.
"""

import pytest
from pathlib import Path

from StelumPy.io.model import Model


class TestModelInit:
    """Tests for Model initialization and file reading."""

    def test_init_with_valid_file(self, sample_model_file):
        """Test Model loads a valid file without errors."""
        model = Model(sample_model_file)
        assert model.file_path == sample_model_file
        assert model.df is not None

    def test_init_with_nonexistent_file(self, temp_dir):
        """Test Model raises FileNotFoundError for missing file."""
        nonexistent = temp_dir / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            Model(nonexistent)

    def test_file_path_is_path_object(self, sample_model_file):
        """Test that file_path is stored as a Path object."""
        model = Model(sample_model_file)
        assert isinstance(model.file_path, Path)


class TestModelHeader:
    """Tests for Model header parsing."""

    def test_data_type_parsed(self, sample_model):
        """Test that data type is correctly parsed."""
        assert sample_model.data_type == "PROFILE"

    def test_mesh_number_parsed(self, sample_model):
        """Test that mesh number is correctly parsed."""
        assert sample_model.mesh_number == 10

    def test_teff_parsed(self, sample_model):
        """Test that T_eff is correctly parsed."""
        assert sample_model.T_eff == 5770.0

    def test_log_g_parsed(self, sample_model):
        """Test that log_g is correctly parsed."""
        assert sample_model.log_g == 4.5


class TestModelData:
    """Tests for Model data parsing and access."""

    def test_dataframe_shape(self, sample_model):
        """Test that DataFrame has correct shape."""
        assert sample_model.df.shape == (10, 54)

    def test_dataframe_columns(self, sample_model):
        """Test that DataFrame has expected columns."""
        expected_cols = ['n', 'r', 'm_r', 'rho', 'P', 'T', 'X_He', 'X_C', 'X_O']
        for col in expected_cols:
            assert col in sample_model.df.columns

    def test_get_column(self, sample_model):
        """Test get_column method returns correct data."""
        column = sample_model.get_column('X_He')
        assert len(column) == 10

    def test_get_column_invalid(self, sample_model):
        """Test get_column raises for invalid column name."""
        with pytest.raises(KeyError):
            sample_model.get_column('nonexistent_column')


class TestModelHeCore:
    """Tests for Model he_core_he method."""

    def test_he_core_he_default(self, sample_model):
        """Test he_core_he with default n_points."""
        result = sample_model.he_core_he()
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_he_core_he_custom_points(self, sample_model):
        """Test he_core_he with custom n_points."""
        result_1 = sample_model.he_core_he(n_points=1)
        result_5 = sample_model.he_core_he(n_points=5)
        assert isinstance(result_1, float)
        assert isinstance(result_5, float)

    def test_he_core_he_no_data(self, sample_model):
        """Test he_core_he raises when no data loaded."""
        sample_model.df = None
        with pytest.raises(ValueError, match="No profile data loaded"):
            sample_model.he_core_he()


class TestModelSummary:
    """Tests for Model summary and representation."""

    def test_summary_runs(self, sample_model, capsys):
        """Test that summary method runs without errors."""
        sample_model.summary()
        captured = capsys.readouterr()
        assert "Model Summary" in captured.out
        assert sample_model.file_path.name in captured.out

    def test_repr(self, sample_model):
        """Test __repr__ returns expected format."""
        result = repr(sample_model)
        assert "Model" in result
        assert "mesh=10" in result
        assert "T_eff=5770" in result


def create_minimal_model_file(filepath: Path, skip_data: bool = False) -> Path:
    """Helper to create minimal model file for edge case testing."""
    if skip_data:
        content = """PROFILE
  0   1.0000E+00   0.0000E+00   4.5000E+00   0.0000E+00   5.7700E+03
"""
    else:
        content = """PROFILE
  1   1.0000E+00   0.0000E+00   4.5000E+00   0.0000E+00   5.7700E+03
  1  1.0000E+10  1.0000E+00  1.0000E+02  1.0000E+15  1.0000E+07
  0.500  0.300  0.700  0.001  0.002  0.003  0.004
  1.000E+01  2.000E+01  3.000E+01  4.000E+01  5.000E+01
  6.000E+01  7.000E+01  8.000E+01  9.000E+01  1.000E+02
"""
    filepath.write_text(content)
    return filepath
