"""
Tests for StelumPy.io.sequence module.
"""

import logging
import pytest
from pathlib import Path

from StelumPy.io.sequence import Sequence
from StelumPy.io.model import Model
from StelumPy.exceptions import SequenceFileError, ValidationError


class TestSequenceInit:
    """Tests for Sequence initialization."""

    def test_init_with_valid_directory(self, temp_dir):
        """Test Sequence loads a valid directory without errors."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=3)
        seq = Sequence(seq_dir, verbose=False)
        assert seq.sq_directory == seq_dir
        assert len(seq.models) == 3

    def test_init_with_nonexistent_directory(self, temp_dir):
        """Test Sequence raises SequenceFileError for missing directory."""
        nonexistent = temp_dir / "nonexistent"
        with pytest.raises(SequenceFileError):
            Sequence(nonexistent, verbose=False)

    def test_init_without_seq_file(self, temp_dir):
        """Test Sequence raises SequenceFileError when seq.txt missing."""
        seq_dir = temp_dir / "incomplete_seq"
        models_dir = seq_dir / "5mext"
        models_dir.mkdir(parents=True)
        # Don't create seq.txt
        with pytest.raises(SequenceFileError):
            Sequence(seq_dir, verbose=False)

    def test_init_without_models_directory(self, temp_dir):
        """Test Sequence raises SequenceFileError when 5mext missing."""
        seq_dir = temp_dir / "incomplete_seq2"
        seq_dir.mkdir()
        (seq_dir / "seq.txt").write_text("dummy")
        # Don't create 5mext/
        with pytest.raises(SequenceFileError):
            Sequence(seq_dir, verbose=False)


class TestSequenceLoading:
    """Tests for Sequence file loading."""

    def test_models_loaded(self, sample_sequence):
        """Test that models are loaded correctly."""
        assert len(sample_sequence.models) == 3
        assert all(isinstance(m, Model) for m in sample_sequence.models)

    def test_seq_data_loaded(self, sample_sequence):
        """Test that seq.txt data is loaded."""
        assert sample_sequence.seq_data is not None
        assert len(sample_sequence.seq_data) == 3

    def test_age_sequence_loaded(self, sample_sequence):
        """Test that age_sequence is extracted from seq.txt."""
        assert sample_sequence.age_sequence is not None
        assert len(sample_sequence.age_sequence) == 3

    def test_file_paths_stored(self, sample_sequence):
        """Test that file_paths are stored correctly."""
        assert len(sample_sequence.file_paths) == 3
        assert all(isinstance(p, Path) for p in sample_sequence.file_paths)

    def test_model_index_stored(self, sample_sequence):
        """Test that model_index is stored correctly."""
        assert len(sample_sequence.model_index) == 3


class TestSequenceAccessors:
    """Tests for Sequence accessor methods."""

    def test_get_model_valid_index(self, sample_sequence):
        """Test get_model with valid index."""
        model = sample_sequence.get_model(0)
        assert isinstance(model, Model)

    def test_get_model_invalid_index(self, sample_sequence):
        """Test get_model with out-of-range index."""
        with pytest.raises(ValidationError):
            sample_sequence.get_model(10)

    def test_get_model_negative_index(self, sample_sequence):
        """Test get_model with negative index."""
        with pytest.raises(ValidationError):
            sample_sequence.get_model(-1)

    def test_get_age_valid_index(self, sample_sequence):
        """Test get_age with valid index."""
        age = sample_sequence.get_age(0)
        assert isinstance(age, float)

    def test_get_age_invalid_index(self, sample_sequence):
        """Test get_age with out-of-range index."""
        with pytest.raises(ValidationError):
            sample_sequence.get_age(10)

    def test_getitem(self, sample_sequence):
        """Test __getitem__ magic method."""
        model = sample_sequence[0]
        assert isinstance(model, Model)

    def test_len(self, sample_sequence):
        """Test __len__ magic method."""
        assert len(sample_sequence) == 3

    def test_repr(self, sample_sequence):
        """Test __repr__ returns expected format."""
        result = repr(sample_sequence)
        assert "Sequence" in result
        assert "num_models=3" in result


class TestSequenceMaxModels:
    """Tests for Sequence max_models parameter."""

    def test_max_models_limits_loading(self, temp_dir):
        """Test that max_models limits the number of loaded models."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=5)
        seq = Sequence(seq_dir, max_models=2, verbose=False)
        assert len(seq.models) == 2
        assert len(seq.file_paths) == 2


class TestSequenceVerbose:
    """Tests for Sequence verbose output."""

    def test_verbose_output(self, temp_dir, caplog):
        """Test that verbose=True produces logging output."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=2)
        
        with caplog.at_level(logging.INFO):
            seq = Sequence(seq_dir, verbose=True)
        
        # Check for expected log messages
        assert any("Loading" in record.message for record in caplog.records)


class TestSequenceExport:
    """Tests for Sequence export functionality."""

    def test_export_evolution_csv(self, sample_sequence, temp_dir):
        """Test export_evolution_csv creates valid file."""
        output_file = temp_dir / "evolution.csv"
        parameters = ['T_eff', 'log_g']
        sample_sequence.export_evolution_csv(output_file, parameters)
        assert output_file.exists()
        
        import pandas as pd
        df = pd.read_csv(output_file)
        assert 'Age' in df.columns
        assert 'T_eff' in df.columns
