"""
Tests for StelumPy.cli module.
"""

import pytest
from pathlib import Path

from StelumPy.cli import find_best_model, _build_parser


class TestFindBestModel:
    """Tests for find_best_model function."""

    def test_find_best_model_basic(self, temp_dir):
        """Test find_best_model returns correct structure."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=3)
        
        result = find_best_model(
            seq_path=seq_dir,
            target_he=0.5,
            n_points=5,
            verbose=False,
        )
        
        assert 'index' in result
        assert 'model' in result
        assert 'he_core' in result
        assert 'delta' in result
        assert 'age' in result
        assert 'T_eff' in result
        assert 'log_g' in result
        assert 'copied_to' in result

    def test_find_best_model_copy(self, temp_dir):
        """Test find_best_model copies file to destination."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=3)
        copy_dir = temp_dir / "copied"
        
        result = find_best_model(
            seq_path=seq_dir,
            target_he=0.5,
            n_points=5,
            copy_to=copy_dir,
            verbose=False,
        )
        
        assert result['copied_to'] is not None
        assert result['copied_to'].exists()

    def test_find_best_model_no_copy(self, temp_dir):
        """Test find_best_model with copy_to=None."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=3)
        
        result = find_best_model(
            seq_path=seq_dir,
            target_he=0.5,
            n_points=5,
            copy_to=None,
            verbose=False,
        )
        
        assert result['copied_to'] is None

    def test_find_best_model_copy_seq(self, temp_dir):
        """Test find_best_model copies seq.txt when requested."""
        from .helpers import create_test_sequence_directory
        seq_dir = create_test_sequence_directory(temp_dir, num_models=3)
        copy_dir = temp_dir / "copied"
        
        result = find_best_model(
            seq_path=seq_dir,
            target_he=0.5,
            n_points=5,
            copy_to=copy_dir,
            copy_seq=True,
            verbose=False,
        )
        
        seq_copy = copy_dir / "seq.txt"
        assert seq_copy.exists()


class TestBuildParser:
    """Tests for CLI argument parser."""

    def test_parser_basic_args(self):
        """Test parser with required arguments."""
        parser = _build_parser()
        args = parser.parse_args(['/path/to/seq', '0.5'])
        
        assert args.seq_path == '/path/to/seq'
        assert args.target_he == 0.5
        assert args.n_points == 1
        assert args.copy_to is None
        assert args.copy_seq is False
        assert args.quiet is False

    def test_parser_all_options(self):
        """Test parser with all options."""
        parser = _build_parser()
        args = parser.parse_args([
            '/path/to/seq', '0.5',
            '--n_points', '5',
            '--copy_to', '/output/dir',
            '--copy_seq',
            '--quiet',
        ])
        
        assert args.n_points == 5
        assert args.copy_to == '/output/dir'
        assert args.copy_seq is True
        assert args.quiet is True

    def test_parser_no_copy(self):
        """Test parser with --no_copy flag."""
        parser = _build_parser()
        args = parser.parse_args([
            '/path/to/seq', '0.5',
            '--no_copy',
        ])
        
        assert args.no_copy is True

    def test_parser_help(self):
        """Test parser --help option."""
        parser = _build_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
