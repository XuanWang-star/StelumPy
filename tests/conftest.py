"""
StelumPy Test Fixtures
======================
Shared fixtures and test data for all tests.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from StelumPy import Model, Sequence
from .helpers import create_test_model_file, create_test_seq_file, create_test_sequence_directory

__all__ = [
    "create_test_model_file",
    "create_test_seq_file",
    "create_test_sequence_directory",
    "temp_dir",
    "sample_model_file",
    "sample_model",
    "sample_sequence",
    "sample_data",
]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_model_file(temp_dir):
    """Create a sample model file and return its path."""
    return create_test_model_file(temp_dir)


@pytest.fixture
def sample_model(sample_model_file):
    """Load a sample Model for testing."""
    return Model(sample_model_file)


@pytest.fixture
def sample_sequence(temp_dir):
    """Create and return a sample Sequence."""
    seq_dir = create_test_sequence_directory(temp_dir, num_models=3)
    return Sequence(seq_dir, verbose=False)


@pytest.fixture
def sample_data():
    """Return sample numpy arrays for testing analysis functions."""
    np.random.seed(42)
    return {
        'log_q': np.linspace(-10, 0, 100),
        'X_He': np.linspace(0.98, 0.30, 100) + np.random.normal(0, 0.01, 100),
        'ages': np.logspace(6, 8, 50),
        'T_eff': np.linspace(5000, 8000, 50),
        'log_g': np.linspace(4.5, 3.5, 50),
    }
