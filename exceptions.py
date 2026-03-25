"""
StelumPy Exceptions
===================
Custom exception classes for StelumPy package.
"""

from __future__ import annotations


class StelumPyError(Exception):
    """Base exception for StelumPy."""
    pass


class ModelFileError(StelumPyError):
    """Error reading or parsing a model file."""
    pass


class SequenceFileError(StelumPyError):
    """Error loading a sequence directory."""
    pass


class MatchingError(StelumPyError):
    """Error during model matching analysis."""
    pass


class ProfileColumnError(StelumPyError):
    """Error accessing profile column data."""
    pass


class ValidationError(StelumPyError):
    """Error in input validation."""
    pass
