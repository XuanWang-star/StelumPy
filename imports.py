"""
StelumPy.imports
=================
Centralised import manager for the StelumPy package.

Run as a script to verify all dependencies are installed:

    python -m StelumPy.imports          # via package
    python imports.py                   # direct

Import in user scripts for a one-line setup:

    from StelumPy.imports import *      # pulls in all public names below
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------------
# Third-party: core
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, RadioButtons

# ---------------------------------------------------------------------------
# StelumPy — I/O
# ---------------------------------------------------------------------------
from StelumPy.io.model    import Model
from StelumPy.io.sequence import Sequence

# ---------------------------------------------------------------------------
# StelumPy — Analysis
# ---------------------------------------------------------------------------
from StelumPy.analysis.matching     import SequenceAnalyzer
from StelumPy.analysis.edgedetector import EdgeDetector

# ---------------------------------------------------------------------------
# StelumPy — Plotting
# ---------------------------------------------------------------------------
from StelumPy.plotting.plots       import SequencePlotter
from StelumPy.plotting.interactive import ModelExplorer, SequenceExplorer

# ---------------------------------------------------------------------------
# Public names exported by  `from StelumPy.imports import *`
# ---------------------------------------------------------------------------
__all__ = [
    # standard library conveniences
    "Path", "Optional", "Tuple", "List", "Dict", "Any",
    # third-party
    "np", "pd", "plt", "matplotlib", "gridspec",
    # StelumPy I/O
    "Model", "Sequence",
    # StelumPy analysis
    "SequenceAnalyzer", "EdgeDetector",
    # StelumPy plotting
    "SequencePlotter", "ModelExplorer", "SequenceExplorer",
]

# ---------------------------------------------------------------------------
# Version table (shown when run as script)
# ---------------------------------------------------------------------------
_PACKAGES: dict[str, str] = {
    "numpy":      np.__version__,
    "pandas":     pd.__version__,
    "matplotlib": matplotlib.__version__,
}

_STELUMPY_MODULES: list[str] = [
    "Model", "Sequence",
    "SequenceAnalyzer", "EdgeDetector",
    "SequencePlotter", "ModelExplorer", "SequenceExplorer",
]

# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sep = "=" * 45
    print(sep)
    print("  StelumPy — Dependency & Import Check")
    print(sep)

    print("\n[Third-party packages]")
    for name, version in _PACKAGES.items():
        print(f"  ok  {name:<15} {version}")

    print("\n[StelumPy modules]")
    all_ok = True
    for name in _STELUMPY_MODULES:
        obj = globals().get(name)
        status = "ok" if obj is not None else "FAIL"
        print(f"  {status}  {name}")
        if obj is None:
            all_ok = False

    print(sep)
    print("  All imports successful." if all_ok else "  Some imports failed.")
    print(sep)
