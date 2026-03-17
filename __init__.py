"""
StelumPy
==========
A Python package for reading, analysing, and visualising stellar model data.

Subpackages
-----------
io          — file I/O: Model, Sequence
analysis    — matching and statistics: SequenceAnalyzer
plotting    — visualisation: SequencePlotter

Quick start
-----------
>>> from StelumPy import Model, Sequence, SequenceAnalyzer, SequencePlotter
>>>
>>> # Load a single static model
>>> target = Model("path/to/static_model.txt")
>>>
>>> # Load an evolution sequence
>>> seq = Sequence("path/to/sq/")
>>>
>>> # Analyse
>>> analyzer = SequenceAnalyzer(seq)
>>> result = analyzer.find_model_by_he_core(target.he_core_he())
>>> metrics = analyzer.he_profile_match(target, result['model'])
>>>
>>> # Plot
>>> plotter = SequencePlotter(seq)
>>> (fig, ax), metrics = plotter.plot_he_profile_match(target, result['model'])
"""

from .io.model     import Model
from .io.sequence  import Sequence
from .analysis.matching import SequenceAnalyzer
from .plotting.plots    import SequencePlotter

__all__ = ["Model", "Sequence", "SequenceAnalyzer", "SequencePlotter"]
__version__ = "0.1.0"
