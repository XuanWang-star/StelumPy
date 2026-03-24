StelumPy
========

Read, analyse, and visualise stellar model profile sequences.

Installation
------------
pip install -e .

# With test dependencies
pip install -e ".[test]"

# With dev dependencies
pip install -e ".[dev]"

Quick Start
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

Documentation
-------------
See docs/README.md for full documentation and docs/EXAMPLES.md for examples.

Testing
-------
Run tests with pytest:

    pytest tests/ -v

Run with coverage:

    pytest tests/ -v --cov=StelumPy --cov-report=html

Command-Line Interface
----------------------
Find the best-matching model by core He abundance:

    find_best_model /path/to/seq 0.2345 --n_points 10 --copy_to ./results

Dependencies
------------
- numpy
- pandas
- matplotlib

Optional (testing):
- pytest
- pytest-cov

License
-------
MIT License
