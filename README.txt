StelumPy - Stellar Model Profile Sequence Analysis
===================================================

StelumPy is a Python package for reading, analysing, and visualising stellar
model profile sequences. It provides tools to load individual stellar models
or entire evolution sequences, perform matching analysis, and generate
publication-quality visualisations.


Requirements
------------
- Python >= 3.10
- numpy
- pandas
- matplotlib


Installation
------------
From source directory:

    pip install .

For development:

    pip install -e .


Quick Start
-----------
>>> from StelumPy import Model, Sequence, SequenceAnalyzer, SequencePlotter

>>> # Load a single static model
>>> target = Model("path/to/static_model.txt")

>>> # Load an evolution sequence
>>> seq = Sequence("path/to/sq/")

>>> # Analyse - find models by core He abundance
>>> analyzer = SequenceAnalyzer(seq)
>>> result = analyzer.find_model_by_he_core(target.he_core_he())

>>> # Compare He profiles between models
>>> metrics = analyzer.he_profile_match(target, result['model'])

>>> # Visualise
>>> plotter = SequencePlotter(seq)
>>> (fig, ax), metrics = plotter.plot_he_profile_match(target, result['model'])


Directory Structure
-------------------
StelumPy/
├── io/          File I/O modules
│   ├── model.py       Model class - reads single stellar model files
│   └── sequence.py    Sequence class - loads model evolution sequences
├── analysis/  Analysis and matching tools
│   └── matching.py    SequenceAnalyzer - profile matching & statistics
├── plotting/  Visualisation modules
│   └── plots.py       SequencePlotter - generate stellar evolution plots
└── __init__.py  Main package exports


Input File Format
-----------------
StelumPy expects stellar model files in the following structure:

Single Model File:
    Line 1: Data type identifier
    Line 2: Header with mesh_number, T_eff, log_g values
    Lines 3+: Profile data (5 lines per mesh point)

Evolution Sequence Directory (sq/):
    sq/
    ├── 5mext/      Text-format profile files (one per time step)
    ├── seq.txt     Global evolution table
    └── idx.txt     Optional index file


Main Classes
------------

Model
    Reads and holds data from one stellar model profile file.
    
    Key attributes:
        - file_path      Path to the model file
        - data_type      Model type identifier
        - mesh_number    Number of mesh points
        - T_eff          Effective temperature
        - log_g          Surface gravity
        - df             DataFrame with all profile columns
    
    Key methods:
        - get_column(name)     Return a profile column
        - he_core_he(n)        Mean X_He in core (n points)
        - summary()            Print model summary

Sequence
    Loads a sequence of stellar model files from sq/ directory structure.
    
    Key attributes:
        - sq_directory     Path to sequence directory
        - models           List of loaded Model objects
        - age_sequence     Age values from seq.txt
        - seq_data         Full evolution table DataFrame
    
    Key methods:
        - get_model(index)   Get model at position
        - get_age(index)     Get age at position
        - summary()          Print sequence summary

SequenceAnalyzer
    Performs matching and data extraction on a Sequence.
    
    Key methods:
        - get_evolution_data(param)       Extract parameter evolution
        - get_he_core_evolution(n)        Core He evolution
        - find_model_by_he_core(he)       Find model by core He
        - find_models_by_he_core(he, n)   Find n best matches
        - he_profile_match(target, snap)  Compare He profiles
        - find_closest_model(Teff, logg)  Match by Teff/log_g
        - create_evolution_dataframe()    Build evolution DataFrame

SequencePlotter
    Generates visualisations for stellar model sequences.
    
    Key methods:
        - plot_evolution(param)           Plot parameter vs age
        - plot_hr_diagram()               Hertzsprung-Russell diagram
        - compare_profiles(col, indices)  Overlay profiles
        - plot_he_profile_match(t, s)     He profile comparison


Example Workflows
-----------------

1. Find Evolution Stage by Core Helium:

    target = Model("observation.txt")
    seq = Sequence("models/sq/")
    analyzer = SequenceAnalyzer(seq)
    
    target_he = target.he_core_he()
    match = analyzer.find_model_by_he_core(target_he)
    
    print(f"Best match at age {match['age']:.2e} yr")
    print(f"Core He: {match['he_core']:.4f} (target: {target_he:.4f})")

2. Compare Helium Profiles:

    metrics = analyzer.he_profile_match(target, match['model'])
    print(f"Profile RMSE: {metrics['rmse']:.4f}")
    print(f"R² score: {metrics['r2']:.4f}")

3. Visualise Evolution:

    plotter = SequencePlotter(seq)
    
    # HR diagram
    fig, ax = plotter.plot_hr_diagram()
    
    # Parameter evolution
    fig, ax = plotter.plot_evolution('X_He')
    
    # Profile comparison at different ages
    fig, ax = plotter.compare_profiles('X_He', [0, 50, 100])

4. Export Evolution Data:

    seq.export_evolution_csv("evolution.csv", 
                             ['X_He', 'X_C', 'X_O', 'T_eff'])


Profile Columns
---------------
Available columns in model profile DataFrames:

    n, r, m_r, rho, P, T, chi_p, chi_T, Del, Del_ad,
    Y, B, log_q, flag14, flag15, unknown,
    L, eta, eta_r, kappa, kappa_rho, kappa_T, Del_rad,
    zeta, epsilon_N, epsilon_rho, epsilon_T,
    tau, w, wtau, Del_r, Del_P, Del_T, Del_tau, Del_L,
    Del_ad_P, Del_ad_T, Del_ad_minus_Del,
    U, dU_dP, dU_dT, C_p, C_V, eta_e, Z_moy, Gamma,
    X_H, X_He, X_C, X_O, log_P_gas, unknown1, unknown2, unknown3


Evolution Table Columns (seq.txt)
---------------------------------
    Mod, Teff, Log_g, Rayon, Age, Lum,
    Log_Tc, Log_Pc, Log_rhoc, M_x_M, Log_q_x,
    Lum_nu, Log_H, Log_He, Log_C, Log_O


License
-------
See LICENSE file for terms.


Version
-------
0.1.0
