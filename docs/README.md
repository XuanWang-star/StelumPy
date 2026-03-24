# StelumPy Documentation

**StelumPy** is a Python package for reading, analysing, and visualising stellar model profile sequences.

## Installation

### Requirements
- Python 3.10+
- numpy
- pandas
- matplotlib

### Install from source
```bash
pip install -e /path/to/StelumPy
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Load a single stellar model
```python
from StelumPy import Model

# Load a static model file
target = Model("path/to/static_model.txt")

# View summary
target.summary()

# Access properties
print(f"T_eff: {target.T_eff} K")
print(f"log g: {target.log_g}")

# Get core He abundance
he_core = target.he_core_he(n_points=10)
print(f"Core He: {he_core:.4f}")

# Access profile data
df = target.df  # pandas DataFrame with all mesh points
X_He_profile = df['X_He']
log_q = df['log_q']
```

### Load an evolution sequence
```python
from StelumPy import Sequence

# Load a sequence directory (must contain 5mext/ and seq.txt)
seq = Sequence("path/to/sq/", verbose=True)

# View summary
seq.summary()

# Access individual models
first_model = seq[0]
last_model = seq[-1]

# Get model at specific index
model_5 = seq.get_model(5)

# Get age at index
age = seq.get_age(5)
```

### Analyse a sequence
```python
from StelumPy import Sequence, SequenceAnalyzer

# Load sequence
seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Find model by core He abundance
target_he = 0.2345
result = analyzer.find_model_by_he_core(target_he, n_points=10)

print(f"Best match: index {result['index']}")
print(f"Core He: {result['he_core']:.6f}")
print(f"Delta: {result['delta']:+.2e}")
print(f"Age: {result['age']:.4e} yr")

# Get top 5 closest matches
top_5 = analyzer.find_models_by_he_core(target_he, n_models=5)

# Compare He profiles between two models
target = Model("static_model.txt")
snapshot = result['model']
metrics = analyzer.he_profile_match(target, snapshot)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### Visualize data
```python
from StelumPy import Sequence, SequencePlotter

# Load sequence
seq = Sequence("path/to/sq/")
plotter = SequencePlotter(seq)

# Plot evolution of a parameter
fig, ax = plotter.plot_evolution('T_eff', use_log_time=True)

# Create HR diagram
fig, ax = plotter.plot_hr_diagram()

# Compare profiles across models
fig, ax = plotter.compare_profiles(
    'X_He',
    model_indices=[0, 5, 10, 15],
    x_column='log_q'
)

# Plot He profile match between two models
target = Model("static_model.txt")
snapshot = seq[10]
(fig, ax), metrics = plotter.plot_he_profile_match(target, snapshot)
```

---

## Command-Line Interface

### find_best_model

Find the best-matching model in a sequence by core He abundance.

```bash
# Basic usage
find_best_model /path/to/seq 0.2345

# With options
find_best_model /path/to/seq 0.2345 \
    --n_points 10 \
    --copy_to /output/dir \
    --copy_seq

# Suppress output
find_best_model /path/to/seq 0.2345 --quiet
```

**Options:**
- `SEQ_PATH`: Path to the `seq/` directory (must contain `5mext/` and `seq.txt`)
- `TARGET_HE`: Target core He abundance (X_He fraction, 0–1)
- `--n_points N`: Number of innermost mesh points for core He estimate (default: 1)
- `--copy_to DIR`: Copy matched model to this directory
- `--copy_seq`: Also copy `seq.txt` when using `--copy_to`
- `--no_copy`: Disable copying entirely
- `--quiet`: Suppress all output

---

## API Reference

### io.model.Model

Represents a single stellar model profile file.

**Attributes:**
- `file_path` (Path): Path to the model file
- `data_type` (str): Data type from header
- `mesh_number` (int): Number of mesh points
- `T_eff` (float): Effective temperature [K]
- `log_g` (float): Surface gravity
- `df` (pd.DataFrame): Profile data with columns:
  - `n`, `r`, `m_r`, `rho`, `P`, `T`
  - `chi_p`, `chi_T`, `Del`, `Del_ad`
  - `Y`, `B`, `log_q`, `X_H`, `X_He`, `X_C`, `X_O`
  - And many more (54 columns total)

**Methods:**
- `summary()`: Print model summary
- `get_column(name)`: Get a single column from profile
- `he_core_he(n_points=10)`: Mean X_He over innermost n mesh points

---

### io.sequence.Sequence

Loads an ordered sequence of stellar model files.

**Attributes:**
- `sq_directory` (Path): Path to sequence directory
- `models_directory` (Path): Path to `5mext/` subdirectory
- `models` (list[Model]): List of loaded Model objects
- `num_models` (int): Number of models
- `seq_data` (pd.DataFrame): Full seq.txt table
- `age_sequence` (np.ndarray): Age column from seq.txt
- `file_paths` (list[Path]): Paths to model files

**Methods:**
- `summary()`: Print sequence summary
- `get_model(index)`: Get model at index (0-based)
- `get_age(index)`: Get stellar age at index
- `export_evolution_csv(output, parameters)`: Export to CSV

---

### analysis.matching.SequenceAnalyzer

Performs matching and analysis on a Sequence.

**Methods:**
- `get_evolution_data(parameter)`: Get array of parameter values across sequence
- `get_profile_evolution(column, mesh_point)`: Get profile column at fixed mesh point
- `get_he_core_evolution(n_points=5)`: Get core He evolution
- `create_evolution_dataframe(parameters)`: Build DataFrame with Age + parameters
- `find_model_by_he_core(target_he, n_points=10)`: Find best match by core He
- `find_models_by_he_core(target_he, n_models=5, n_points=10)`: Find top N matches
- `he_profile_match(target, snapshot, n_points=200)`: Compare He profiles
- `find_closest_model(T_eff, log_g, ...)`: Find model by T_eff/log_g
- `find_closest_models_around(T_eff, log_g, n_models=5)`: Find top N by T_eff/log_g

---

### plotting.plots.SequencePlotter

Creates visualizations for a Sequence.

**Methods:**
- `plot_evolution(parameter, xlabel='Age', ...)`: Plot parameter vs age
- `plot_hr_diagram()`: Hertzsprung-Russell diagram (log L vs T_eff)
- `compare_profiles(column, model_indices, x_column='r')`: Overlay profiles
- `plot_he_profile_match(target, snapshot, n_points=200)`: He profile comparison

---

## File Format

### Model File Format

```
PROFILE
  10   1.0000E+00   0.0000E+00   4.5000E+04   0.0000E+00   5.7700E+03
<mesh point 1: 5 lines of 10-11 values each>
<mesh point 2: 5 lines of 10-11 values each>
...
```

**Header line 2:**
- Column 1: mesh_number
- Column 4: log_g
- Column 6: T_eff

### seq.txt Format

```
SEQ
  STELLAR EVOLUTION SEQUENCE
  INITIAL MASS: 5.0 MSUN
  METALLICITY: Z=0.02
  DATE: 2024-01-01
  Mod   Teff    Log_g   Rayon     Age       Lum       ...
    1   5100.0     4.40    1.10    1.00e+06    101.0   ...
    2   5200.0     4.30    1.20    2.00e+06    102.0   ...
```

---

## Logging

StelumPy uses Python's logging module. Configure logging level:

```python
import logging

# Set StelumPy logging level
logging.getLogger('StelumPy').setLevel(logging.DEBUG)

# Or configure root logger
logging.basicConfig(level=logging.DEBUG)
```

---

## Testing

Run tests with pytest:

```bash
cd StelumPy
pytest tests/ -v
```

---

## License

MIT License
