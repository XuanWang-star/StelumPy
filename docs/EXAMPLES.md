# StelumPy Examples

This document provides detailed examples for common StelumPy workflows.

---

## Table of Contents

1. [Loading Data](#loading-data)
2. [Finding Best Matches](#finding-best-matches)
3. [Profile Analysis](#profile-analysis)
4. [Visualization](#visualization)
5. [Exporting Data](#exporting-data)

---

## Loading Data

### Single Model

```python
from StelumPy import Model

# Load a model
model = Model("path/to/model/md001.txt")

# Inspect
model.summary()

# Access header info
print(f"Effective temperature: {model.T_eff:.1f} K")
print(f"Surface gravity: {model.log_g:.2f}")
print(f"Mesh points: {model.mesh_number}")

# Access profile data
print(model.df.columns.tolist())  # All available columns

# Extract specific profiles
log_q = model.df['log_q']    # Mass coordinate
X_He = model.df['X_He']      # Helium abundance
X_H = model.df['X_H']        # Hydrogen abundance
X_C = model.df['X_C']        # Carbon abundance
X_O = model.df['X_O']        # Oxygen abundance

# Calculate core helium (average over innermost 10 mesh points)
he_core = model.he_core_he(n_points=10)
print(f"Core He abundance: {he_core:.6f}")
```

### Evolution Sequence

```python
from StelumPy import Sequence

# Load entire sequence
seq = Sequence("path/to/sq/", verbose=True)

# Basic info
seq.summary()
print(f"Number of models: {len(seq)}")

# Access models
first = seq[0]
middle = seq[len(seq) // 2]
last = seq[-1]

# Get specific model by index
model_10 = seq.get_model(10)

# Get corresponding age
age_10 = seq.get_age(10)
print(f"Age at model 10: {age_10:.4e} years")

# Access evolution data directly
ages = seq.age_sequence
print(f"Age range: {ages.min():.2e} to {ages.max():.2e} years")
```

### Loading with Options

```python
# Load only first 10 models (faster for testing)
seq_subset = Sequence("path/to/sq/", max_models=10, verbose=True)

# Suppress verbose output
seq_quiet = Sequence("path/to/sq/", verbose=False)
```

---

## Finding Best Matches

### By Core Helium Abundance

```python
from StelumPy import Sequence, SequenceAnalyzer, Model

# Setup
seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Target model (e.g., from observations or different code)
target = Model("static_model.txt")
target_he = target.he_core_he(n_points=10)

# Find best match
result = analyzer.find_model_by_he_core(target_he, n_points=10)

print(f"Best match found:")
print(f"  Index: {result['index']}")
print(f"  Core He: {result['he_core']:.6f}")
print(f"  Delta: {result['delta']:+.6f}")
print(f"  Age: {result['age']:.4e} yr")
print(f"  T_eff: {result['T_eff']:.1f} K")
print(f"  log g: {result['log_g']:.4f}")

# Get top 5 closest matches
top_5 = analyzer.find_models_by_he_core(target_he, n_models=5, n_points=10)

print("\nTop 5 matches:")
for i, match in enumerate(top_5):
    print(f"  {i+1}. Index {match['index']}: He={match['he_core']:.6f}, "
          f"Δ={match['delta']:+.2e}, Age={match['age']:.2e}")
```

### By T_eff and log_g

```python
from StelumPy import Sequence, SequenceAnalyzer

seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Find closest model to given T_eff and log_g
result = analyzer.find_closest_model(
    T_eff_target=6000,
    log_g_target=4.0,
    weight_T_eff=1.0,  # Weight for T_eff in distance calculation
    weight_log_g=1.0,  # Weight for log_g
)

print(f"Closest model:")
print(f"  T_eff: {result['T_eff']:.1f} K (Δ={result['dT_eff']:+.1f})")
print(f"  log g: {result['log_g']:.4f} (Δ={result['dlog_g']:+.4f})")
print(f"  Distance: {result['distance']:.4f}")

# Find multiple close models
close_models = analyzer.find_closest_models_around(
    T_eff_target=6000,
    log_g_target=4.0,
    n_models=5,
)
```

---

## Profile Analysis

### Compare He Profiles

```python
from StelumPy import Sequence, SequenceAnalyzer, Model

seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)
target = Model("static_model.txt")

# Get best match
result = analyzer.find_model_by_he_core(target.he_core_he())
snapshot = result['model']

# Quantitative comparison
metrics = analyzer.he_profile_match(target, snapshot, n_points=200)

print("Profile match metrics:")
print(f"  RMSE: {metrics['rmse']:.6f}")
print(f"  MAE: {metrics['mae']:.6f}")
print(f"  R²: {metrics['r2']:.6f}")
print(f"  Max deviation: {metrics['max_dev']:.6f}")
```

### Evolution of Core Properties

```python
from StelumPy import Sequence, SequenceAnalyzer

seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Get evolution of various parameters
T_eff_evol = analyzer.get_evolution_data('T_eff')
log_g_evol = analyzer.get_evolution_data('log_g')
he_core_evol = analyzer.get_he_core_evolution(n_points=5)

# Create custom evolution DataFrame
df = analyzer.create_evolution_dataframe(['T_eff', 'log_g', 'X_He'])
print(df.head())
```

### Profile at Fixed Mesh Point

```python
from StelumPy import Sequence, SequenceAnalyzer

seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Get central X_He evolution
center_he = analyzer.get_profile_evolution('X_He', mesh_point='center')

# Get surface X_He evolution
surface_he = analyzer.get_profile_evolution('X_He', mesh_point='surface')

# Get X_He at specific mesh point index
he_at_10 = analyzer.get_profile_evolution('X_He', mesh_point=10)
```

---

## Visualization

### Evolution Plots

```python
from StelumPy import Sequence, SequencePlotter
import matplotlib.pyplot as plt

seq = Sequence("path/to/sq/")
plotter = SequencePlotter(seq)

# Plot T_eff evolution
fig, ax = plotter.plot_evolution('T_eff')
plt.show()

# Plot with log time axis
fig, ax = plotter.plot_evolution('T_eff', use_log_time=True)
plt.show()

# Custom labels
fig, ax = plotter.plot_evolution(
    'log_g',
    xlabel='Time [years]',
    ylabel='log g [cm/s²]',
    title='Surface Gravity Evolution',
    color='red',
    marker='o',
)
plt.show()
```

### HR Diagram

```python
from StelumPy import Sequence, SequencePlotter

seq = Sequence("path/to/sq/")
plotter = SequencePlotter(seq)

# Basic HR diagram
fig, ax = plotter.plot_hr_diagram()
plt.show()

# With custom styling
fig, ax = plotter.plot_hr_diagram(
    color='blue',
    marker='o',
    linewidth=2,
    markersize=4,
)
plt.show()
```

### Profile Comparison

```python
from StelumPy import Sequence, SequencePlotter

seq = Sequence("path/to/sq/")
plotter = SequencePlotter(seq)

# Compare X_He profiles at different ages
fig, ax = plotter.compare_profiles(
    'X_He',
    model_indices=[0, 5, 10, 15, 20],
    x_column='log_q',  # or 'r' for radius
)
plt.show()

# Compare temperature profiles
fig, ax = plotter.compare_profiles(
    'T',
    model_indices=[0, 10, 20],
    x_column='r',
)
plt.show()
```

### He Profile Match Plot

```python
from StelumPy import Sequence, SequencePlotter, Model

seq = Sequence("path/to/sq/")
plotter = SequencePlotter(seq)
target = Model("static_model.txt")

# Find best match
from StelumPy import SequenceAnalyzer
analyzer = SequenceAnalyzer(seq)
result = analyzer.find_model_by_he_core(target.he_core_he())

# Plot comparison with metrics
(fig, ax), metrics = plotter.plot_he_profile_match(
    target,
    result['model'],
    n_points=200,
)
plt.show()

print(f"Match quality: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
```

### Using Existing Axes

```python
import matplotlib.pyplot as plt
from StelumPy import Sequence, SequencePlotter

seq = Sequence("path/to/sq/")
plotter = SequencePlotter(seq)

# Create multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# HR diagram
plotter.plot_hr_diagram(ax=axes[0, 0])

# T_eff evolution
plotter.plot_evolution('T_eff', ax=axes[0, 1])

# log_g evolution
plotter.plot_evolution('log_g', ax=axes[1, 0])

# Profile comparison
plotter.compare_profiles('X_He', model_indices=[0, 5, 10], ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

---

## Exporting Data

### Export Evolution to CSV

```python
from StelumPy import Sequence

seq = Sequence("path/to/sq/")

# Export specific parameters
seq.export_evolution_csv(
    "evolution.csv",
    parameters=['T_eff', 'log_g', 'X_He', 'X_C', 'X_O'],
)
```

### Manual Export with pandas

```python
from StelumPy import Sequence, SequenceAnalyzer

seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Create DataFrame
df = analyzer.create_evolution_dataframe(['T_eff', 'log_g', 'he_core'])

# Add additional columns
df['log_Teff'] = np.log10(df['T_eff'])

# Export
df.to_csv("custom_evolution.csv", index=False)
df.to_excel("custom_evolution.xlsx", index=False)  # Requires openpyxl
```

### Extract Profile Data

```python
from StelumPy import Sequence

seq = Sequence("path/to/sq/")

# Extract profile from specific model
model = seq[10]
profile_df = model.df

# Save to CSV
profile_df.to_csv(f"model_{10}_profile.csv", index=False)

# Extract all profiles to separate files
for i, model in enumerate(seq.models):
    model.df.to_csv(f"profile_{i:03d}.csv", index=False)
```

---

## Advanced Workflows

### Batch Processing Multiple Sequences

```python
from pathlib import Path
from StelumPy import Sequence, SequenceAnalyzer

sequences_dir = Path("path/to/sequences/")
results = []

for seq_dir in sequences_dir.glob("seq_*"):
    if not seq_dir.is_dir():
        continue
    
    seq = Sequence(seq_dir, verbose=False)
    analyzer = SequenceAnalyzer(seq)
    
    # Find best match for target He = 0.3
    result = analyzer.find_model_by_he_core(0.3)
    
    results.append({
        'sequence': seq_dir.name,
        'best_index': result['index'],
        'best_he': result['he_core'],
        'age': result['age'],
        'T_eff': result['T_eff'],
    })

# Convert to DataFrame
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)
```

### Custom Matching Criteria

```python
from StelumPy import Sequence, SequenceAnalyzer
import numpy as np

seq = Sequence("path/to/sq/")
analyzer = SequenceAnalyzer(seq)

# Get all T_eff and log_g values
T_eff_all = analyzer.get_evolution_data('T_eff')
log_g_all = analyzer.get_evolution_data('log_g')
ages = seq.age_sequence

# Custom selection: T_eff in range AND log_g in range
mask = (T_eff_all > 5500) & (T_eff_all < 6500) & \
       (log_g_all > 3.8) & (log_g_all < 4.2)

selected_indices = np.where(mask)[0]
print(f"Found {len(selected_indices)} matching models")

for idx in selected_indices:
    print(f"  Index {idx}: T_eff={T_eff_all[idx]:.0f} K, "
          f"log_g={log_g_all[idx]:.2f}, Age={ages[idx]:.2e}")
```
