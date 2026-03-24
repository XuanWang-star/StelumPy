# `find_best_model` Usage Guide

Finds the best-matching stellar model in an evolution sequence by core He abundance (X_He), with optional copying of the matched file to a target directory.

---

## Table of Contents

- [Installation](#installation)
- [Two Usage Modes](#two-usage-modes)
- [Python Function](#python-function)
- [Command-Line Interface](#command-line-interface)
- [Parameter Reference](#parameter-reference)
- [Return Value](#return-value)
- [Example Scenarios](#example-scenarios)
- [Troubleshooting](#troubleshooting)

---

## Installation

Make sure StelumPy is installed in editable mode:

```bash
cd /path/to/StelumPy
pip install -e .
```

After installation, the package can be imported in Python and the `find-best-model` command becomes available in the terminal.

---

## Two Usage Modes

| Mode | When to use |
|------|-------------|
| Python function | Interactive analysis in VS Code / Jupyter; result needs further processing |
| Command-line script | Batch processing, automation pipelines, quick inspection |

---

## Python Function

### Import

```python
from StelumPy.cli import find_best_model
```

### Basic usage

```python
result = find_best_model(
    seq_path  = "/path/to/seq",
    target_he = 0.2345,
)
```

Default behaviour: estimates core He using the innermost 1 mesh point; copies the matched file into `seq_path` itself.

### Disable file copying

```python
result = find_best_model(
    seq_path  = "/path/to/seq",
    target_he = 0.2345,
    copy_to   = None,       # explicitly pass None to disable copying
)
```

### Copy to a custom directory and include seq.txt

```python
result = find_best_model(
    seq_path  = "/path/to/seq",
    target_he = 0.2345,
    copy_to   = "/data/results/run_01",
    copy_seq  = True,
)
```

### Use the returned Model object for further analysis

```python
from StelumPy import Sequence, SequenceAnalyzer, Model

result = find_best_model("/path/to/seq", 0.2345, copy_to=None)

model = result["model"]          # Model object
model.summary()                  # print model header info

# Example: He profile matching against a reference model
target_model = Model("/path/to/target_model.txt")
analyser = SequenceAnalyzer(Sequence("/path/to/seq"))
metrics = analyser.he_profile_match(target_model, model)
print(metrics)
```

---

## Command-Line Interface

After installation, use `find-best-model` directly in the terminal:

```bash
find-best-model SEQ_PATH TARGET_HE [options]
```

If the package is not installed, the script can still be run directly:

```bash
python StelumPy/cli.py SEQ_PATH TARGET_HE [options]
```

### Basic usage

```bash
# Find the best-matching model; copy result into seq_path
find-best-model /path/to/seq 0.2345
```

### Common examples

```bash
# Use 5 innermost mesh points to average core He
find-best-model /path/to/seq 0.2345 --n_points 5

# Copy matched file to a custom directory (created automatically if absent)
find-best-model /path/to/seq 0.2345 --copy_to ./results/run_01

# Also copy seq.txt alongside the model file
find-best-model /path/to/seq 0.2345 --copy_to ./results --copy_seq

# Do not copy any files; only print the match result
find-best-model /path/to/seq 0.2345 --no_copy

# Silent mode — no output (useful inside shell scripts)
find-best-model /path/to/seq 0.2345 --quiet
```

### Typical output

```
Loading sequence from: /path/to/seq
============================================================
  Best-Match Model  —  core He search
============================================================
  Target X_He     : 0.234500  (n_points=1)
  Matched X_He    : 0.234487
  Δ (match−target): -1.30e-05
  Sequence index  : 312
  Model file      : /path/to/seq/5mext/md00312.txt
  T_eff           : 28341.2 K
  log g           : 5.3812
  Age             : 1.2043e+08 yr
  Copied model to : /path/to/seq/md00312.txt
============================================================
```

### Using in shell scripts

The command returns an exit code based on match quality, which can be used for conditional logic in automation:

```bash
find-best-model /path/to/seq 0.2345 --copy_to ./results
if [ $? -ne 0 ]; then
    echo "WARNING: large residual — please check the target He value"
fi
```

| Exit code | Meaning |
|-----------|---------|
| `0` | Success; \|Δ\| ≤ 0.05 |
| `1` | Large residual \|Δ\| > 0.05; result is still written but should be inspected |

---

## Parameter Reference

### Function parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_path` | `str / Path` | required | Path to the `seq/` directory; must contain `5mext/` and `seq.txt` |
| `target_he` | `float` | required | Target core He abundance (X_He fraction, 0–1) |
| `n_points` | `int` | `1` | Number of innermost mesh points used to estimate core He |
| `copy_to` | `str / Path / None` | `seq_path` (the `seq/` directory itself) | Destination directory for the copied file; pass `None` to disable |
| `copy_seq` | `bool` | `False` | Whether to also copy `seq.txt` |
| `verbose` | `bool` | `True` | Whether to print the summary table |

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `SEQ_PATH` | required | Path to the `seq/` directory |
| `TARGET_HE` | required | Target core He abundance |
| `--n_points N` | `1` | Number of innermost mesh points |
| `--copy_to DIR` | `seq_path` (the `seq/` directory itself) | Destination directory for copying |
| `--no_copy` | — | Disable file copying entirely |
| `--copy_seq` | — | Also copy `seq.txt` |
| `--quiet` | — | Suppress all stdout output |

---

## Return Value

`find_best_model()` returns a `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `index` | `int` | 0-based index of the matched model in the sequence |
| `model` | `Model` | The full `Model` object; can be passed directly to other analysis methods |
| `he_core` | `float` | Actual core He value of the matched model |
| `delta` | `float` | Signed residual: `he_core − target_he` |
| `age` | `float` | Stellar age in years from `seq.txt`; `nan` if unavailable |
| `T_eff` | `float` | Effective temperature (K) |
| `log_g` | `float` | Surface gravity log g |
| `copied_to` | `Path / None` | Path of the copied file; `None` if copying was disabled |

---

## Example Scenarios

### Scenario 1: Quick inspection of a single sequence

```python
result = find_best_model("/data/seq", 0.15)
print(f"Matched model : {result['model'].file_path.name}")
print(f"Residual      : {result['delta']:.2e}")
```

### Scenario 2: Batch processing over multiple sequences

```python
from pathlib import Path
from StelumPy.cli import find_best_model

seq_dirs  = sorted(Path("/data").glob("*/seq"))
target_he = 0.2345

for seq_path in seq_dirs:
    result = find_best_model(seq_path, target_he, verbose=False)
    print(f"{seq_path.parent.name}  →  Δ={result['delta']:+.2e}  "
          f"T_eff={result['T_eff']:.0f} K  log_g={result['log_g']:.3f}")
```

### Scenario 3: Batch archiving from the command line

```bash
for seq_dir in /data/*/seq; do
    run=$(basename $(dirname $seq_dir))
    find-best-model "$seq_dir" 0.2345 \
        --copy_to "/archive/$run" \
        --copy_seq \
        --quiet
done
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: StelumPy is not installed` | Package not installed | `pip install -e /path/to/StelumPy` |
| `FileNotFoundError: sq directory not found` | Wrong path | Check that `seq_path` points to the directory containing `5mext/` |
| `FileNotFoundError: seq.txt` | Missing evolution table | Confirm that `seq/seq.txt` exists |
| `ValueError: No models loaded` | Empty `5mext/` directory | Check that model files are present |
| Exit code `1` (CLI) | Residual \|Δ\| > 0.05 | Check whether `target_he` falls within the sequence's He range |
