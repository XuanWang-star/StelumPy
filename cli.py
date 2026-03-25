"""
find_best_model.py
==================
Utility for finding the best-matching stellar model in a StelumPy Sequence
by core He abundance.

Two usage modes
---------------
1. **Import in VS Code / Jupyter**::

       from find_best_model import find_best_model

       result = find_best_model(
           seq_path  = "/path/to/seq",
           target_he = 0.2345,
           n_points  = 1,
           copy_to   = "/path/to/output_dir",   # optional
       )

2. **Command line**::

       python find_best_model.py /path/to/seq 0.2345 \\
           --n_points 1 \\
           --copy_to  /path/to/output_dir

       # Copy both the model file AND seq.txt:
       python find_best_model.py /path/to/seq 0.2345 --copy_to ./results --copy_seq

       # Suppress printed output:
       python find_best_model.py /path/to/seq 0.2345 --quiet
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

# Sentinel value for default copy_to behavior
_COPY_TO_DEFAULT = object()


def find_best_model(
    seq_path: str | Path,
    target_he: float,
    n_points: int = 1,
    copy_to: str | Path | None | object = _COPY_TO_DEFAULT,  # Use sentinel
    copy_seq: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load a :class:`StelumPy.Sequence` and return the model whose core He
    abundance is closest to *target_he*.

    Parameters
    ----------
    seq_path : str or Path
        Path to the ``seq/`` directory (must contain ``5mext/`` and
        ``seq.txt``).
    target_he : float
        Target core He abundance (X_He fraction, 0–1).
    n_points : int, optional
        Number of innermost mesh points used to estimate the core He value.
        Default is ``1`` (centre only).
    copy_to : str or Path or None, optional
        Directory to copy the matched model file into.
        Defaults to *seq_path* itself (i.e. the ``seq/`` directory).
        Pass ``None`` explicitly to disable copying.
    copy_seq : bool, optional
        When *copy_to* is set, also copy ``seq.txt`` from the sequence
        directory.  Default ``False``.
    verbose : bool, optional
        Print a summary table to stdout.  Default ``True``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``index``
            0-based position of the matched model in the sequence.
        ``model``
            The :class:`StelumPy.io.model.Model` object.
        ``he_core``
            Actual core He value of the matched model.
        ``delta``
            Signed difference  ``he_core − target_he``.
        ``age``
            Stellar age (from ``seq.txt``), or ``nan`` if unavailable.
        ``T_eff``
            Effective temperature [K].
        ``log_g``
            Surface gravity log g.
        ``copied_to``
            Path of the copied file, or ``None`` if *copy_to* was not set.
    """
    # ---- lazy import so the module can be loaded without StelumPy ----
    try:
        from StelumPy import Sequence, SequenceAnalyzer
    except ImportError as exc:
        raise ImportError(
            "StelumPy is not installed.  "
            "Run:  pip install -e /path/to/StelumPy"
        ) from exc

    seq_path = Path(seq_path)

    # Resolve default copy destination: the seq_path directory itself
    if copy_to is _COPY_TO_DEFAULT:
        copy_to = seq_path
    elif copy_to is None:
        # Explicitly disable copying
        pass

    if verbose:
        logger.info("Loading sequence from: %s", seq_path)

    seq      = Sequence(seq_path, verbose=verbose)
    analyser = SequenceAnalyzer(seq)

    result = analyser.find_model_by_he_core(target_he, n_points=n_points)

    # ---- optional file copy ----
    copied_to: Path | None = None
    if copy_to is not None:
        dest_dir = Path(copy_to)
        dest_dir.mkdir(parents=True, exist_ok=True)

        model_src  = Path(result["model"].file_path)
        copied_to  = dest_dir / model_src.name
        shutil.copy2(model_src, copied_to)

        if copy_seq:
            seq_src = seq_path / "seq.txt"
            if seq_src.exists():
                shutil.copy2(seq_src, dest_dir / "seq.txt")
            else:
                logger.warning("  ⚠ seq.txt not found at %s", seq_src)

    result["copied_to"] = copied_to

    # ---- summary ----
    if verbose:
        _print_summary(result, target_he, n_points, copied_to, copy_seq, seq_path)

    return result


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_summary(
    result: dict,
    target_he: float,
    n_points: int,
    copied_to: Path | None,
    copy_seq: bool,
    seq_path: Path,
) -> None:
    """Print a summary of the matching result to stdout."""
    sep = "=" * 60
    lines = [
        sep,
        "  Best-Match Model  —  core He search",
        sep,
        f"  Target X_He    : {target_he:.6f}  (n_points={n_points})",
        f"  Matched X_He   : {result['he_core']:.6f}",
        f"  Δ (match−target): {result['delta']:+.2e}",
        f"  Sequence index  : {result['index']}",
        f"  Model file      : {result['model'].file_path}",
        f"  T_eff           : {result['T_eff']:.1f} K",
        f"  log g           : {result['log_g']:.4f}",
    ]
    if result['age'] == result['age']:  # not nan
        lines.append(f"  Age             : {result['age']:.4e} yr")
    if copied_to is not None:
        lines.append(f"  Copied model to : {copied_to}")
        if copy_seq:
            lines.append(f"  Copied seq.txt  : {copied_to.parent / 'seq.txt'}")
    lines.append(sep)
    
    logger.info("\n".join(lines))
    # Also print for backward compatibility
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find_best_model",
        description=(
            "Find the stellar model in a StelumPy sequence whose core He "
            "abundance is closest to TARGET_HE, and optionally copy the "
            "matched file to an output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "seq_path",
        metavar="SEQ_PATH",
        help="Path to the seq/ directory (must contain 5mext/ and seq.txt).",
    )
    p.add_argument(
        "target_he",
        metavar="TARGET_HE",
        type=float,
        help="Target core He abundance (X_He fraction, 0–1).",
    )
    p.add_argument(
        "--n_points",
        type=int,
        default=1,
        metavar="N",
        help="Number of innermost mesh points used to estimate core He.",
    )
    p.add_argument(
        "--copy_to",
        default=None,          # None means "use default (parent of seq_path)"
        metavar="DIR",
        help=(
            "Copy the matched model file to this directory. "
            "Defaults to the parent directory of SEQ_PATH. "
            "Use --no_copy to disable."
        ),
    )
    p.add_argument(
        "--no_copy",
        action="store_true",
        help="Disable copying entirely (overrides --copy_to).",
    )
    p.add_argument(
        "--copy_seq",
        action="store_true",
        help="Also copy seq.txt when --copy_to is set.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all stdout output.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    result = find_best_model(
        seq_path  = args.seq_path,
        target_he = args.target_he,
        n_points  = args.n_points,
        copy_to   = _COPY_TO_DEFAULT if args.no_copy else (seq_path if args.copy_to is None else args.copy_to),
        copy_seq  = args.copy_seq,
        verbose   = not args.quiet,
    )

    # Return a non-zero exit code if the match is poor (|Δ| > 0.05)
    if abs(result["delta"]) > 0.05:
        logger.warning(
            "WARNING: large residual |Δ| = %.4f (> 0.05). Check your target value.",
            abs(result["delta"]),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
