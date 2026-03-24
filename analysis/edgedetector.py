"""
StelumPy.analysis.edgedetector
--------------------------------
EdgeDetector — finds the rising point of X_He in a stellar model profile,
i.e. where the helium abundance starts to increase from the surface inward
(as seen on the -log q axis, core on the right).

Algorithm
---------
1. Extract log_q and X_He from the model DataFrame.
2. Reverse the arrays so the x-axis runs from surface (low -log q)
   toward the core (high -log q), matching the visual convention.
3. Apply a moving-average to suppress mesh-level noise.
4. Find the first point where the smoothed profile exceeds
   (baseline + noise_tolerance).
5. Refine with linear interpolation on the raw data near the crossing.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EdgeDetector:
    """
    Find the rising point of X_He with respect to -log q in a Model profile.

    Parameters
    ----------
    model : Model-like
        Any object with a ``df`` attribute that is a :class:`pandas.DataFrame`
        containing columns ``'log_q'`` and ``'X_He'``.
    """

    def __init__(self, model) -> None:
        if not hasattr(model, "df") or model.df is None:
            raise ValueError("model must have a non-None 'df' DataFrame attribute.")
        self.model = model

    # ------------------------------------------------------------------

    def detect_ascent_point(
        self,
        baseline_value:  float = 0.0,
        noise_tolerance: float = 0.05,
        window_size:     int   = 5,
        search_range:    Optional[Tuple[float, float]] = None,
    ) -> Optional[float]:
        """
        Find the -log q position where X_He first rises above
        ``baseline_value + noise_tolerance``.

        Parameters
        ----------
        baseline_value : float
            Reference X_He level to measure the rise from.
            Default 0.0 (detects any significant rise from near-zero).
        noise_tolerance : float
            The X_He increment above *baseline_value* that counts as a
            genuine rise.  Default 0.05.
        window_size : int
            Moving-average window applied to X_He before detection.
            Larger values suppress more noise but reduce position accuracy.
            Default 5.
        search_range : (float, float) or None
            Restrict the search to a sub-range of -log q values
            (start, end).  ``None`` means search the full profile.

        Returns
        -------
        float or None
            The -log q value of the interpolated crossing point,
            or ``None`` if no crossing is found.
        """
        df = self.model.df

        # ---- 1. Validate columns ----------------------------------------
        for col in ("log_q", "X_He"):
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in model.df. "
                    f"Available columns: {list(df.columns)}"
                )

        # ---- 2. Build -log_q axis (surface left → core right) -----------
        neg_log_q = -df["log_q"].values.astype(float)
        x_he      = df["X_He"].values.astype(float)

        # Sort by -log_q ascending (surface → core)
        order     = np.argsort(neg_log_q)
        neg_log_q = neg_log_q[order]
        x_he      = x_he[order]

        # ---- 3. Optional range restriction --------------------------------
        if search_range is not None:
            lo, hi = search_range
            mask      = (neg_log_q >= lo) & (neg_log_q <= hi)
            neg_log_q = neg_log_q[mask]
            x_he      = x_he[mask]

        n = len(x_he)
        if n < window_size:
            logger.warning(
                "Data length (%d) < window_size (%d); skipping smoothing.",
                n, window_size,
            )
            smoothed  = x_he
            offset    = 0
        else:
            # ---- 4. Moving-average smoothing ------------------------------
            kernel   = np.ones(window_size) / window_size
            smoothed = np.convolve(x_he, kernel, mode="valid")
            # smoothed[i] corresponds to raw index i + window_size//2
            offset = window_size // 2

        # ---- 5. Find first smoothed crossing above threshold --------------
        threshold = baseline_value + noise_tolerance
        above     = np.where(smoothed > threshold)[0]

        if len(above) == 0:
            logger.debug("No crossing above threshold %.4f found.", threshold)
            return None

        # The smoothed index maps back to the raw array via + offset
        first_smooth_idx = above[0]
        first_raw_idx    = min(first_smooth_idx + offset, n - 1)

        # ---- 6. Linear interpolation on raw data near the crossing --------
        # Search a small window around first_raw_idx for the actual crossing
        search_start = max(0, first_raw_idx - window_size)
        search_end   = min(n - 2, first_raw_idx + 1)  # need i+1 to exist

        for i in range(search_start, search_end + 1):
            y0, y1 = x_he[i], x_he[i + 1]
            if y0 <= threshold < y1:
                t0, t1   = neg_log_q[i], neg_log_q[i + 1]
                fraction = (threshold - y0) / (y1 - y0)
                crossing = t0 + fraction * (t1 - t0)
                return float(crossing)

        # ---- 7. Fallback: return the raw position of first_raw_idx --------
        logger.debug(
            "Precise crossing not found in raw data; returning smoothed position."
        )
        return float(neg_log_q[first_raw_idx])

    # ------------------------------------------------------------------

    def detect_all_ascents(
        self,
        baseline_value:  float = 0.0,
        noise_tolerance: float = 0.05,
        window_size:     int   = 5,
        min_gap:         float = 0.5,
    ) -> list[float]:
        """
        Find *all* rising crossings of the threshold, not just the first.

        Useful when the X_He profile has multiple step-like features
        (e.g. partially mixed layers).

        Parameters
        ----------
        min_gap : float
            Minimum separation (in -log q) between two reported crossings.
            Prevents reporting the same feature twice.

        Returns
        -------
        list of float
            -log q positions of all detected crossings, sorted ascending.
        """
        df        = self.model.df
        neg_log_q = -df["log_q"].values.astype(float)
        x_he      = df["X_He"].values.astype(float)
        order     = np.argsort(neg_log_q)
        neg_log_q = neg_log_q[order]
        x_he      = x_he[order]

        threshold  = baseline_value + noise_tolerance
        crossings: list[float] = []
        last_pos   = -np.inf

        for i in range(len(x_he) - 1):
            if x_he[i] <= threshold < x_he[i + 1]:
                t0, t1   = neg_log_q[i], neg_log_q[i + 1]
                fraction = (threshold - x_he[i]) / (x_he[i + 1] - x_he[i])
                pos      = t0 + fraction * (t1 - t0)
                if pos - last_pos >= min_gap:
                    crossings.append(float(pos))
                    last_pos = pos

        return crossings


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    class _DummyModel:
        """Mock model with a synthetic X_He profile."""
        def __init__(self):
            # -log_q from 0 (surface) to 14 (core)
            neg_log_q = np.linspace(0, 14, 300)
            log_q     = -neg_log_q

            # X_He: ~0 at surface, step rise around -log_q = 3,
            # then plateau, then drops back to 0 toward deep core
            x_he = (
                0.95 / (1 + np.exp(-4 * (neg_log_q - 3.0)))
                * (1 - 0.5 / (1 + np.exp(-6 * (neg_log_q - 11.0))))
            )
            x_he += np.random.default_rng(42).normal(0, 0.005, len(x_he))
            x_he  = np.clip(x_he, 0, 1)

            self.df = pd.DataFrame({"log_q": log_q, "X_He": x_he})

    model    = _DummyModel()
    detector = EdgeDetector(model)

    pos = detector.detect_ascent_point(baseline_value=0.0, noise_tolerance=0.05)
    print(f"First ascent at -log q = {pos:.4f}" if pos is not None
          else "No ascent detected")

    all_pos = detector.detect_all_ascents(baseline_value=0.0, noise_tolerance=0.05)
    print(f"All ascents: {[f'{p:.4f}' for p in all_pos]}")
