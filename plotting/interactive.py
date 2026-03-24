"""
StelumPy.plotting.interactive
------------------------------
Interactive explorers using a native Tk window with Combobox dropdowns,
embedded matplotlib canvas, and plot/save buttons.

Classes
-------
ModelExplorer     — profile explorer for a single Model
SequenceExplorer  — evolution track + profile snapshot for a Sequence

Usage
-----
    from StelumPy import Model, Sequence
    from StelumPy.plotting.interactive import ModelExplorer, SequenceExplorer

    ModelExplorer(Model("/path/to/model.txt")).show()
    SequenceExplorer(Sequence("/path/to/seq")).show()

Run from the terminal (not the VS Code run button) for full interactivity:

    python your_script.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")                          # ensure Tk backend

# High-quality text rendering for physics notation
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts for better math
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150          # higher DPI for sharper text

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import RectangleSelector

from ..io.model    import Model
from ..io.sequence import Sequence
from ..analysis.matching import SequenceAnalyzer


# ---------------------------------------------------------------------------
# Axis-tick helper: minor ticks, inward direction, no grid
# ---------------------------------------------------------------------------

def _style_ax(ax) -> None:
    """Apply consistent tick style: minor ticks, inward, no grid."""
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in",
                   top=True, right=True,
                   labelsize=9)
    ax.tick_params(which="minor", length=3)
    ax.tick_params(which="major", length=6)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Full parameter catalogue
# Format: (column, combo_label, xlabel, ylabel)
# - combo_label: plain text for Tkinter Combobox (no mathtext)
# - xlabel/ylabel: matplotlib mathtext for axis labels (supports subscripts/Greek)
# ---------------------------------------------------------------------------

PROFILE_PARAMS: list[tuple[str, str, str, str]] = [
    # ---- mass coordinate ----
    ("log_q",           "log q  (mass coord)",
                        r"$-\log\,q$",                    r"$-\log\,q$"),
    # ---- structure ----
    ("r",               "r  (radius)",
                        r"$r$",                           r"$r$"),
    ("m_r",             "m(r)  (enclosed mass)",
                        r"$m(r)$",                        r"$m(r)$"),
    ("rho",             "rho  (density)",
                        r"$\rho$",                        r"$\rho$"),
    ("P",               "P  (pressure)",
                        r"$P$",                           r"$P$"),
    ("T",               "T  (temperature)",
                        r"$T$",                           r"$T$"),
    ("L",               "L  (luminosity)",
                        r"$L$",                           r"$L$"),
    ("log_P_gas",       "log P_gas",
                        r"$\log\,P_\mathrm{gas}$",        r"$\log\,P_\mathrm{gas}$"),
    # ---- composition ----
    ("X_H",             "X_H  (hydrogen)",
                        r"$X_\mathrm{H}$",                r"$X_\mathrm{H}$"),
    ("X_He",            "X_He  (helium)",
                        r"$X_\mathrm{He}$",               r"$X_\mathrm{He}$"),
    ("X_C",             "X_C  (carbon)",
                        r"$X_\mathrm{C}$",                r"$X_\mathrm{C}$"),
    ("X_O",             "X_O  (oxygen)",
                        r"$X_\mathrm{O}$",                r"$X_\mathrm{O}$"),
    ("Y",               "Y  (He mass fraction, local)",
                        r"$Y$",                           r"$Y$"),
    # ---- gradients ----
    ("Del",             "nabla  (actual gradient)",
                        r"$\nabla$",                      r"$\nabla$"),
    ("Del_ad",          "nabla_ad",
                        r"$\nabla_\mathrm{ad}$",          r"$\nabla_\mathrm{ad}$"),
    ("Del_rad",         "nabla_rad",
                        r"$\nabla_\mathrm{rad}$",         r"$\nabla_\mathrm{rad}$"),
    ("Del_ad_minus_Del","nabla_ad - nabla",
                        r"$\nabla_\mathrm{ad}-\nabla$",   r"$\nabla_\mathrm{ad}-\nabla$"),
    ("Del_r",           "nabla_r",
                        r"$\nabla_r$",                    r"$\nabla_r$"),
    ("Del_P",           "nabla_P",
                        r"$\nabla_P$",                    r"$\nabla_P$"),
    ("Del_T",           "nabla_T",
                        r"$\nabla_T$",                    r"$\nabla_T$"),
    ("Del_tau",         "nabla_tau",
                        r"$\nabla_\tau$",                 r"$\nabla_\tau$"),
    ("Del_L",           "nabla_L",
                        r"$\nabla_L$",                    r"$\nabla_L$"),
    ("Del_ad_P",        "nabla_ad,P",
                        r"$\nabla_{\mathrm{ad},P}$",      r"$\nabla_{\mathrm{ad},P}$"),
    ("Del_ad_T",        "nabla_ad,T",
                        r"$\nabla_{\mathrm{ad},T}$",      r"$\nabla_{\mathrm{ad},T}$"),
    # ---- opacity & energy ----
    ("kappa",           "kappa  (opacity)",
                        r"$\kappa$",                      r"$\kappa$"),
    ("kappa_rho",       "kappa_rho",
                        r"$\kappa_\rho$",                 r"$\kappa_\rho$"),
    ("kappa_T",         "kappa_T",
                        r"$\kappa_T$",                    r"$\kappa_T$"),
    ("epsilon_N",       "epsilon_N  (nuclear)",
                        r"$\varepsilon_N$",               r"$\varepsilon_N$"),
    ("epsilon_rho",     "epsilon_rho",
                        r"$\varepsilon_\rho$",            r"$\varepsilon_\rho$"),
    ("epsilon_T",       "epsilon_T",
                        r"$\varepsilon_T$",               r"$\varepsilon_T$"),
    # ---- thermodynamics ----
    ("chi_p",           "chi_P",
                        r"$\chi_P$",                      r"$\chi_P$"),
    ("chi_T",           "chi_T",
                        r"$\chi_T$",                      r"$\chi_T$"),
    ("C_p",             "C_P  (specific heat)",
                        r"$C_P$",                         r"$C_P$"),
    ("C_V",             "C_V  (specific heat)",
                        r"$C_V$",                         r"$C_V$"),
    ("Gamma",           "Gamma  (adiabatic index)",
                        r"$\Gamma$",                      r"$\Gamma$"),
    ("U",               "U",
                        r"$U$",                           r"$U$"),
    ("dU_dP",           "dU/dP",
                        r"$\partial U/\partial P$",       r"$\partial U/\partial P$"),
    ("dU_dT",           "dU/dT",
                        r"$\partial U/\partial T$",       r"$\partial U/\partial T$"),
    # ---- degeneracy / radiation ----
    ("eta",             "eta  (degeneracy)",
                        r"$\eta$",                        r"$\eta$"),
    ("eta_r",           "eta_r",
                        r"$\eta_r$",                      r"$\eta_r$"),
    ("eta_e",           "eta_e",
                        r"$\eta_e$",                      r"$\eta_e$"),
    ("tau",             "tau  (optical depth)",
                        r"$\tau$",                        r"$\tau$"),
    ("B",               "B",
                        r"$B$",                           r"$B$"),
    ("Z_moy",           "Z_mean  (mean charge)",
                        r"$\bar{Z}$",                     r"$\bar{Z}$"),
    # ---- velocity / mixing ----
    ("zeta",            "zeta",
                        r"$\zeta$",                       r"$\zeta$"),
    ("w",               "w  (convective velocity)",
                        r"$w$",                           r"$w$"),
    ("wtau",            "w*tau",
                        r"$w\tau$",                       r"$w\tau$"),
]

SEQ_PARAMS: list[tuple[str, str, str, str]] = [
    ("Age",      "Age (yr)",                        r"Age (yr)",                  r"Age (yr)"),
    ("Teff",     "T_eff (K)",                       r"$T_\mathrm{eff}$ (K)",      r"$T_\mathrm{eff}$ (K)"),
    ("Log_g",    "log g",                           r"$\log\,g$",                 r"$\log\,g$"),
    ("Rayon",    "R (R_sun)",                       r"$R$ ($R_\odot$)",           r"$R$ ($R_\odot$)"),
    ("Lum",      "L (L_sun)",                       r"$L$ ($L_\odot$)",           r"$L$ ($L_\odot$)"),
    ("Log_Tc",   "log T_c",                         r"$\log\,T_\mathrm{c}$",      r"$\log\,T_\mathrm{c}$"),
    ("Log_Pc",   "log P_c",                         r"$\log\,P_\mathrm{c}$",      r"$\log\,P_\mathrm{c}$"),
    ("Log_rhoc", "log rho_c",                       r"$\log\,\rho_\mathrm{c}$",   r"$\log\,\rho_\mathrm{c}$"),
    ("Log_He",   "log He",                          r"$\log\,\mathrm{He}$",       r"$\log\,\mathrm{He}$"),
    ("Log_H",    "log H",                           r"$\log\,\mathrm{H}$",        r"$\log\,\mathrm{H}$"),
    ("Log_C",    "log C",                           r"$\log\,\mathrm{C}$",        r"$\log\,\mathrm{C}$"),
    ("Log_O",    "log O",                           r"$\log\,\mathrm{O}$",        r"$\log\,\mathrm{O}$"),
    ("Mod",      "Model number",                    r"Model number",              r"Model number"),
]

_SKIP_COLS = {"n", "flag14", "flag15", "unknown", "unknown1", "unknown2", "unknown3"}


def _col_to_values(df, col: str) -> np.ndarray:
    v = df[col].values.astype(float)
    return -v if col == "log_q" else v


def _filter_params(params: list[tuple], available: set[str]) -> list[tuple]:
    return [p for p in params if p[0] in available and p[0] not in _SKIP_COLS]


def _save_figure(fig: Figure, stem: str) -> None:
    for fmt in ("png", "pdf", "svg"):
        out = Path(f"{stem}.{fmt}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {out.resolve()}")


# ---------------------------------------------------------------------------
# Shared Tk + matplotlib base
# ---------------------------------------------------------------------------

class _BaseExplorer:
    """Common Tk window + embedded matplotlib canvas."""

    _WIN_TITLE = "StelumPy Explorer"
    _FIG_SIZE  = (10.0, 6.0)         # inches for the matplotlib figure
    _FIG_DPI   = 150                 # DPI for sharp text rendering

    def _build_window(self) -> tuple[tk.Tk, Figure, FigureCanvasTkAgg]:
        root = tk.Tk()
        root.title(self._WIN_TITLE)
        root.configure(bg="#f0f0f0")

        # Left frame: controls
        self._ctrl_frame = ttk.Frame(root, padding=10)
        self._ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Right frame: matplotlib canvas with toolbar
        canvas_frame = ttk.Frame(root)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=self._FIG_SIZE, dpi=self._FIG_DPI)
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar with zoom/pan buttons
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        
        # Help label
        ttk.Label(canvas_frame, 
                  text="Tip: Use toolbar icons (🔍 zoom, ✋ pan) or press: z=zoom, p=pan, h=reset, s=save",
                  font=("", 8), foreground="#666").pack(fill=tk.X, pady=2)

        return root, fig, canvas

    @staticmethod
    def _make_combo(parent, label_text: str, values: list[str],
                    default: str) -> ttk.Combobox:
        ttk.Label(parent, text=label_text, font=("", 9, "bold")).pack(
            anchor=tk.W, pady=(8, 1)
        )
        combo = ttk.Combobox(parent, values=values, state="readonly", width=30)
        combo.pack(anchor=tk.W)
        if default in values:
            combo.current(values.index(default))
        else:
            combo.current(0)
        return combo

    @staticmethod
    def _make_button(parent, text: str, command, bg: str = "#4a90d9") -> tk.Button:
        btn = tk.Button(
            parent, text=text, command=command,
            bg=bg, fg="white",
            font=("", 10, "bold"),
            relief=tk.FLAT, padx=12, pady=5,
            activebackground=bg, activeforeground="white",
            cursor="hand2",
        )
        btn.pack(fill=tk.X, pady=(6, 2))
        return btn


# ---------------------------------------------------------------------------
# ModelExplorer
# ---------------------------------------------------------------------------

class ModelExplorer(_BaseExplorer):
    """
    Interactive profile explorer for a single Model.

    Select X and Y from Combobox dropdowns, click Plot.
    log_q is shown as -log q (core on the right).
    Ticks: minor ticks, inward direction, no grid.
    Save exports png / pdf / svg.
    """

    _WIN_TITLE = "ModelExplorer"
    _DEFAULT_X = "log_q"
    _DEFAULT_Y = "X_He"

    def __init__(self, model: Model, save_stem: str | None = None):
        if model.df is None:
            raise ValueError("Model has no profile data loaded.")
        self.model     = model
        self.save_stem = save_stem or Path(model.file_path).stem

        avail         = set(model.df.columns)
        self._params  = _filter_params(PROFILE_PARAMS, avail)
        self._cols    = [p[0] for p in self._params]
        self._clabels = [p[1] for p in self._params]   # combobox display labels
        self._xlabels = {p[0]: p[2] for p in self._params}
        self._ylabels = {p[0]: p[3] for p in self._params}

        self._fig: Figure | None    = None
        self._canvas                = None
        self._ax: plt.Axes | None   = None

    def show(self) -> None:
        root, fig, canvas = self._build_window()
        self._fig    = fig
        self._canvas = canvas
        self._ax     = fig.add_subplot(111)

        # Enable matplotlib keyboard shortcuts: z=zoom, p=pan, h=home, s=save, g=grid
        canvas.mpl_connect("key_press_event", self._on_key_press)

        # --- placeholder text ---
        self._ax.text(0.5, 0.5, "Select axes and click  ▶ Plot",
                      ha="center", va="center",
                      transform=self._ax.transAxes,
                      fontsize=13, color="#b0b0b0")
        self._ax.set_xticks([]); self._ax.set_yticks([])
        canvas.draw()

        f = self._ctrl_frame

        # Title
        ttk.Label(f, text=Path(self.model.file_path).name,
                  font=("", 10, "bold"), foreground="#333").pack(anchor=tk.W)
        ttk.Label(f,
                  text=f"Teff={self.model.T_eff:.0f} K    log g={self.model.log_g:.3f}",
                  font=("", 9), foreground="#666").pack(anchor=tk.W)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self._combo_x = self._make_combo(f, "X axis", self._clabels,
                                          self._clabels[self._cols.index(self._DEFAULT_X)]
                                          if self._DEFAULT_X in self._cols
                                          else self._clabels[0])
        self._combo_y = self._make_combo(f, "Y axis", self._clabels,
                                          self._clabels[self._cols.index(self._DEFAULT_Y)]
                                          if self._DEFAULT_Y in self._cols
                                          else self._clabels[1])

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self._make_button(f, "▶  Plot",  self._on_plot, bg="#4a90d9")
        self._make_button(f, "💾  Save", self._on_save, bg="#5cb85c")

        root.mainloop()

    def _on_plot(self) -> None:
        x_label = self._combo_x.get()
        y_label = self._combo_y.get()
        x_col   = self._cols[self._clabels.index(x_label)]
        y_col   = self._cols[self._clabels.index(y_label)]

        ax = self._ax
        ax.cla()

        x = _col_to_values(self.model.df, x_col)
        y = _col_to_values(self.model.df, y_col)

        ax.plot(x, y, lw=1.8, color="#2166ac")
        ax.set_xlabel(self._xlabels[x_col], fontsize=11)
        ax.set_ylabel(self._ylabels[y_col], fontsize=11)
        fname = Path(self.model.file_path).name
        ax.set_title(
            f"{fname}   "
            f"$T_{{\\rm eff}}={self.model.T_eff:.0f}$ K   "
            f"$\\log g={self.model.log_g:.3f}$",
            fontsize=10,
        )
        _style_ax(ax)
        self._fig.tight_layout()
        self._canvas.draw()

    def _on_save(self) -> None:
        _save_figure(self._fig, self.save_stem)

    def _on_key_press(self, event) -> None:
        """Handle keyboard shortcuts for zoom/pan."""
        # Press 'z' or 'p' to activate zoom/pan mode, then click-and-drag on the figure
        # Press 'h' to reset view, 's' to save
        if event.key in ("z", "Z"):
            print("Zoom mode: click and drag to select an area")
        elif event.key in ("p", "P"):
            print("Pan mode: click and drag to pan")
        elif event.key in ("h", "H"):
            self._ax.autoscale_view()
            self._canvas.draw()
        elif event.key in ("s", "S"):
            self._on_save()


# ---------------------------------------------------------------------------
# SequenceExplorer
# ---------------------------------------------------------------------------

class SequenceExplorer(_BaseExplorer):
    """
    Interactive evolution + profile explorer for a Sequence.

    Top subplot  — evolution track; click on it to pick a snapshot.
    Bottom subplot — profile of the nearest model to the click.

    Four Comboboxes: evolution X/Y and profile X/Y.
    """

    _WIN_TITLE = "SequenceExplorer"
    _FIG_SIZE  = (10.0, 9.0)       # larger figure for two subplots
    _FIG_DPI   = 150               # DPI for sharp text rendering
    _DEFAULT_EV_X = "Age"
    _DEFAULT_EV_Y = "Teff"
    _DEFAULT_PR_X = "log_q"
    _DEFAULT_PR_Y = "X_He"

    def __init__(self, sequence: Sequence, save_stem: str | None = None):
        if sequence.seq_data is None:
            raise ValueError("seq_data not available.")
        if not sequence.models:
            raise ValueError("No models loaded.")

        self.seq       = sequence
        self.analyzer  = SequenceAnalyzer(sequence)
        self.save_stem = save_stem or "sequence_explorer"

        avail_ev         = set(sequence.seq_data.columns)
        self._ev_params  = _filter_params(SEQ_PARAMS, avail_ev)
        self._ev_cols    = [p[0] for p in self._ev_params]
        self._ev_clabels = [p[1] for p in self._ev_params]
        self._ev_xlabels = {p[0]: p[2] for p in self._ev_params}
        self._ev_ylabels = {p[0]: p[3] for p in self._ev_params}

        _m0 = next((m for m in sequence.models if m.df is not None), None)
        if _m0 is None:
            raise ValueError("No model with profile data found.")
        avail_pr         = set(_m0.df.columns)
        self._pr_params  = _filter_params(PROFILE_PARAMS, avail_pr)
        self._pr_cols    = [p[0] for p in self._pr_params]
        self._pr_clabels = [p[1] for p in self._pr_params]
        self._pr_xlabels = {p[0]: p[2] for p in self._pr_params}
        self._pr_ylabels = {p[0]: p[3] for p in self._pr_params}

        self._sel_idx: int | None = None
        self._fig: Figure | None  = None
        self._canvas              = None
        self._ax_ev               = None
        self._ax_pr               = None
        self._cid                 = None

    def show(self) -> None:
        root, fig, canvas = self._build_window()
        self._fig    = fig
        self._canvas = canvas

        self._ax_ev = fig.add_subplot(211)
        self._ax_pr = fig.add_subplot(212)
        for ax in (self._ax_ev, self._ax_pr):
            ax.text(0.5, 0.5, "Click  ▶ Plot  to draw",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#b8b8b8")
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=2.0)
        canvas.draw()

        # Enable keyboard shortcuts for zoom/pan on both subplots
        canvas.mpl_connect("key_press_event", self._on_key_press)

        f = self._ctrl_frame

        ttk.Label(f, text="Sequence Explorer",
                  font=("", 11, "bold"), foreground="#333").pack(anchor=tk.W)
        ttk.Label(f, text=f"{len(self.seq.models)} models loaded",
                  font=("", 9), foreground="#666").pack(anchor=tk.W)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(f, text="── Evolution track ──",
                  font=("", 9, "italic"), foreground="#555").pack(anchor=tk.W)
        self._combo_ev_x = self._make_combo(
            f, "X axis", self._ev_clabels,
            self._ev_clabels[self._ev_cols.index(self._DEFAULT_EV_X)]
            if self._DEFAULT_EV_X in self._ev_cols else self._ev_clabels[0],
        )
        self._combo_ev_y = self._make_combo(
            f, "Y axis", self._ev_clabels,
            self._ev_clabels[self._ev_cols.index(self._DEFAULT_EV_Y)]
            if self._DEFAULT_EV_Y in self._ev_cols else self._ev_clabels[1],
        )

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(f, text="── Profile snapshot ──",
                  font=("", 9, "italic"), foreground="#555").pack(anchor=tk.W)
        self._combo_pr_x = self._make_combo(
            f, "X axis", self._pr_clabels,
            self._pr_clabels[self._pr_cols.index(self._DEFAULT_PR_X)]
            if self._DEFAULT_PR_X in self._pr_cols else self._pr_clabels[0],
        )
        self._combo_pr_y = self._make_combo(
            f, "Y axis", self._pr_clabels,
            self._pr_clabels[self._pr_cols.index(self._DEFAULT_PR_Y)]
            if self._DEFAULT_PR_Y in self._pr_cols else self._pr_clabels[1],
        )

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self._make_button(f, "▶  Plot",  self._on_plot, bg="#4a90d9")
        self._make_button(f, "💾  Save", self._on_save, bg="#5cb85c")
        ttk.Label(f, text="Click on the evolution\ntrack to pick a snapshot",
                  font=("", 8), foreground="#888", justify=tk.LEFT).pack(
                  anchor=tk.W, pady=(8, 0))

        self._cid = canvas.mpl_connect("button_press_event", self._on_click)
        root.mainloop()

    # ------------------------------------------------------------------

    def _on_plot(self) -> None:
        self._draw_evolution()
        self._draw_profile()

    def _on_save(self) -> None:
        _save_figure(self._fig, self.save_stem)

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_ev or event.xdata is None:
            return
        ev_x_col = self._ev_cols[self._ev_clabels.index(self._combo_ev_x.get())]
        x_arr    = self.seq.seq_data[ev_x_col].values.astype(float)
        self._sel_idx = int(np.argmin(np.abs(x_arr - event.xdata)))
        self._draw_evolution()
        self._draw_profile()

    def _on_key_press(self, event) -> None:
        """Handle keyboard shortcuts for zoom/pan."""
        # Press 'z' or 'p' to activate zoom/pan mode, then click-and-drag on the figure
        # Press 'h' to reset view, 's' to save
        if event.key in ("z", "Z"):
            print("Zoom mode: click and drag to select an area")
        elif event.key in ("p", "P"):
            print("Pan mode: click and drag to pan")
        elif event.key in ("h", "H"):
            # Reset both axes
            self._ax_ev.autoscale_view()
            self._ax_pr.autoscale_view()
            self._canvas.draw()
        elif event.key in ("s", "S"):
            self._on_save()

    # ------------------------------------------------------------------

    def _draw_evolution(self) -> None:
        ev_x_col = self._ev_cols[self._ev_clabels.index(self._combo_ev_x.get())]
        ev_y_col = self._ev_cols[self._ev_clabels.index(self._combo_ev_y.get())]

        ax = self._ax_ev
        ax.cla()
        x = self.seq.seq_data[ev_x_col].values.astype(float)
        y = self.seq.seq_data[ev_y_col].values.astype(float)
        ax.plot(x, y, lw=1.8, color="#2166ac")

        if self._sel_idx is not None:
            sx, sy = x[self._sel_idx], y[self._sel_idx]
            ax.axvline(sx, color="#d73027", lw=0.8, ls="--", alpha=0.6)
            ax.scatter([sx], [sy], color="#d73027", zorder=5, s=40,
                       label=f"model #{self._sel_idx}")
            ax.legend(fontsize=8)

        ax.set_xlabel(self._ev_xlabels[ev_x_col], fontsize=10)
        ax.set_ylabel(self._ev_ylabels[ev_y_col], fontsize=10)
        ax.set_title("Evolution track  —  click to inspect a snapshot", fontsize=9)
        _style_ax(ax)
        self._fig.tight_layout(pad=2.0)
        self._canvas.draw()

    def _draw_profile(self) -> None:
        pr_x_col = self._pr_cols[self._pr_clabels.index(self._combo_pr_x.get())]
        pr_y_col = self._pr_cols[self._pr_clabels.index(self._combo_pr_y.get())]

        ax    = self._ax_pr
        ax.cla()
        idx   = self._sel_idx if self._sel_idx is not None else 0
        model = self.seq.models[idx]

        if (model.df is None
                or pr_x_col not in model.df.columns
                or pr_y_col not in model.df.columns):
            ax.text(0.5, 0.5, "Column not available for this model",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaaaaa")
            self._canvas.draw()
            return

        x = _col_to_values(model.df, pr_x_col)
        y = _col_to_values(model.df, pr_y_col)

        age_str = ""
        if self.seq.age_sequence is not None:
            age_str = f"   age $= {self.seq.age_sequence[idx]:.3e}$ yr"

        ax.plot(x, y, lw=1.8, color="#d95f02")
        ax.set_xlabel(self._pr_xlabels[pr_x_col], fontsize=10)
        ax.set_ylabel(self._pr_ylabels[pr_y_col], fontsize=10)
        ax.set_title(
            f"Profile snapshot — model #{idx}   "
            f"$T_{{\\rm eff}}={model.T_eff:.0f}$ K   "
            f"$\\log g={model.log_g:.3f}${age_str}",
            fontsize=9,
        )
        _style_ax(ax)
        self._fig.tight_layout(pad=2.0)
        self._canvas.draw()
