"""
stellarmod.io.sequence
----------------------
Loads an ordered sequence of stellar model files from the sq/ directory
structure and reads the accompanying seq.txt evolution table.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

from .model import Model


class Sequence:
    """
    Load and hold a sequence of stellar model files from the sq/ directory.

    Expected layout::

        sq/
        ├── 5mext/      ← text-format profile files (one per time step)
        ├── seq.txt     ← global evolution table (one row per time step)
        └── idx.txt     ← (optional) index file

    Attributes
    ----------
    sq_directory : Path
    models_directory : Path
    models : list[Model]
    num_models : int
    seq_data : pd.DataFrame | None   Full seq.txt table.
    age_sequence : np.ndarray | None  Age column from seq.txt.
    file_paths : list[Path]
    model_index : list[int | None]
    """

    SEQ_COLUMNS = [
        'Mod', 'Teff', 'Log_g', 'Rayon', 'Age', 'Lum',
        'Log_Tc', 'Log_Pc', 'Log_rhoc', 'M_x_M', 'Log_q_x',
        'Lum_nu', 'Log_H', 'Log_He', 'Log_C', 'Log_O',
    ]

    _MD_PATTERN = re.compile(r'^md\d+\.txt$', re.IGNORECASE)

    def __init__(
        self,
        sq_directory: str | Path,
        max_models: int | None = None,
        verbose: bool = False,
    ):
        self.sq_directory = Path(sq_directory)
        self.models_directory = self.sq_directory / '5mext'
        self.seq_file = self.sq_directory / 'seq.txt'
        self.idx_file = self.sq_directory / 'idx.txt'

        self.max_models = max_models
        self.verbose = verbose

        self.file_paths: list[Path] = []
        self.models: list[Model] = []
        self.num_models: int = 0
        self.model_index: list[int | None] = []
        self.seq_data: pd.DataFrame | None = None
        self.age_sequence: np.ndarray | None = None

        self._validate_structure()
        self._load_seq_data()
        self._load_models()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:
        for path, label in [
            (self.sq_directory,    'sq directory'),
            (self.models_directory,'5mext directory'),
            (self.seq_file,        'seq.txt'),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
            if self.verbose:
                print(f"✓ Found {label}: {path}")

        if self.verbose and self.idx_file.exists():
            print(f"✓ Found idx.txt: {self.idx_file}")

    def _load_seq_data(self) -> None:
        if self.verbose:
            print(f"\nReading evolution sequence from {self.seq_file.name}…")
        try:
            self.seq_data = pd.read_csv(
                self.seq_file,
                sep=r'\s+',
                skiprows=5,
                names=self.SEQ_COLUMNS,
                header=None,
            )
            self.age_sequence = self.seq_data['Age'].values
            if self.verbose:
                print(f"✓ Loaded {len(self.seq_data)} rows from seq.txt")
                print(f"  Age range: {self.age_sequence.min():.2e} – {self.age_sequence.max():.2e}")
        except Exception as exc:
            if self.verbose:
                print(f"⚠ Could not parse seq.txt: {exc}")
                print("  Will use model indices as time sequence.")
            self.seq_data = None
            self.age_sequence = None

    def _get_file_list(self) -> tuple[list[Path], list[int | None]]:
        files = [f for f in self.models_directory.glob('*') if f.is_file()]
        if not files:
            raise FileNotFoundError(f"No model files found in {self.models_directory}")

        # md\d+.txt files sort to the end; everything else sorts by name
        files.sort(key=lambda x: (1 if self._MD_PATTERN.match(x.name) else 0, x.name))

        model_index: list[int | None] = []
        for f in files:
            m = re.search(r'(\d+)', f.name)
            model_index.append(int(m.group(1)) if m else None)

        if self.max_models is not None:
            files = files[: self.max_models]
            model_index = model_index[: self.max_models]

        return files, model_index

    def _load_models(self) -> None:
        self.file_paths, self.model_index = self._get_file_list()
        self.num_models = len(self.file_paths)

        if self.verbose:
            print(f"\nLoading {self.num_models} models from {self.models_directory}…")

        for i, path in enumerate(self.file_paths):
            try:
                self.models.append(Model(path))
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{self.num_models} loaded…")
            except Exception as exc:
                if self.verbose:
                    print(f"  ⚠ Skipping {path.name}: {exc}")

        if self.verbose:
            print(f"✓ Successfully loaded {len(self.models)} models.")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_model(self, index: int) -> Model:
        """Return model at *index* (0-based)."""
        if not 0 <= index < len(self.models):
            raise IndexError(f"Index {index} out of range [0, {len(self.models) - 1}]")
        return self.models[index]

    def get_age(self, index: int) -> float:
        """Return the stellar age at *index*."""
        if self.age_sequence is None:
            raise ValueError("age_sequence not available (seq.txt could not be parsed).")
        if not 0 <= index < len(self.age_sequence):
            raise IndexError(f"Index {index} out of range.")
        return float(self.age_sequence[index])

    def export_evolution_csv(self, output_file: str | Path, parameters: list[str]) -> None:
        """
        Export the evolution of *parameters* to a CSV file.
        Requires a SequenceAnalyzer; kept here for convenience via lazy import.
        """
        from ..analysis.matching import SequenceAnalyzer
        analyzer = SequenceAnalyzer(self)
        df = analyzer.create_evolution_dataframe(parameters)
        df.to_csv(output_file, index=False)
        print(f"Evolution data exported to {output_file}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        sep = '=' * 70
        print(sep)
        print("Sequence Summary")
        print(sep)
        print(f"sq directory:     {self.sq_directory}")
        print(f"Models directory: {self.models_directory}")
        print(f"Models loaded:    {len(self.models)}")
        print(sep)
        if self.models and self.age_sequence is not None:
            print(f"\nAge range:  {self.age_sequence.min():.2e} – {self.age_sequence.max():.2e}")
            print(f"\nFirst model: {self.file_paths[0].name}  "
                    f"T_eff={self.models[0].T_eff:.1f} K  "
                    f"log_g={self.models[0].log_g:.2f}")
            print(f"Last  model: {self.file_paths[-1].name}  "
                    f"T_eff={self.models[-1].T_eff:.1f} K  "
                    f"log_g={self.models[-1].log_g:.2f}")
        print(sep)

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, index: int) -> Model:
        return self.get_model(index)

    def __repr__(self) -> str:
        return (
            f"Sequence('{self.sq_directory}', "
            f"num_models={len(self.models)})"
        )
