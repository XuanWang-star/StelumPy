"""
stellarmod.io.sequence
----------------------
Loads an ordered sequence of stellar model files from the sq/ directory
structure and reads the accompanying seq.txt evolution table.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import SequenceFileError, ValidationError
from .model import Model

logger = logging.getLogger(__name__)


class Sequence:
    """
    Load and hold a sequence of stellar model files from the sq/ directory.

    Expected layout::

        sq/
        ├── 5mext/      ← text-format profile files (one per time step)
        ├── seq.txt     ← global evolution table (one row per time step)
        └── idx.txt     ← (optional) index file

    Parameters
    ----------
    sq_directory : str or Path
        Path to the sequence directory.
    max_models : int, optional
        Maximum number of models to load (None = load all).
    verbose : bool, optional
        Print loading progress (default: False).

    Attributes
    ----------
    sq_directory : Path
        Path to the sequence directory.
    models_directory : Path
        Path to the 5mext subdirectory.
    models : list[Model]
        List of loaded Model objects.
    num_models : int
        Number of models loaded.
    seq_data : pd.DataFrame or None
        Full seq.txt evolution table.
    age_sequence : np.ndarray or None
        Age column from seq.txt.
    file_paths : list[Path]
        Paths to all model files.
    model_index : list[int or None]
        Model indices extracted from filenames.

    Examples
    --------
    >>> seq = Sequence("path/to/sq/")  # doctest: +SKIP
    >>> len(seq)  # doctest: +SKIP
    100
    >>> seq.get_age(0)  # doctest: +SKIP
    1.0e+06
    """

    SEQ_COLUMNS: list[str] = [
        'Mod', 'Teff', 'Log_g', 'Rayon', 'Age', 'Lum',
        'Log_Tc', 'Log_Pc', 'Log_rhoc', 'M_x_M', 'Log_q_x',
        'Lum_nu', 'Log_H', 'Log_He', 'Log_C', 'Log_O',
    ]

    _MD_PATTERN = re.compile(r'^md\d+\.txt$', re.IGNORECASE)

    def __init__(
        self,
        sq_directory: str | Path,
        max_models: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.sq_directory: Path = Path(sq_directory)
        self.models_directory: Path = self.sq_directory / '5mext'
        self.seq_file: Path = self.sq_directory / 'seq.txt'
        self.idx_file: Path = self.sq_directory / 'idx.txt'

        self.max_models: Optional[int] = max_models
        self.verbose: bool = verbose

        self.file_paths: list[Path] = []
        self.models: list[Model] = []
        self.num_models: int = 0
        self.model_index: list[int | None] = []
        self.seq_data: pd.DataFrame | None = None
        self.age_sequence: np.ndarray | None = None

        self._validate_structure()
        self._load_seq_data()
        self._load_models()

    def _validate_structure(self) -> None:
        """Validate the sequence directory structure."""
        for path, label in [
            (self.sq_directory,    'sq directory'),
            (self.models_directory,'5mext directory'),
            (self.seq_file,        'seq.txt'),
        ]:
            if not path.exists():
                raise SequenceFileError(f"{label} not found: {path}")
            if self.verbose:
                logger.info("✓ Found %s: %s", label, path)

        if self.verbose and self.idx_file.exists():
            logger.info("✓ Found idx.txt: %s", self.idx_file)

    def _load_seq_data(self) -> None:
        """Load the seq.txt evolution table."""
        if self.verbose:
            logger.info("\nReading evolution sequence from %s…", self.seq_file.name)
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
                logger.info("✓ Loaded %d rows from seq.txt", len(self.seq_data))
                logger.info(
                    "  Age range: %.2e – %.2e",
                    self.age_sequence.min(),
                    self.age_sequence.max(),
                )
        except Exception as exc:
            if self.verbose:
                logger.warning("⚠ Could not parse seq.txt: %s", exc)
                logger.warning("  Will use model indices as time sequence.")
            self.seq_data = None
            self.age_sequence = None

    def _get_file_list(self) -> tuple[list[Path], list[int | None]]:
        """Get sorted list of model files and their indices."""
        files = [f for f in self.models_directory.glob('*') if f.is_file()]
        if not files:
            raise SequenceFileError(f"No model files found in {self.models_directory}")

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
        """Load all model files."""
        self.file_paths, self.model_index = self._get_file_list()
        self.num_models = len(self.file_paths)

        if self.verbose:
            logger.info(
                "\nLoading %d models from %s…",
                self.num_models,
                self.models_directory,
            )

        for i, path in enumerate(self.file_paths):
            try:
                self.models.append(Model(path))
                if self.verbose and (i + 1) % 10 == 0:
                    logger.info("  %d/%d loaded…", i + 1, self.num_models)
            except Exception as exc:
                if self.verbose:
                    logger.warning("  ⚠ Skipping %s: %s", path.name, exc)

        if self.verbose:
            logger.info("✓ Successfully loaded %d models.", len(self.models))

    def get_model(self, index: int) -> Model:
        """
        Return model at index (0-based).

        Parameters
        ----------
        index : int
            Index of the model to retrieve.

        Returns
        -------
        Model
            The requested model.

        Raises
        ------
        ValidationError
            If index is out of range.
        """
        if not 0 <= index < len(self.models):
            raise ValidationError(
                f"Index {index} out of range [0, {len(self.models) - 1}]"
            )
        return self.models[index]

    def get_age(self, index: int) -> float:
        """
        Return the stellar age at index.

        Parameters
        ----------
        index : int
            Index of the age to retrieve.

        Returns
        -------
        float
            The stellar age in years.

        Raises
        ------
        ValidationError
            If index is out of range or age_sequence not available.
        """
        if self.age_sequence is None:
            raise ValidationError(
                "age_sequence not available (seq.txt could not be parsed)"
            )
        if not 0 <= index < len(self.age_sequence):
            raise ValidationError(f"Index {index} out of range")
        return float(self.age_sequence[index])

    def export_evolution_csv(
        self,
        output_file: str | Path,
        parameters: list[str],
    ) -> None:
        """
        Export the evolution of parameters to a CSV file.

        Parameters
        ----------
        output_file : str or Path
            Path to the output CSV file.
        parameters : list[str]
            List of parameter names to export.

        Raises
        ------
        SequenceFileError
            If export fails.
        """
        try:
            from ..analysis.matching import SequenceAnalyzer
            analyzer = SequenceAnalyzer(self)
            df = analyzer.create_evolution_dataframe(parameters)
            df.to_csv(output_file, index=False)
            logger.info("Evolution data exported to %s", output_file)
            print(f"Evolution data exported to {output_file}")
        except Exception as exc:
            raise SequenceFileError(f"Failed to export evolution data: {exc}") from exc

    def summary(self) -> None:
        """Print a summary of the sequence to stdout."""
        sep = '=' * 70
        lines = [
            sep,
            "Sequence Summary",
            sep,
            f"sq directory:     {self.sq_directory}",
            f"Models directory: {self.models_directory}",
            f"Models loaded:    {len(self.models)}",
            sep,
        ]
        if self.models and self.age_sequence is not None:
            lines.extend([
                "",
                f"Age range:  {self.age_sequence.min():.2e} – {self.age_sequence.max():.2e}",
                "",
                f"First model: {self.file_paths[0].name}  "
                f"T_eff={self.models[0].T_eff:.1f} K  "
                f"log_g={self.models[0].log_g:.2f}",
                f"Last  model: {self.file_paths[-1].name}  "
                f"T_eff={self.models[-1].T_eff:.1f} K  "
                f"log_g={self.models[-1].log_g:.2f}",
            ])
        lines.append(sep)
        logger.info("\n".join(lines))
        # Also print for backward compatibility
        for line in lines:
            print(line)

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, index: int) -> Model:
        return self.get_model(index)

    def __repr__(self) -> str:
        return (
            f"Sequence('{self.sq_directory}', "
            f"num_models={len(self.models)})"
        )
