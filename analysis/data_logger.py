"""
CSV data logger — records per-timestep simulation data.
"""

import os
import csv
import numpy as np


class DataLogger:
    """Append-only structured data logger for simulation output."""

    def __init__(self):
        self.data = {}
        self.columns = []
        self._n_rows = 0

    def record(self, row: dict):
        """Record one row of data. Columns auto-registered on first call."""
        if not self.columns:
            self.columns = list(row.keys())
            for col in self.columns:
                self.data[col] = []

        for col in self.columns:
            self.data[col].append(row.get(col, np.nan))

        self._n_rows += 1

    def get_column(self, name: str) -> np.ndarray:
        """Return column as numpy array."""
        if name in self.data:
            return np.array(self.data[name])
        return np.array([])

    def save(self, path: str):
        """Save recorded data to CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            for i in range(self._n_rows):
                row = [self.data[col][i] for col in self.columns]
                writer.writerow(row)
        print(f"[Logger] Saved {self._n_rows} rows to {path}")

    def save_config(self, path: str, config: dict):
        """Save config dict as JSON."""
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def clear(self):
        self.data.clear()
        self.columns.clear()
        self._n_rows = 0
