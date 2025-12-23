#hardware\src\utils\logger_hareware.py

import pandas as pd
from pathlib import Path
from datetime import datetime


class Logger:
    """
    Lightweight logger optimized for embedded devices (e.g., Jetson Nano).

    Design principles:
    - Minimize DataFrame concatenation
    - Use list-based buffering
    - Periodically flush to CSV
    """

    def __init__(self, flush_every=1000):
        """
        Parameters
        ----------
        flush_every : int
            Number of rows before automatic flush to CSV buffer
        """
        self.buffer = []               # temporary storage (list of dict)
        self.df = pd.DataFrame()       # optional in-memory view
        self.flush_every = flush_every
        self.current_path = Path.cwd()
        self._csv_path = None

    # ==========================================================
    # Logging
    # ==========================================================
    def add_data_log(self, columns_name, data_list):
        """
        Add data to internal buffer (NOT directly to DataFrame).
        """
        def safe_len(x):
            return len(x) if hasattr(x, "__len__") else 1

        max_len = max(safe_len(d) for d in data_list)

        normalized = []
        for d in data_list:
            if not hasattr(d, "__len__"):
                d = [d] * max_len
            elif len(d) < max_len:
                d = list(d) + [None] * (max_len - len(d))
            else:
                d = list(d)[:max_len]
            normalized.append(d)

        for i in range(max_len):
            row = {col: normalized[j][i] for j, col in enumerate(columns_name)}
            self.buffer.append(row)

        if len(self.buffer) >= self.flush_every:
            self.flush()

    # ==========================================================
    # Flush / Save
    # ==========================================================
    def flush(self):
        """
        Convert buffer to DataFrame and append to internal DataFrame / CSV.
        """
        if not self.buffer:
            return

        new_df = pd.DataFrame(self.buffer)
        self.buffer.clear()

        if self.df.empty:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

        # Optional: auto-save if path already defined
        if self._csv_path is not None:
            header = not self._csv_path.exists()
            self.df.to_csv(self._csv_path, mode="a", index=False, header=header)
            self.df = pd.DataFrame()  # free RAM

    def save_to_csv(self, file_name, folder_name=None, path_name=None):
        """
        Define CSV output and flush remaining buffer.
        """
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        base_path = Path(path_name) if path_name else self.current_path
        folder = base_path / (folder_name or datetime.now().strftime("%Y-%m-%d"))
        folder.mkdir(parents=True, exist_ok=True)

        self._csv_path = folder / file_name
        self.flush()

        print(f"Logging to {self._csv_path}")

    # ==========================================================
    # Utility
    # ==========================================================
    def clear_data(self):
        self.buffer.clear()
        self.df = pd.DataFrame()
        print("Logger cleared")

    def show_data(self, tail=5):
        """
        Show last few rows only (safe for Jetson).
        """
        if not self.df.empty:
            print(self.df.tail(tail))
        else:
            print("No in-memory data")
