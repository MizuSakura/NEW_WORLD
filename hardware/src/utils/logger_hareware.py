#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/utils/logger_hareware.py
# hardware/src/utils/logger_hareware.py
"""
แก้ไขให้ compatible กับ Python 3.6.9, pandas 0.22.0

pandas 0.22.0 ต่างจากเวอร์ชันใหม่:
    - pd.concat ยังใช้ได้ แต่ ignore_index ใช้ได้แล้ว (มีตั้งแต่ 0.13)
    - DataFrame.empty ใช้ได้
    - ระวัง: ไม่มี pd.DataFrame.to_csv mode="a" ใน pandas เก่ามาก
      (pandas 0.22 มี mode parameter แล้ว OK)
"""

from __future__ import print_function

import pandas as pd
from pathlib import Path
from datetime import datetime


class Logger(object):
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
        self.buffer      = []
        self.df          = pd.DataFrame()
        self.flush_every = flush_every
        self.current_path = Path.cwd()
        self._csv_path   = None

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
        self.buffer = []   # clear (Python 3.6 ใช้ .clear() ได้แต่ reassign ปลอดภัยกว่า)

        if self.df.empty:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)

        if self._csv_path is not None:
            header = not self._csv_path.exists()
            self.df.to_csv(str(self._csv_path), mode="a", index=False, header=header)
            self.df = pd.DataFrame()

    def save_to_csv(self, file_name, folder_name=None, path_name=None):
        """
        Define CSV output and flush remaining buffer.
        """
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        base_path = Path(path_name) if path_name else self.current_path
        folder_arg = folder_name or datetime.now().strftime("%Y-%m-%d")
        # folder_name อาจเป็น "" (empty string) → ใช้ base_path ตรงๆ
        if folder_arg:
            folder = base_path / folder_arg
        else:
            folder = base_path
        folder.mkdir(parents=True, exist_ok=True)

        self._csv_path = folder / file_name
        self.flush()

        print("Logging to {}".format(self._csv_path))

    # ==========================================================
    # Utility
    # ==========================================================
    def clear_data(self):
        self.buffer = []
        self.df     = pd.DataFrame()
        print("Logger cleared")

    def show_data(self, tail=5):
        """
        Show last few rows only (safe for Jetson).
        """
        if not self.df.empty:
            print(self.df.tail(tail))
        else:
            print("No in-memory data")
if __name__ == "__main__":
    # Quick Test
    logger = Logger(flush_every=5)
    logger.save_to_csv("test_log.csv", folder_name="test_output")
    
    cols = ["step", "val"]
    for i in range(7):
        logger.add_data_log(cols, [i, i*10])
    
    logger.flush()
    print("Test finished.")
