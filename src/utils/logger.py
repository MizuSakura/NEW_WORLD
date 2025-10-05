# src/utils/logger.py
import pandas as pd
from pathlib import Path
from datetime import datetime

class Logger:
    """
    Logger สำหรับบันทึกข้อมูล simulation / experiment
    รองรับ:
      - add_data_log: เพิ่ม row ข้อมูล
      - save_to_csv / load_csv
      - append CSV จาก folder
      - check_column_condition
      - dynamic column handling
    """
    def __init__(self):
        self.df = pd.DataFrame()
        self.current_path = Path.cwd()

    # -----------------------------
    # Logging / Adding Data
    # -----------------------------
    def add_data_log(self, columns_name, data_list):
        if len(columns_name) != len(set(columns_name)):
            raise ValueError("columns_name contains duplicate columns")

        def safe_len(x):
            return len(x) if hasattr(x, '__len__') else 1

        max_len = max(safe_len(col_data) for col_data in data_list)

        padded_data = []
        for col_data in data_list:
            if not hasattr(col_data, '__len__'):
                col_data = [col_data] * max_len
            elif len(col_data) < max_len:
                col_data = list(col_data) + [None] * (max_len - len(col_data))
            else:
                col_data = list(col_data)[:max_len]
            padded_data.append(col_data)

        new_data = {col: data for col, data in zip(columns_name, padded_data)}

        if self.df.empty:
            self.df = pd.DataFrame(new_data)
        else:
            # เพิ่ม column ใหม่อัตโนมัติ
            for col in self.df.columns:
                if col not in new_data:
                    new_data[col] = [None] * max_len
            for col in new_data:
                if col not in self.df.columns:
                    self.df[col] = [None] * len(self.df)
            # concat row ใหม่
            new_rows = pd.DataFrame(new_data)
            new_rows = new_rows[self.df.columns]  # reorder columns
            self.df = pd.concat([self.df, new_rows], ignore_index=True)

    # -----------------------------
    # Clear / Show
    # -----------------------------
    def clear_data(self):
        self.df = pd.DataFrame()
        print("Data cleared")

    def show_data(self):
        print(self.df)

    # -----------------------------
    # Get Column Data
    # -----------------------------
    def result_column(self, column_name=None):
        if self.df.empty:
            print("No data to display")
            return None
        if column_name is None:
            return None
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found")
            return None
        return self.df[column_name].to_numpy()

    # -----------------------------
    # Save / Load CSV
    # -----------------------------
    def save_to_csv(self, file_name, folder_name=None, path_name=None):
        if not file_name.endswith('.csv'):
            file_name += '.csv'

        path_to_save = Path(path_name) if path_name else self.current_path
        folder_to_save = path_to_save / (folder_name or datetime.now().strftime("%Y-%m-%d"))
        folder_to_save.mkdir(parents=True, exist_ok=True)
        path_file = folder_to_save / file_name
        self.df.to_csv(path_file, index=False)
        print(f"Data saved to {path_file}")

    def load_csv(self, path_file):
        path = Path(path_file)
        if path.suffix.lower() != ".csv":
            raise ValueError("File must be a CSV")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        self.df = pd.read_csv(path)
        print(f"Data loaded from {path}")

    # -----------------------------
    # Append CSVs from folder
    # -----------------------------
    def append_from_csv_folder(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            print(f"Invalid folder: {folder}")
            return

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files in {folder}")
            return

        combined_df = pd.DataFrame()
        for file in csv_files:
            try:
                temp_df = pd.read_csv(file)
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")

        if self.df.empty:
            self.df = combined_df
        else:
            # รวม column ใหม่อัตโนมัติ
            for col in combined_df.columns:
                if col not in self.df.columns:
                    self.df[col] = [None] * len(self.df)
            for col in self.df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = [None] * len(combined_df)
            combined_df = combined_df[self.df.columns]
            self.df = pd.concat([self.df, combined_df], ignore_index=True)

        print(f"Appended {len(csv_files)} files from {folder}")

    # -----------------------------
    # Check column condition
    # -----------------------------
    def check_column_condition(self, column_name, target_value=True, min_count=0, start=None, end=None):
        if self.df.empty:
            print("No data to check")
            return False
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found")
            return False

        data_slice = self.df[column_name]
        if start is not None or end is not None:
            data_slice = data_slice[start:end]
        count = (data_slice == target_value).sum()
        return count >= min_count
