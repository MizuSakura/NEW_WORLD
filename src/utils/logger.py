# src/utils/logger.py

import pandas as pd
from pathlib import Path
from datetime import datetime


class Logger:
    """
    A lightweight and flexible logger for recording simulation or experiment data.

    This class provides high-level tools to:
      - Add new data rows dynamically (`add_data_log`)
      - Save and load logs as CSV files
      - Append and merge multiple CSV logs from a folder
      - Check conditions on specific columns (useful for monitoring states)
      - Handle dynamic columns automatically

    The logger stores all data internally as a pandas DataFrame (`self.df`).

    Example
    -------
    >>> from utils.logger import Logger
    >>> log = Logger()
    >>> log.add_data_log(["time", "value"], [[1, 2, 3], [10, 20, 30]])
    >>> log.save_to_csv("run1.csv")
    >>> log.show_data()
    """

    def __init__(self):
        """Initialize an empty logger."""
        self.df = pd.DataFrame()
        self.current_path = Path.cwd()

    # ==========================================================
    # Logging / Adding Data
    # ==========================================================
    def add_data_log(self, columns_name, data_list):
        """
        Add a new log entry or multiple rows to the DataFrame.

        Handles:
          - Variable-length data across columns
          - Automatic column creation if new columns are detected

        Parameters
        ----------
        columns_name : list of str
            Column names for the data being added.
        data_list : list
            List of lists or values corresponding to each column.

        Raises
        ------
        ValueError
            If duplicate column names are provided.
        """
        if len(columns_name) != len(set(columns_name)):
            raise ValueError("columns_name contains duplicate columns")

        # Helper: return length of data safely (handles scalars)
        def safe_len(x):
            return len(x) if hasattr(x, "__len__") else 1

        # Determine the longest data column
        max_len = max(safe_len(col_data) for col_data in data_list)

        # Normalize each column to have the same length
        padded_data = []
        for col_data in data_list:
            if not hasattr(col_data, "__len__"):
                col_data = [col_data] * max_len
            elif len(col_data) < max_len:
                col_data = list(col_data) + [None] * (max_len - len(col_data))
            else:
                col_data = list(col_data)[:max_len]
            padded_data.append(col_data)

        # Combine into a dict for DataFrame creation
        new_data = {col: data for col, data in zip(columns_name, padded_data)}

        if self.df.empty:
            # If no data yet, initialize new DataFrame
            self.df = pd.DataFrame(new_data)
        else:
            # Ensure all columns align dynamically
            for col in self.df.columns:
                if col not in new_data:
                    new_data[col] = [None] * max_len
            for col in new_data:
                if col not in self.df.columns:
                    self.df[col] = [None] * len(self.df)

            # Create DataFrame for new rows, reorder columns
            new_rows = pd.DataFrame(new_data)
            new_rows = new_rows[self.df.columns]
            self.df = pd.concat([self.df, new_rows], ignore_index=True)

    # ==========================================================
    # Clear / Show
    # ==========================================================
    def clear_data(self):
        """Clear all logged data."""
        self.df = pd.DataFrame()
        print("Data cleared")

    def show_data(self):
        """Display the current logged data."""
        print(self.df)

    # ==========================================================
    # Get Column Data
    # ==========================================================
    def result_column(self, column_name=None):
        """
        Retrieve a specific column as a NumPy array.

        Parameters
        ----------
        column_name : str, optional
            Name of the column to extract.

        Returns
        -------
        numpy.ndarray or None
            Column values as an array, or None if not found.
        """
        if self.df.empty:
            print("No data to display")
            return None
        if column_name is None:
            return None
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found")
            return None
        return self.df[column_name].to_numpy()

    # ==========================================================
    # Save / Load CSV
    # ==========================================================
    def save_to_csv(self, file_name, folder_name=None, path_name=None):
        """
        Save the current log to a CSV file.

        Parameters
        ----------
        file_name : str
            Name of the file (without extension).
        folder_name : str, optional
            Folder where to save the file. Defaults to date-based folder.
        path_name : str, optional
            Base directory path. Defaults to current working directory.

        Example
        -------
        >>> log.save_to_csv("session1", folder_name="logs")
        """
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        path_to_save = Path(path_name) if path_name else self.current_path
        folder_to_save = path_to_save / (folder_name or datetime.now().strftime("%Y-%m-%d"))
        folder_to_save.mkdir(parents=True, exist_ok=True)
        path_file = folder_to_save / file_name
        self.df.to_csv(path_file, index=False)
        print(f"Data saved to {path_file}")

    def load_csv(self, path_file):
        """
        Load a CSV file into the logger.

        Parameters
        ----------
        path_file : str
            Path to the CSV file to load.
        """
        path = Path(path_file)
        if path.suffix.lower() != ".csv":
            raise ValueError("File must be a CSV")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        self.df = pd.read_csv(path)
        print(f"Data loaded from {path}")

    # ==========================================================
    # Append CSVs from folder
    # ==========================================================
    def append_from_csv_folder(self, folder_path):
        """
        Append and merge all CSV files from a folder into the current DataFrame.

        Handles inconsistent columns by automatically aligning them.

        Parameters
        ----------
        folder_path : str or Path
            Folder path containing CSV files.

        Example
        -------
        >>> log.append_from_csv_folder("./results")
        """
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

        # Merge combined data with current logger state
        if self.df.empty:
            self.df = combined_df
        else:
            # Auto-align new columns
            for col in combined_df.columns:
                if col not in self.df.columns:
                    self.df[col] = [None] * len(self.df)
            for col in self.df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = [None] * len(combined_df)
            combined_df = combined_df[self.df.columns]
            self.df = pd.concat([self.df, combined_df], ignore_index=True)

        print(f"Appended {len(csv_files)} files from {folder}")

    # ==========================================================
    # Check column condition
    # ==========================================================
    def check_column_condition(self, column_name, target_value=True, min_count=0, start=None, end=None):
        """
        Check if a column meets a target condition within a range.

        Useful for monitoring state changes or thresholds in logged data.

        Parameters
        ----------
        column_name : str
            Name of the column to check.
        target_value : Any, optional (default=True)
            Value to match within the column.
        min_count : int, optional (default=0)
            Minimum number of occurrences required.
        start, end : int, optional
            Slice indices for checking within a specific range.

        Returns
        -------
        bool
            True if condition is met, otherwise False.
        """
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


# ==========================================================
# Example usage (for testing and demonstration)
# ==========================================================
if __name__ == "__main__":
    log = Logger()

    # Add example data
    log.add_data_log(["step", "reward"], [[1, 2, 3], [10, 20, 30]])
    log.add_data_log(["done"], [[False, False, True]])

    # Show results
    log.show_data()

    # Save to CSV
    log.save_to_csv("example_log")

    # Load it back
    log.load_csv("2025-10-07/example_log.csv")

    # Check a condition
    print("Condition met:", log.check_column_condition("done", target_value=True, min_count=1))
