# src/utils/scaling_reference_combined.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from pathlib import Path
import zipfile
import os
import yaml
import io


class GlobalScalingReference:
    """
    GlobalScalingReference
    =======================
    A scalable and efficient tool for fitting input/output scalers across 
    multiple CSV files. This class supports incremental fitting for large datasets 
    (via `chunksize`) and saves all scaling artifacts and metadata into a single ZIP file.
    
    This utility is commonly used to prepare normalized references for 
    machine learning or control system models (e.g., LSTM forecasters, RL agents).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing all CSV files used for fitting the scalers.
    dataset_name : str, optional
        Name of the dataset (default: "RC_Tank_Env_Training").
    input_features : list of str, optional
        Column names used as input features.
        Default: ["PWM_duty", "Prev_output"]
    output_features : list of str, optional
        Column names used as output (target) features.
        Default: ["Tank_level"]
    scaler_type : {"MinMaxScaler", "StandardScaler"}, optional
        Type of scaler to use for fitting (default: "MinMaxScaler").
    chunk_size : int, optional
        Number of rows to process at once when reading CSVs (default: 10,000).
    save_dir : str or Path, optional
        Output directory to save the ZIP archive. Defaults to `<data_dir>/scaling_reference`.

    Attributes
    ----------
    scaler_in : sklearn.preprocessing.MinMaxScaler or StandardScaler
        Scaler fitted on input features.
    scaler_out : sklearn.preprocessing.MinMaxScaler or StandardScaler
        Scaler fitted on output (target) features.
    metadata : dict
        Summary information including ranges, feature names, and scaler settings.

    Example
    -------
    >>> scaler_ref = GlobalScalingReference(
    ...     data_dir="./data/raw",
    ...     dataset_name="RC_Tank_Env_Training",
    ...     input_features=["PWM_duty", "Prev_output"],
    ...     output_features=["Tank_level"]
    ... )
    >>> result = scaler_ref.run()
    >>> print(result["metadata"]["input_min"])
    [0.0, 0.1]
    >>> print(result["zip"])
    ./data/raw/scaling_reference/RC_Tank_Env_Training_scalers.zip
    """

    def __init__(self,
                 data_dir,
                 dataset_name="RC_Tank_Env_Training",
                 input_features=None,
                 output_features=None,
                 scaler_type="MinMaxScaler",
                 chunk_size=10000,
                 save_dir=None):

        # ✅ Setup paths and parameters
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.input_features = input_features or ["PWM_duty", "Prev_output"]
        self.output_features = output_features or ["Tank_level"]
        self.chunk_size = chunk_size
        self.scaler_type = scaler_type

        # ✅ Define saving directory
        self.save_dir = Path(save_dir) if save_dir else (self.data_dir / "scaling_reference")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Collect CSV files
        self.csv_files = list(self.data_dir.glob("*.csv"))
        if not self.csv_files:
            raise FileNotFoundError(f"❌ No CSV files found in {self.data_dir}")

        # ✅ Initialize scalers (MinMax or Standard)
        scaler_cls = MinMaxScaler if scaler_type == "MinMaxScaler" else StandardScaler
        self.scaler_in = scaler_cls()
        self.scaler_out = scaler_cls()

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------
    def fit_from_folder(self):
        """
        Incrementally fit input and output scalers using all CSV files in the folder.

        Reads each file in chunks to handle large datasets efficiently and updates
        the scaling parameters (`min_`, `max_`, or `mean_`, `var_`) without storing
        all data in memory.

        Returns
        -------
        metadata : dict
            Dictionary containing dataset metadata and feature ranges.
        """
        print("🚀 Starting Global Scaler Fitting Process ...")
        print(f"📁 Processing {len(self.csv_files)} files from: {self.data_dir}\n")

        for i, csv_path in enumerate(self.csv_files, 1):
            print(f"  [{i}/{len(self.csv_files)}] Reading {csv_path.name}")
            with pd.read_csv(csv_path, chunksize=self.chunk_size) as reader:
                for chunk in reader:
                    # ✅ Check for required columns
                    if not all(col in chunk.columns for col in self.input_features + self.output_features):
                        raise KeyError(f"❌ Missing required columns in {csv_path.name}")

                    X = chunk[self.input_features].values
                    y = chunk[self.output_features].values

                    # ✅ Incremental fit
                    self.scaler_in.partial_fit(X)
                    self.scaler_out.partial_fit(y)

        # ✅ Save metadata after fitting
        self.metadata = {
            "dataset_name": self.dataset_name,
            "scaler_type": self.scaler_type,
            "num_files": len(self.csv_files),
            "input_features": self.input_features,
            "output_features": self.output_features,
            "input_min": self.scaler_in.data_min_.tolist(),
            "input_max": self.scaler_in.data_max_.tolist(),
            "output_min": self.scaler_out.data_min_.tolist(),
            "output_max": self.scaler_out.data_max_.tolist()
        }

        print("\n✅ Scaler fitting complete!")
        print(f"  Input range:  {self.scaler_in.data_min_} → {self.scaler_in.data_max_}")
        print(f"  Output range: {self.scaler_out.data_min_} → {self.scaler_out.data_max_}")
        return self.metadata

    # -------------------------------------------------------------------------
    # Save ZIP
    # -------------------------------------------------------------------------
    def save_all_to_zip(self):
        """
        Save fitted input/output scalers and metadata into a single compressed ZIP file.

        This ZIP archive includes:
        - `input_scaler.pkl` : Pickle of the fitted input scaler
        - `output_scaler.pkl`: Pickle of the fitted output scaler
        - `metadata.yaml`    : Human-readable metadata file

        Returns
        -------
        zip_path : Path
            Path to the saved ZIP archive.
        """
        zip_path = self.save_dir / f"{self.dataset_name}_scalers.zip"
        print(f"\n💾 Creating single ZIP archive: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # ✅ Serialize and write both scalers directly into the ZIP
            for name, scaler in [("input_scaler.pkl", self.scaler_in),
                                 ("output_scaler.pkl", self.scaler_out)]:
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                zipf.writestr(name, buffer.getvalue())

            # ✅ Write metadata YAML
            yaml_str = yaml.dump(self.metadata, allow_unicode=True, sort_keys=False)
            zipf.writestr("metadata.yaml", yaml_str)

        print(f"📦 ZIP created successfully at: {zip_path}")
        return zip_path

    # -------------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------------
    def run(self):
        """
        Execute the full scaling reference generation pipeline:
        1. Fit scalers across all CSV files
        2. Save all results into a ZIP archive

        Returns
        -------
        result : dict
            Contains the following keys:
              - 'metadata' : dict of scaler statistics and info
              - 'zip' : Path to the generated ZIP archive
        """
        self.fit_from_folder()
        zip_path = self.save_all_to_zip()
        return {"metadata": self.metadata, "zip": zip_path}


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    FOLDER_DATA = r"D:\Project_end\New_world\my_project\data\raw"
    FOLDER_SAVE_SCALE = r"D:\Project_end\New_world\my_project\config"
    NAME_FILE = "RC_Tank_Env_Training2"
    COLUMN_INPUT = "DATA_INPUT"
    COLUMN_OUTPUT = "DATA_OUTPUT"
    SCALER_TYPE = "MinMaxScaler"
    CHUNK_SIZE = 10000

    # 🧩 Initialize and run global scaler builder
    scaler_ref = GlobalScalingReference(
        data_dir=FOLDER_DATA,
        save_dir=FOLDER_SAVE_SCALE,
        dataset_name=NAME_FILE,
        input_features=[COLUMN_INPUT],
        output_features=[COLUMN_OUTPUT],
        scaler_type=SCALER_TYPE,
        chunk_size=CHUNK_SIZE
    )

    result = scaler_ref.run()

    # 🧾 Display summary table
    print("\n🎯 Summary:")
    print(pd.DataFrame([result["metadata"]]))
