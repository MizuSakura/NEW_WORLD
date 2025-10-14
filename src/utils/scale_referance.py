import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from pathlib import Path
import zipfile
import os
import yaml
import io
import platform
import torch
from datetime import datetime
import sklearn
import psutil 


class GlobalScalingReference:
    """
    GlobalScalingReference
    =======================
    A scalable and efficient tool for fitting input/output scalers across
    multiple CSV files. This class supports **incremental fitting** (via `chunksize`)
    for large datasets, and exports all fitted scalers and metadata into
    a single ZIP archive.

    The generated ZIP file contains:
      - `input_scaler.pkl` : fitted scaler for input features
      - `output_scaler.pkl` : fitted scaler for output features
      - `metadata.yaml` : detailed metadata including project, dataset, and system info

    Parameters
    ----------
    user_creatr : str
        Name or identifier of the user who created the scaling reference.
    name_project : str
        Name of the project associated with this scaling reference.
    data_dir : str or Path
        Directory containing raw CSV data files.
    dataset_name : str, default="RC_Tank_Env_Training"
        Name of the dataset or experiment.
    input_features : list[str], optional
        Input feature column names (default: ["PWM_duty", "Prev_output"]).
    output_features : list[str], optional
        Output feature column names (default: ["Tank_level"]).
    scaler_type : str, default="MinMaxScaler"
        Type of scaler to use, either `"MinMaxScaler"` or `"StandardScaler"`.
    chunk_size : int, default=10000
        Number of rows per chunk for incremental fitting.
    save_dir : str or Path, optional
        Directory to store the resulting ZIP file (default: `<data_dir>/scaling_reference`).
    time_format : str, default="%Y-%m-%d %H:%M:%S"
        Datetime format for timestamps.
    project_version : str, default="1.0.0"
        Version identifier for the project.
    description : str, optional
        A short description of the scaling reference purpose.
    notes : str, optional
        Additional notes or context.

    Attributes
    ----------
    scaler_in : object
        The fitted input scaler (MinMaxScaler or StandardScaler).
    scaler_out : object
        The fitted output scaler (MinMaxScaler or StandardScaler).
    metadata : dict
        Dictionary containing detailed metadata generated after fitting.

    Example
    -------
    >>> scaler_ref = GlobalScalingReference(
    ...     user_creatr="Researcher",
    ...     name_project="WaterLevelRL",
    ...     data_dir="./data/raw",
    ...     input_features=["PWM_duty", "Prev_output"],
    ...     output_features=["Tank_level"]
    ... )
    >>> result = scaler_ref.run()
    >>> print(result["metadata"]["scaling"]["method"])
    'MinMaxScaler'
    """

    def __init__(self, user_create, name_project,
                 data_dir,
                 dataset_name="RC_Tank_Env_Training",
                 input_features=None,
                 output_features=None,
                 scaler_type="MinMaxScaler",
                 chunk_size=10000,
                 save_dir=None,
                 time_format="%Y-%m-%d %H:%M:%S",
                 project_version="1.0.0",
                 description="Scaling reference for ML model preprocessing",
                 notes=None):

        # Initialize class attributes
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.input_features = input_features or ["PWM_duty"]
        self.output_features = output_features or ["Tank_level"]
        self.chunk_size = chunk_size
        self.scaler_type = scaler_type
        self.time_format = time_format

        # --- Project metadata parameters ---
        self.user_create = user_create
        self.name_project = name_project
        self.project_version = project_version
        self.description = description
        self.notes = notes

        # Create save directory if not specified
        self.save_dir = Path(save_dir) if save_dir else (self.data_dir / "scaling_reference")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Find all CSV files in the directory
        self.csv_files = sorted(list(self.data_dir.glob("*.csv")))
        if not self.csv_files:
            raise FileNotFoundError(f"❌ No CSV files found in {self.data_dir}")

        # Initialize scaler type
        scaler_cls = MinMaxScaler if scaler_type == "MinMaxScaler" else StandardScaler
        self.scaler_in = scaler_cls()
        self.scaler_out = scaler_cls()

        # Dynamic tracking attributes
        self.total_rows = 0
        self.metadata = {}

    def _get_current_timestamp(self):
        """Return the current local timestamp formatted according to `self.time_format`."""
        return datetime.now().strftime(self.time_format)

    def fit_from_folder(self):
        """
        Incrementally fit both input and output scalers using all CSV files found in the data directory.

        This method reads each CSV file in chunks to efficiently handle large datasets
        without loading them entirely into memory. It performs partial fitting on
        both input and output scalers using the defined feature columns.

        Returns
        -------
        dict
            The metadata dictionary generated after fitting.
        """
        print("🚀 Starting Global Scaler Fitting Process ...")
        print(f"📁 Processing {len(self.csv_files)} files from: {self.data_dir}\n")

        self.total_rows = 0  # Reset counter for each run

        # Iterate through all CSV files
        for i, csv_path in enumerate(self.csv_files, 1):
            print(f"  [{i}/{len(self.csv_files)}] Reading {csv_path.name}")

            # Read large CSV file in chunks
            with pd.read_csv(csv_path, chunksize=self.chunk_size) as reader:
                for chunk in reader:
                    # Ensure all required columns exist
                    if not all(col in chunk.columns for col in self.input_features + self.output_features):
                        raise KeyError(f"❌ Missing required columns in {csv_path.name}")

                    # Count rows processed
                    self.total_rows += len(chunk)

                    # Extract input/output features
                    X = chunk[self.input_features].values
                    y = chunk[self.output_features].values

                    # Incrementally fit the scalers
                    self.scaler_in.partial_fit(X)
                    self.scaler_out.partial_fit(y)

        print("\n✅ Scaler fitting complete!")
        self._create_metadata()  # Generate metadata after fitting

        # Display summary of fitted parameters
        if self.scaler_type == "MinMaxScaler":
            print(f"  Input range:  {self.scaler_in.data_min_} → {self.scaler_in.data_max_}")
            print(f"  Output range: {self.scaler_out.data_min_} → {self.scaler_out.data_max_}")
        else:
            print(f"  Input Mean: {self.scaler_in.mean_}")
            print(f"  Output Mean: {self.scaler_out.mean_}")

        return self.metadata

    def _create_metadata(self):
        """
        Create a complete metadata dictionary describing the fitted scalers, dataset, and system information.
        This includes project metadata, dataset info, scaling parameters, and runtime system configuration.
        """
        now = self._get_current_timestamp()

        # Build parameter summary based on scaler type
        scaler_params = {}
        if self.scaler_type == "MinMaxScaler":
            scaler_params['input'] = {
                'min': self.scaler_in.min_.tolist(), 'scale': self.scaler_in.scale_.tolist(),
                'data_min': self.scaler_in.data_min_.tolist(), 'data_max': self.scaler_in.data_max_.tolist(),
                'feature_range': list(self.scaler_in.feature_range)
            }
            scaler_params['output'] = {
                'min': self.scaler_out.min_.tolist(), 'scale': self.scaler_out.scale_.tolist(),
                'data_min': self.scaler_out.data_min_.tolist(), 'data_max': self.scaler_out.data_max_.tolist(),
                'feature_range': list(self.scaler_out.feature_range)
            }
        else:  # StandardScaler
            scaler_params['input'] = {
                'mean': self.scaler_in.mean_.tolist(), 'scale': self.scaler_in.scale_.tolist(),
                'var': self.scaler_in.var_.tolist(), 'n_samples_seen': int(self.scaler_in.n_samples_seen_)
            }
            scaler_params['output'] = {
                'mean': self.scaler_out.mean_.tolist(), 'scale': self.scaler_out.scale_.tolist(),
                'var': self.scaler_out.var_.tolist(), 'n_samples_seen': int(self.scaler_out.n_samples_seen_)
            }

        # Attempt to get system memory info
        try:
            ram_gb = f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
        except (ImportError, AttributeError):
            ram_gb = "Unknown (psutil not installed)"

        # Assemble metadata structure
        self.metadata = {
            "project": {
                "name": self.name_project,
                "version": self.project_version,
                "created_by": self.user_create,
                "created_at": now,
            },
            "dataset": {
                "name": self.dataset_name,
                "source_files": [f.name for f in self.csv_files],
                "total_rows": self.total_rows, "input_features": self.input_features,
                "output_features": self.output_features,
            },
            "scaling": {
                "method": self.scaler_type, "parameters": scaler_params
            },
            "system_info": {
                "description": self.description, "notes": self.notes,
                "training_device": {
                    "os": f"{platform.system()} {platform.release()}",
                    "cpu": platform.processor(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                    "ram": ram_gb, "python_version": platform.python_version(),
                },
            }
        }

    def save_all_to_zip(self):
        """
        Save all fitted scalers and generated metadata into a single compressed ZIP archive.

        The ZIP will contain:
          - input_scaler.pkl
          - output_scaler.pkl
          - metadata.yaml

        Returns
        -------
        Path
            Path to the created ZIP file.
        """
        if not self.metadata:
            raise RuntimeError("Metadata not created. Please run 'fit_from_folder()' first.")

        zip_path = self.save_dir / f"{self.dataset_name}_scalers.zip"
        print(f"\n💾 Creating single ZIP archive: {zip_path}")

        # Add update timestamp
        self.metadata["project"]["updated_at"] = self._get_current_timestamp()

        # Write all artifacts into a single ZIP
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for name, scaler in [("input_scaler.pkl", self.scaler_in), ("output_scaler.pkl", self.scaler_out)]:
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                zipf.writestr(name, buffer.getvalue())

            # Convert metadata dict → YAML string
            yaml_str = yaml.dump(self.metadata, allow_unicode=True, sort_keys=False, indent=2)
            zipf.writestr("metadata.yaml", yaml_str)

        print(f"📦 ZIP created successfully at: {zip_path}")
        return zip_path

    def run(self):
        """
        Execute the full scaling reference generation pipeline:
        1. Fit scalers across all CSV files.
        2. Generate metadata.
        3. Save all artifacts into a single ZIP.

        Returns
        -------
        dict
            Dictionary with keys:
              - 'metadata': scaling and project metadata
              - 'zip': path to saved ZIP archive
        """
        self.fit_from_folder()
        zip_path = self.save_all_to_zip()
        return {"metadata": self.metadata, "zip": zip_path}


# =============================================================================
#           MAIN EXECUTION BLOCK (Example)
# =============================================================================
if __name__ == "__main__":
    # Example standalone execution for testing
    # Users can modify paths and parameters as needed

    ROOT_DIR = Path(__file__).resolve().parents[2]  
    FOLDER_DATA = ROOT_DIR / "data" / "raw"        
    FOLDER_SAVE_SCALE = ROOT_DIR / "config" 
    DATASET_NAME = "Test_scale1"

    INPUT_FEATURES = ["DATA_INPUT"]
    OUTPUT_FEATURES = ["DATA_OUTPUT"]

    SCALER_TYPE = "MinMaxScaler"
    CHUNK_SIZE = 20000

    PROJECT_NAME = "RC_Tank_RL_Control"
    USER_CREATOR = "DataScience Team"
    PROJECT_VERSION = "1.1.0"
    DESCRIPTION = "Global scalers for the RC Tank reinforcement learning model."
    NOTES = "Fitted on the complete training dataset from Q4 2025."

    print("=====================================================")
    print(f"  Starting Scaler Generation for: {PROJECT_NAME}")
    print("=====================================================")

    # Initialize the scaling reference class
    scaler_ref = GlobalScalingReference(
        data_dir=FOLDER_DATA,
        save_dir=FOLDER_SAVE_SCALE,
        dataset_name=DATASET_NAME,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        scaler_type=SCALER_TYPE,
        chunk_size=CHUNK_SIZE,
        user_creatr=USER_CREATOR,
        name_project=PROJECT_NAME,
        project_version=PROJECT_VERSION,
        description=DESCRIPTION,
        notes=NOTES
    )

    # Run the entire scaling process
    result = scaler_ref.run()

    # Print a clean summary of scaling information
    print("\n\n=====================================================")
    print("                      SUMMARY")
    print("=====================================================")
    print(f"✅ Process complete. Artifacts saved to: {result['zip']}")
    print("\n🎯 Scaling Parameters Overview:")
    print("--------------------------------")
    print(yaml.dump(result["metadata"]["scaling"], indent=2, allow_unicode=True))
