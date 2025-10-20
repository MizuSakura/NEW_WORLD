# src/utils/scaling_zip_loader.py
"""
ScalingZipLoader (Final Auto-Align Version)
===========================================

Enhanced loader for trained scalers (.pkl or joblib) with automatic feature
alignment and compatibility across training, csv2pt, and inference stages.

New features:
- Supports flexible filename matching inside ZIP (input_scaler.pkl, scaler_in.pkl, etc.)
- Loads metadata from either `metadata.yaml` or `meta.yaml`
- Automatically aligns DataFrame columns to scaler feature order
- Preserves backward compatibility with `.meta`
- Silences sklearn feature name warnings safely
"""

import io
import yaml
import joblib
import zipfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ScalingZipLoader:
    """Load input/output scalers and metadata from a ZIP file."""

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        self.scaler_in = None
        self.scaler_out = None
        self.metadata = None  # Preferred attribute
        self.meta = None      # Backward-compatible alias

        if not self.zip_path.exists():
            raise FileNotFoundError(f"❌ ZIP file not found at: {self.zip_path}")

        self._load_from_zip()
        print(f"✅ Successfully loaded artifacts from: {self.zip_path.name}")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _load_from_zip(self):
        """Load scalers and metadata from ZIP archive with flexible matching."""
        with zipfile.ZipFile(self.zip_path, "r") as zipf:
            file_list = zipf.namelist()

            # Find scaler files flexibly
            input_scaler_name = next((f for f in file_list if "input" in f and f.endswith(".pkl")), None)
            output_scaler_name = next((f for f in file_list if "output" in f and f.endswith(".pkl")), None)

            if not input_scaler_name or not output_scaler_name:
                raise FileNotFoundError(f"❌ Could not find input/output scaler .pkl files inside {self.zip_path}")

            self.scaler_in = self._load_joblib_from_zip(zipf, input_scaler_name)
            self.scaler_out = self._load_joblib_from_zip(zipf, output_scaler_name)

            # Load metadata.yaml or meta.yaml
            meta_file = None
            for candidate in ["metadata.yaml", "meta.yaml"]:
                if candidate in file_list:
                    meta_file = candidate
                    break

            if meta_file:
                self.metadata = self._load_yaml_from_zip(zipf, meta_file)
                self.meta = self.metadata
            else:
                warnings.warn("⚠️ No metadata.yaml or meta.yaml found.", UserWarning)

    def _load_joblib_from_zip(self, zipf, filename):
        """Helper to load joblib-serialized object from ZIP."""
        with zipf.open(filename) as f:
            return joblib.load(io.BytesIO(f.read()))

    def _load_yaml_from_zip(self, zipf, filename):
        """Helper to safely load YAML metadata file."""
        with zipf.open(filename) as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Internal Alignment Helper
    # ------------------------------------------------------------------
    def _align_features(self, df: pd.DataFrame, scaler):
        """Ensure DataFrame columns match scaler training order."""
        if not isinstance(df, pd.DataFrame):
            return df  # skip if it's already ndarray

        if hasattr(scaler, "feature_names_in_"):
            scaler_features = list(scaler.feature_names_in_)
            missing = [f for f in scaler_features if f not in df.columns]
            if missing:
                raise ValueError(f"❌ Missing features in input: {missing}")

            # Reorder DataFrame columns to scaler feature order
            return df[scaler_features]
        elif self.metadata and "dataset" in self.metadata:
            # fallback: use metadata order
            expected_cols = self.metadata["dataset"].get("input_features", [])
            if expected_cols:
                available = [c for c in expected_cols if c in df.columns]
                return df[available]
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summary(self):
        """Display metadata summary."""
        print("\n" + "=" * 50)
        print(" " * 15 + "METADATA SUMMARY")
        print("=" * 50)
        if self.metadata:
            print(yaml.dump(self.metadata, allow_unicode=True, sort_keys=False, indent=2))
        else:
            print("⚠️ No metadata available.")
        print("=" * 50)

    def transform_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform input data with auto feature alignment."""
        if isinstance(X, pd.DataFrame):
            X_aligned = self._align_features(X, self.scaler_in)
            X_values = X_aligned.values
        else:
            X_values = np.asarray(X)

        # Silence sklearn feature name warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.scaler_in.transform(X_values)

    def inverse_output(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled output data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.scaler_out.inverse_transform(y_scaled)

    def is_loaded(self) -> bool:
        """Check if all artifacts are loaded successfully."""
        return all([self.scaler_in, self.scaler_out])


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    ZIP_FILE_PATH = Path(r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip")
    print(f"🧪 Attempting to load scaling artifacts from: {ZIP_FILE_PATH}")

    try:
        loader = ScalingZipLoader(ZIP_FILE_PATH)
        loader.summary()

        if loader.is_loaded():
            import pandas as pd
            input_cols = loader.metadata["dataset"]["input_features"]
            X_test = pd.DataFrame([[12, 15]], columns=input_cols)
            y_test_scaled = np.array([[0.5]])

            X_scaled = loader.transform_input(X_test)
            y_inverse = loader.inverse_output(y_test_scaled)

            print("\n🔹 Example Data Transformation:")
            print(f"  Original Input:\n{X_test}")
            print(f"  Scaled Input:\n{X_scaled}")
            print("-" * 30)
            print(f"  Scaled Output:\n{y_test_scaled}")
            print(f"  Inverse Output:\n{y_inverse}")

        print("\n🎉 Test completed successfully.")

    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
