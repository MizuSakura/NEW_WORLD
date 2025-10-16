# src/utils/scaling_zip_loader.py
"""
ScalingZipLoader (Updated)
==========================

Enhanced loader for trained scalers with automatic feature validation
and mapping based on metadata.

- Automatically selects correct input columns according to metadata.
- Checks feature mismatch before scaling.
- Provides clear error messages for debugging.
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
    """
    Load input/output scalers and metadata from a ZIP file.

    Automatically validates input features against the metadata to prevent
    feature mismatch errors.
    """

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        self.scaler_in = None
        self.scaler_out = None
        self.metadata = None

        if not self.zip_path.exists():
            raise FileNotFoundError(f"❌ ZIP file not found at: {self.zip_path}")

        self._load_from_zip()
        print(f"✅ Successfully loaded artifacts from: {self.zip_path.name}")

    # ---------------------------------------------------------------------
    # Internal helper methods
    # ---------------------------------------------------------------------
    def _load_from_zip(self):
        """Load scalers and metadata from ZIP archive."""
        with zipfile.ZipFile(self.zip_path, "r") as zipf:
            self.scaler_in = self._load_joblib_from_zip(zipf, "input_scaler.pkl")
            self.scaler_out = self._load_joblib_from_zip(zipf, "output_scaler.pkl")
            self.metadata = self._load_yaml_from_zip(zipf, "metadata.yaml")

    def _load_joblib_from_zip(self, zipf, filename):
        """Helper to load joblib-serialized object from ZIP."""
        try:
            with zipf.open(filename) as f:
                return joblib.load(io.BytesIO(f.read()))
        except KeyError:
            raise FileNotFoundError(f"❌ '{filename}' not found in ZIP.")

    def _load_yaml_from_zip(self, zipf, filename):
        """Helper to load YAML files safely."""
        try:
            with zipf.open(filename) as f:
                return yaml.safe_load(f)
        except KeyError:
            warnings.warn(f"⚠️ '{filename}' not found in ZIP.", UserWarning)
            return None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
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
        """
        Transform input data with automatic feature mapping and validation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data. If DataFrame, column names are matched against metadata.

        Returns
        -------
        np.ndarray
            Scaled input data.

        Raises
        ------
        ValueError
            If input features do not match scaler's expected features.
        """
        if isinstance(X, pd.DataFrame):
            # Map DataFrame columns according to metadata
            expected_cols = self.metadata["dataset"]["input_features"]
            missing = [c for c in expected_cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"❌ Missing expected input columns: {missing}"
                )
            X_mapped = X[expected_cols].values
        else:
            X_mapped = np.asarray(X)

        # Validate feature count
        expected_features = self.scaler_in.n_features_in_
        if X_mapped.shape[1] != expected_features:
            raise ValueError(
                f"❌ Input feature mismatch: expected {expected_features} features "
                f"({self.metadata['dataset']['input_features']}), but got {X_mapped.shape[1]}"
            )

        return self.scaler_in.transform(X_mapped)

    def inverse_output(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled output data."""
        return self.scaler_out.inverse_transform(y_scaled)

    def is_loaded(self) -> bool:
        """Check if all artifacts are loaded successfully."""
        return all([self.scaler_in, self.scaler_out, self.metadata])


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ZIP_FILE_PATH = Path(r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip")
    print(f"🧪 Attempting to load scaling artifacts from: {ZIP_FILE_PATH}")

    try:
        loader = ScalingZipLoader(ZIP_FILE_PATH)
        loader.summary()

        if loader.is_loaded():
            # Test with DataFrame input using metadata columns
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
