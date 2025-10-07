# src/utils/scaling_zip_loader.py
"""
ScalingZipLoader
================

Utility class for loading trained scalers and metadata from a ZIP archive.

This tool restores serialized `MinMaxScaler` or `StandardScaler` instances,
along with dataset metadata (feature names, data ranges, etc.)
produced by the `GlobalScalingReference` pipeline.

Main Features
-------------
- Load `input_scaler.pkl`, `output_scaler.pkl`, and `metadata.yaml` directly from a ZIP file.
- Perform consistent scaling and inverse scaling for inference.
- Quickly inspect stored metadata for debugging or validation.

Example
-------
>>> from utils.scaling_zip_loader import ScalingZipLoader
>>> loader = ScalingZipLoader("config/RC_Tank_Env_scalers.zip")
>>> loader.summary()
>>> X_scaled = loader.transform_input([[12.0, 15.5]])
>>> y_original = loader.inverse_output([[0.8]])
>>> print(X_scaled, y_original)
"""

import io
import yaml
import joblib
import zipfile
import warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ScalingZipLoader:
    """
    Load input/output scalers and metadata from a ZIP file.

    The ZIP archive should contain:
    - `input_scaler.pkl`  : Input data scaler
    - `output_scaler.pkl` : Output data scaler
    - `metadata.yaml`     : Metadata (feature info, min/max range, etc.)

    Parameters
    ----------
    zip_path : str or Path
        Path to the ZIP file containing scalers and metadata.

    Attributes
    ----------
    scaler_in : sklearn.preprocessing.BaseEstimator
        Loaded input scaler.
    scaler_out : sklearn.preprocessing.BaseEstimator
        Loaded output scaler.
    metadata : dict
        Metadata dictionary loaded from YAML.

    Example
    -------
    >>> loader = ScalingZipLoader("config/RC_Tank_Env_scalers.zip")
    >>> loader.summary()
    >>> loader.transform_input([[10, 20]])
    array([[0.25, 0.70]])
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
        """Load all objects (scalers and metadata) from ZIP archive."""
        with zipfile.ZipFile(self.zip_path, "r") as zipf:
            self.scaler_in = self._load_joblib_from_zip(zipf, "input_scaler.pkl")
            self.scaler_out = self._load_joblib_from_zip(zipf, "output_scaler.pkl")
            self.metadata = self._load_yaml_from_zip(zipf, "metadata.yaml")

    def _load_joblib_from_zip(self, zipf, filename):
        """Helper to load joblib-serialized objects from ZIP."""
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
        """
        Display formatted metadata information.

        Example
        -------
        >>> loader = ScalingZipLoader("config/RC_Tank_Env_scalers.zip")
        >>> loader.summary()
        dataset_name: RC_Tank_Env_Training
        input_features: ['DATA_INPUT']
        output_features: ['DATA_OUTPUT']
        """
        print("\n" + "=" * 50)
        print(" " * 15 + "METADATA SUMMARY")
        print("=" * 50)
        if self.metadata:
            print(yaml.dump(self.metadata, allow_unicode=True, sort_keys=False, indent=2))
        else:
            print("⚠️ No metadata available.")
        print("=" * 50)

    def transform_input(self, X: np.ndarray) -> np.ndarray:
        """
        Transform (scale) raw input data.

        Parameters
        ----------
        X : np.ndarray
            Raw input data, shape = (n_samples, n_input_features)

        Returns
        -------
        np.ndarray
            Scaled input data.

        Example
        -------
        >>> X = np.array([[12.0, 15.5], [24.0, 23.9]])
        >>> X_scaled = loader.transform_input(X)
        """
        return self.scaler_in.transform(X)

    def inverse_output(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse-transform scaled output data to the original range.

        Parameters
        ----------
        y_scaled : np.ndarray
            Scaled output, shape = (n_samples, n_output_features)

        Returns
        -------
        np.ndarray
            Original output values.

        Example
        -------
        >>> y_scaled = np.array([[0.5], [1.0]])
        >>> y_original = loader.inverse_output(y_scaled)
        """
        return self.scaler_out.inverse_transform(y_scaled)

    def is_loaded(self) -> bool:
        """
        Check whether both scalers and metadata were successfully loaded.

        Returns
        -------
        bool
            True if all artifacts are loaded, False otherwise.

        Example
        -------
        >>> loader.is_loaded()
        True
        """
        return all([self.scaler_in, self.scaler_out, self.metadata])


# ---------------------------------------------------------------------
# Example Standalone Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ZIP_FILE_PATH = Path(
        r"D:\Project_end\New_world\my_project\config\RC_Tank_Env_scalers.zip"
    )
    print(f"🧪 Attempting to load scaling artifacts from: {ZIP_FILE_PATH}")

    try:
        loader = ScalingZipLoader(ZIP_FILE_PATH)
        loader.summary()

        if loader.is_loaded():
            X_test = np.array([[12.0, 15.5], [24.0, 23.9]])
            y_test_scaled = np.array([[0.5], [1.0]])

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
