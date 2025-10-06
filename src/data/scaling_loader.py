import joblib
import yaml
import zipfile
import io
import warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class ScalingZipLoader:
    """
    A utility class for loading input/output scalers and metadata from a ZIP file.

    This class is designed to:
    - Load serialized `MinMaxScaler` objects (both input and output)
    - Optionally load metadata from a YAML file inside the ZIP archive
    - Provide methods to transform and inverse-transform data conveniently

    Typical use cases include:
    - Restoring normalization parameters for ML model inference
    - Reusing pre-fitted scalers for consistent preprocessing pipelines

    Attributes
    ----------
    zip_path : Path
        Path to the ZIP file.
    scaler_in : MinMaxScaler | None
        Scaler object for input data normalization.
    scaler_out : MinMaxScaler | None
        Scaler object for output (target) normalization.
    metadata : dict | None
        Optional YAML metadata dictionary describing scaling ranges or context.

    Examples
    --------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> loader = ScalingZipLoader(Path("RC_Tank_Env_Training2_scalers.zip"))
    >>> loader.is_loaded()
    True
    >>> X = np.array([[0.5], [1.5]])
    >>> X_scaled = loader.transform_input(X)
    >>> print(X_scaled)
    [[-0.5]
     [ 0.5]]
    >>> y_inverse = loader.inverse_output(X_scaled)
    >>> print(y_inverse)
    [[2.5]
     [7.5]]
    >>> loader.summary()
    📦 ScalingZipLoader Summary:
      input_feature_range: (-1, 1)
      output_feature_range: (0, 10)
      description: Test scaler zip file
    """

    def __init__(self, zip_path):
        """
        Initialize the ScalingZipLoader and automatically load its contents.

        Parameters
        ----------
        zip_path : str or Path
            Path to the ZIP file containing the scalers and metadata.

        Examples
        --------
        >>> loader = ScalingZipLoader("path/to/scaler_archive.zip")
        >>> print(loader.is_loaded())
        True
        """
        self.zip_path = Path(zip_path)
        self.scaler_in = None
        self.scaler_out = None
        self.metadata = None
        self._load_from_zip()  # Automatically loads all components upon initialization

    def _get_file_from_zip(self, zipf, name_keyword):
        """
        Find and read a file from the ZIP archive whose name contains the given keyword.

        Parameters
        ----------
        zipf : zipfile.ZipFile
            An opened ZIP file object.
        name_keyword : str
            Keyword used to identify the target file within the archive.

        Returns
        -------
        bytes
            File content in bytes.

        Raises
        ------
        FileNotFoundError
            If no file matching the keyword is found.

        Notes
        -----
        The method scans through the ZIP file entries and returns the first match found.

        Examples
        --------
        >>> with zipfile.ZipFile("archive.zip", "r") as zipf:
        ...     data = ScalingZipLoader._get_file_from_zip(self=None, zipf=zipf, name_keyword="input_scaler")
        """
        for info in zipf.infolist():
            # Match file name by substring keyword
            if name_keyword in info.filename:
                with zipf.open(info) as f:
                    return f.read()
        raise FileNotFoundError(f"❌ File containing '{name_keyword}' not found in ZIP")

    def _safe_load_yaml(self, zipf, filename="metadata.yaml"):
        """
        Safely load the YAML metadata file from the ZIP archive.

        Parameters
        ----------
        zipf : zipfile.ZipFile
            Opened ZIP file object.
        filename : str, default="metadata.yaml"
            Name of the YAML file expected inside the ZIP.

        Returns
        -------
        dict | None
            Parsed metadata dictionary if found, otherwise None.

        Examples
        --------
        >>> with zipfile.ZipFile("archive.zip", "r") as zipf:
        ...     meta = ScalingZipLoader._safe_load_yaml(self=None, zipf=zipf)
        """
        try:
            with zipf.open(filename) as f:
                return yaml.safe_load(f)
        except KeyError:
            # Warn the user if metadata is missing (not critical)
            warnings.warn(f"⚠️ {filename} not found in ZIP — metadata will be None", UserWarning)
            return None

    def _load_from_zip(self):
        """
        Load the input/output scalers and metadata from the ZIP file.

        Raises
        ------
        FileNotFoundError
            If the specified ZIP file path does not exist.

        Examples
        --------
        >>> loader = ScalingZipLoader("scaler_archive.zip")
        >>> loader._load_from_zip()  # Manually reload if needed
        """
        if not self.zip_path.exists():
            raise FileNotFoundError(f"❌ ZIP file not found at {self.zip_path}")

        with zipfile.ZipFile(self.zip_path, "r") as zipf:
            # Load serialized input/output scalers directly into memory
            self.scaler_in = joblib.load(io.BytesIO(self._get_file_from_zip(zipf, "input_scaler")))
            self.scaler_out = joblib.load(io.BytesIO(self._get_file_from_zip(zipf, "output_scaler")))
            # Attempt to load metadata (optional)
            self.metadata = self._safe_load_yaml(zipf)

    def transform_input(self, X):
        """
        Transform (normalize) input data using the loaded input scaler.

        Parameters
        ----------
        X : np.ndarray
            Input data array to be scaled.

        Returns
        -------
        np.ndarray
            Scaled input data array.

        Examples
        --------
        >>> X = np.array([[1.0], [2.0], [3.0]])
        >>> loader = ScalingZipLoader("scaler_archive.zip")
        >>> X_scaled = loader.transform_input(X)
        """
        return self.scaler_in.transform(X)

    def inverse_output(self, y_scaled):
        """
        Inverse-transform scaled output data using the output scaler.

        Parameters
        ----------
        y_scaled : np.ndarray
            Scaled output data to be inverse-transformed.

        Returns
        -------
        np.ndarray
            Original (unscaled) output data.

        Examples
        --------
        >>> y_scaled = np.array([[0.0], [0.5], [1.0]])
        >>> loader = ScalingZipLoader("scaler_archive.zip")
        >>> y_original = loader.inverse_output(y_scaled)
        """
        return self.scaler_out.inverse_transform(y_scaled)

    def is_loaded(self):
        """
        Check whether both scalers are successfully loaded.

        Returns
        -------
        bool
            True if both scalers exist, False otherwise.

        Examples
        --------
        >>> loader = ScalingZipLoader("scaler_archive.zip")
        >>> loader.is_loaded()
        True
        """
        return all([self.scaler_in is not None, self.scaler_out is not None])

    def summary(self):
        """
        Print a summary of loaded scalers and metadata.

        This is a human-readable representation of the loaded metadata
        that helps verify the content of the ZIP archive.

        Examples
        --------
        >>> loader = ScalingZipLoader("scaler_archive.zip")
        >>> loader.summary()
        📦 ScalingZipLoader Summary:
          input_feature_range: (-1, 1)
          output_feature_range: (0, 10)
          description: Test scaler zip file
        """
        print("📦 ScalingZipLoader Summary:")
        if self.metadata:
            for k, v in self.metadata.items():
                print(f"  {k}: {v}")
        else:
            print("  ⚠️ No metadata available")


# =====================================================
# 🔽 Test Section — Demonstration of usage
# =====================================================
if __name__ == "__main__":
    print("🧪 Starting ScalingZipLoader test ...")

    test_zip_path = Path(r"D:\Project_end\New_world\my_project\config\RC_Tank_Env_Training2_scalers.zip")

    # Check whether ZIP file already exists
    if test_zip_path.exists():
        print(f"📂 File already exists at: {test_zip_path}")
        choice = input("Would you like to (r) read the file or (w) overwrite it? [r/w]: ").strip().lower()
    else:
        choice = "w"

    # Option 1: Create new ZIP archive with test data
    if choice == "w":
        print("✏️ Creating a new ZIP file...")

        X = np.array([[0], [1], [2], [3], [4]], dtype=float)
        scaler_in = MinMaxScaler(feature_range=(-1, 1)).fit(X)
        scaler_out = MinMaxScaler(feature_range=(0, 10)).fit(X)
        metadata = {
            "input_feature_range": "(-1, 1)",
            "output_feature_range": "(0, 10)",
            "description": "Test scaler zip file",
        }

        with zipfile.ZipFile(test_zip_path, "w") as zipf:
            # Save both scalers in serialized form
            buffer_in = io.BytesIO()
            joblib.dump(scaler_in, buffer_in)
            zipf.writestr("input_scaler.pkl", buffer_in.getvalue())

            buffer_out = io.BytesIO()
            joblib.dump(scaler_out, buffer_out)
            zipf.writestr("output_scaler.pkl", buffer_out.getvalue())

            # Save YAML metadata
            zipf.writestr("metadata.yaml", yaml.safe_dump(metadata))

        print(f"✅ New ZIP file created at: {test_zip_path.resolve()}")

    elif choice == "r":
        print("📖 Reading existing ZIP file...")

    else:
        print("⚠️ Invalid choice — reading existing file instead")

    # Load ZIP using ScalingZipLoader
    loader = ScalingZipLoader(test_zip_path)

    # Display metadata summary
    loader.summary()

    # Demonstrate transform and inverse-transform behavior
    X_test = np.array([[1.5], [3.0]])
    X_scaled = loader.transform_input(X_test)
    y_inverse = loader.inverse_output(X_scaled)

    print("\n🔹 Example data transformation:")
    print(f"  Original data:\n{X_test}")
    print(f"  Scaled with input_scaler:\n{X_scaled}")
    print(f"  Inverse-transformed with output_scaler:\n{y_inverse}")

    print("\n🎉 Test completed successfully")
