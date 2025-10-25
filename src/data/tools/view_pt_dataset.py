# ======================================================================
# 🔍 VIEW .PT DATASET FROM ZIP PACKAGE (WITH SCALERS, MULTI-FEATURE, MARKER)
# ======================================================================
#my_project\src\data\tools\view_pt_dataset.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import joblib
import io
from pathlib import Path

# ==========================================================
# 🔹 Dataset Viewer Class
# ==========================================================
class PTDatasetViewer:
    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"❌ Zip file not found: {self.zip_path}")
        self.X = None
        self.y = None
        self.scaler_input = None

    def load_pt(self, pt_index=0):
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
            if not pt_files:
                raise ValueError("❌ No .pt files found in the zip package.")
            print(f"✅ Found {len(pt_files)} .pt files in zip.")
            print(f"📄 Loading: {pt_files[pt_index]}")
            with zf.open(pt_files[pt_index]) as f:
                buffer = io.BytesIO(f.read())
                data = torch.load(buffer, map_location="cpu")
            self.X = data["X"].numpy()
            self.y = data["y"].numpy()
        print(f"✅ Loaded X: {self.X.shape}, y: {self.y.shape}")

    def load_scaler(self):
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            if "input_scaler.pkl" in zf.namelist():
                with zf.open("input_scaler.pkl") as f:
                    self.scaler_input = joblib.load(io.BytesIO(f.read()))
                print("✅ Loaded input scaler from zip.")
            else:
                print("⚠️ 'input_scaler.pkl' not found in zip. Skipping inverse scaling.")

    def _select_features(self, feature_idx):
        """Normalize feature index input: int, list, or slice"""
        if isinstance(feature_idx, int):
            feature_idx = [feature_idx]
        elif isinstance(feature_idx, slice):
            feature_idx = list(range(*feature_idx.indices(self.X.shape[2])))
        return feature_idx

    def plot_sequence(self, seq_idx=0, feature_idx=0, inverse_scale=False):
        feature_idx = self._select_features(feature_idx)
        seq = self.X[seq_idx]
        if inverse_scale and self.scaler_input is not None:
            seq = self.scaler_input.inverse_transform(seq)

        plt.figure(figsize=(8,4))
        for f_idx in feature_idx:
            plt.plot(seq[:, f_idx], '-o', label=f"Feature {f_idx}", markersize=4)
        title = "Inverse-scaled" if inverse_scale else "Scaled"
        plt.title(f"{title} Sequence View (Seq {seq_idx})")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_multiple_sequences(self, num_show=5, feature_idx=0, inverse_scale=False, sliding=True, y_offset=0.0):
        """
        Plot multiple sequences with optional sliding window and y-offset for better visibility.

        Args:
            num_show (int): Number of sequences to show.
            feature_idx (int/list/slice): Feature index or indices to plot.
            inverse_scale (bool): Whether to inverse scale the sequences.
            sliding (bool): If True, x-axis shifts by sequence index (sliding window).
            y_offset (float): Vertical shift applied per sequence to separate overlapping lines.
        """
        feature_idx = self._select_features(feature_idx)
        plt.figure(figsize=(10,5))

        for i in range(min(num_show, len(self.X))):
            seq = self.X[i].copy()
            if inverse_scale and self.scaler_input is not None:
                seq = self.scaler_input.inverse_transform(seq)

            seq_len = seq.shape[0]
            # กำหนด x-axis
            if sliding:
                x_vals = np.arange(i, i + seq_len)  # sliding window: shift by i
            else:
                x_vals = np.arange(seq_len)

            for f_idx in feature_idx:
                plt.plot(x_vals, seq[:, f_idx] + i*y_offset, '-o', label=f"Seq {i} - Feature {f_idx}", markersize=3, alpha=0.6)

        title = "Inverse-scaled" if inverse_scale else "Scaled"
        plt.title(f"First {num_show} Sequences - {title} (Sliding Window: {sliding})")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_concatenated_sequences(self, num_concat=20, feature_idx=0, inverse_scale=False, show_y=True):
        """
        Plot concatenated sequences for input features and optionally the target y.

        Args:
            num_concat (int): Number of sequences to concatenate.
            feature_idx (int/list/slice): Input feature index/indices to plot.
            inverse_scale (bool): Whether to inverse scale the input features (X) and target y.
            show_y (bool): If True, plot target y as well.
        """
        feature_idx = self._select_features(feature_idx)
        X_concat = self.X[:num_concat]
        X_long = X_concat.reshape(-1, X_concat.shape[-1])

        y_long = None
        if show_y and self.y is not None:
            y_concat = self.y[:num_concat].reshape(-1, self.y.shape[-1])
            y_long = y_concat.copy()

        # Apply inverse scaling if available
        if inverse_scale:
            if self.scaler_input is not None:
                X_long = self.scaler_input.inverse_transform(X_long)
            if show_y and hasattr(self, "scaler_y") and self.scaler_y is not None and y_long is not None:
                y_long = self.scaler_y.inverse_transform(y_long)

        plt.figure(figsize=(12,4))

        # Plot input features
        for f_idx in feature_idx:
            plt.plot(X_long[:, f_idx], '-o', label=f"Feature {f_idx}", markersize=2, alpha=0.7)

        # Plot target y
        if show_y and y_long is not None:
            plt.plot(y_long, '-x', label="Target y", markersize=3, color='red', alpha=0.8)

        title = "Inverse-scaled" if inverse_scale else "Scaled"
        plt.title(f"Concatenated {num_concat} sequences - {title}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ==========================================================
# 🔹 Main Script
# ==========================================================
if __name__ == "__main__":
    # ================= CONFIG =================
    ZIP_PATH = Path(r"D:\Project_end\New_world\my_project\data\processed\test2_dataset_package.zip")
    SEQ_IDX = 0
    FEATURE_IDX = [0,1]     # ✅ Multi-feature support
    NUM_SHOW = 5
    NUM_CONCAT = 20
    SHOW_ORIGINAL = True
    # =========================================

    viewer = PTDatasetViewer(ZIP_PATH)
    viewer.load_pt(pt_index=0)
    if SHOW_ORIGINAL:
        viewer.load_scaler()

    # Single sequence
    viewer.plot_sequence(seq_idx=SEQ_IDX, feature_idx=FEATURE_IDX, inverse_scale=SHOW_ORIGINAL)

    # Multiple sequences
    viewer.plot_multiple_sequences(num_show=NUM_SHOW, feature_idx=FEATURE_IDX, inverse_scale=SHOW_ORIGINAL)

    # Concatenated sequences
    viewer.plot_concatenated_sequences(num_concat=NUM_CONCAT, feature_idx=FEATURE_IDX, inverse_scale=SHOW_ORIGINAL)
