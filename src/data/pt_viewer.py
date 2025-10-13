from pathlib import Path
import torch
import numpy as np
import pandas as pd
from src.data.scaling_loader import ScalingZipLoader
import matplotlib.pyplot as plt
import math


class PreviewPTData:
 

    def __init__(self, pt_path, csv_path, scale_path, input_col, output_col):
        self.pt_path = Path(pt_path)
        self.csv_path = Path(csv_path)
        self.scale_path = Path(scale_path)

        # Normalize column inputs to lists
        if isinstance(input_col, str):
            input_col = [input_col]
        if isinstance(output_col, str):
            output_col = [output_col]
        self.input_col = list(input_col)
        self.output_col = list(output_col)

        self.scaling_loader = ScalingZipLoader(scale_path)
        self.scaler_in = self.scaling_loader.scaler_in
        self.scaler_out = self.scaling_loader.scaler_out

        self.data_pt = torch.load(pt_path)
        self.data_csv = pd.read_csv(csv_path)

        # Expect keys "X" and "y" inside the .pt
        self.X_pt = self.data_pt["X"].numpy()  # shape: (num_samples, window, n_features)
        self.y_pt = self.data_pt["y"].numpy()  # shape: (num_samples, n_outputs) or (num_samples,)

        # ensure y is 2D for scaler compatibility
        if self.y_pt.ndim == 1:
            self.y_pt = self.y_pt.reshape(-1, 1)

    # ----------------------------------------------------------------------
    def _check_index(self, index):
        n = len(self.X_pt)
        if index < 0 or index >= n:
            raise IndexError(f"Index {index} out of range for .pt (0..{n-1})")

    # ----------------------------------------------------------------------
    def show_sample(self, index=0, inverse=True):
        self._check_index(index)
        X_sample = self.X_pt[index]
        y_sample = self.y_pt[index]

        print("=" * 60)
        print(f"[SAMPLE #{index}]")
        print(f"Input shape : {X_sample.shape}")
        print(f"Output shape: {y_sample.shape}")
        print("-" * 60)
        print("Scaled Input (first 5 rows):\n", X_sample[:5])
        print("Scaled Output:\n", y_sample)

        if inverse:
            X_inv = self.scaler_in.inverse_transform(X_sample)
            y_inv = self.scaler_out.inverse_transform(y_sample.reshape(1, -1))[0]
            print("-" * 60)
            print("Inverse Input (first 5 rows):\n", X_inv[:5])
            print("Inverse Output:\n", y_inv)
        print("=" * 60)

    # ----------------------------------------------------------------------
    def show_samples(self, indices=None, inverse=True):
        if indices is None:
            indices = [0]
        print(f"\n=== Showing {len(indices)} samples ===")
        for i in indices:
            self.show_sample(index=i, inverse=inverse)


    # --- NEW FUNCTION START ---
    def plot_grid_comparison_formal(self, indices=None):
        if indices is None:
            indices = [0]
        n_samples = len(indices)
        
        fig_width = 12
        fig_height = 4 * n_samples
        
        fig, axes = plt.subplots(n_samples, 2, 
                                 figsize=(fig_width, fig_height), 
                                 constrained_layout=True)
        
        if n_samples == 1:
            axes = np.array([axes])

        title_font = {'fontsize': 10, 'fontweight': 'normal'}
        label_font = {'fontsize': 9}

        for row_idx, sample_idx in enumerate(indices):
            self._check_index(sample_idx)
            
            # --- Get Decoded Data from .pt ---
            X_sample_pt = self.X_pt[sample_idx]
            y_sample_pt = self.y_pt[sample_idx]
            X_decoded = self.scaler_in.inverse_transform(X_sample_pt)
            y_decoded = self.scaler_out.inverse_transform(y_sample_pt.reshape(1, -1))[0]

            # --- Get Original Data from .csv ---
            window_size = X_decoded.shape[0]
            start_idx_csv = sample_idx
            end_idx_csv_input = start_idx_csv + window_size
            
            # Bounds check for CSV
            if end_idx_csv_input + 1 > len(self.data_csv):
                print(f"Warning: Skipping sample #{sample_idx} due to insufficient data in CSV.")
                continue

            df_input_csv = self.data_csv.iloc[start_idx_csv:end_idx_csv_input]
            df_output_csv = self.data_csv.iloc[end_idx_csv_input:end_idx_csv_input + 1]
            y_csv = df_output_csv[self.output_col].values[0]

            # --- Input Comparison Plot ---
            ax_in = axes[row_idx, 0]
            t = np.arange(window_size)
            # Plot decoded .pt data
            for i, col_name in enumerate(self.input_col):
                ax_in.plot(t, X_decoded[:, i], marker='.', label=f"Decoded (.pt)", linewidth=2, zorder=2)
            # Plot original CSV data
            for i, col_name in enumerate(self.input_col):
                 ax_in.plot(t, df_input_csv[col_name].values, linestyle="--", marker="o", markersize=4, label=f"Original (CSV)", alpha=0.7, zorder=1)

            ax_in.set_title(f"Sample #{sample_idx} - Input Comparison", **title_font)
            ax_in.set_xlabel("Time Step", **label_font)
            ax_in.set_ylabel("Value", **label_font)
            ax_in.grid(True, linestyle='--', linewidth=0.5)
            ax_in.legend()

            # --- Output Comparison Plot ---
            ax_out = axes[row_idx, 1]
            x_pos = np.arange(len(self.output_col))
            # Plot original CSV data
            ax_out.scatter(x_pos - 0.05, y_csv, color='darkorange', marker='o', s=150, label='Original (CSV)', edgecolors="black", zorder=2)
            # Plot decoded .pt data
            ax_out.scatter(x_pos + 0.05, y_decoded, color='dodgerblue', marker='X', s=120, label='Decoded (.pt)', zorder=3)

            ax_out.set_xticks(x_pos)
            ax_out.set_xticklabels(self.output_col, rotation=0, fontsize=9)
            ax_out.set_title(f"Sample #{sample_idx} - Output Comparison", **title_font)
            ax_out.set_ylabel("Value", **label_font)
            ax_out.grid(True, linestyle='--', linewidth=0.5)
            ax_out.legend()
        
        plt.show()
    # --- NEW FUNCTION END ---
    
if __name__ == "__main__":
    

    # --- กำหนดพาธไฟล์ที่ต้องการทดสอบ ---
    pt_path = r"D:\Project_end\New_world\my_project\data\processed\pwm_duty_0.56_freq_0.01_pwm.pt"
    csv_path = r"D:\Project_end\New_world\my_project\data\raw\pwm_duty_0.56_freq_0.01_pwm.csv"
    scale_path = r"D:\Project_end\New_world\my_project\config\Test_scale_test_pt_scalers.zip"

    # --- ชื่อคอลัมน์ input/output ---
    input_col = ["DATA_INPUT"]
    output_col = ["DATA_OUTPUT"]

    # --- สร้าง instance ---
    viewer = PreviewPTData(
        pt_path=pt_path,
        csv_path=csv_path,
        scale_path=scale_path,
        input_col=input_col,
        output_col=output_col
    )

    # === ตัวอย่างการใช้งาน ===
    # ให้เรียกใช้ฟังก์ชันใหม่ plot_grid_comparison_formal
    viewer.plot_grid_comparison_formal(indices=[0, 1, 2])
