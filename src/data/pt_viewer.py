# src/data/preview_pt.py
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

        # โหลด Scaler
        self.scaling_loader = ScalingZipLoader(scale_path)
        self.scaler_in = self.scaling_loader.scaler_in
        self.scaler_out = self.scaling_loader.scaler_out

        # โหลดข้อมูล
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
            # scaler_in expects 2D array with features as columns. X_sample is (window, n_features)
            X_inv = self.scaler_in.inverse_transform(X_sample)
            # y_sample is (n_outputs,)
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

    # ----------------------------------------------------------------------
    def compare_with_csv(self, seq_index=0, save_fig=None):
        self._check_index(seq_index)
        X_sample = self.X_pt[seq_index]
        y_sample = self.y_pt[seq_index]

        X_inv = self.scaler_in.inverse_transform(X_sample)
        y_inv = self.scaler_out.inverse_transform(y_sample.reshape(1, -1))[0]

        # Validate CSV columns exist
        for c in self.input_col + self.output_col:
            if c not in self.data_csv.columns:
                raise KeyError(f"Column '{c}' not found in CSV.")

        # เรียกดูช่วงจาก CSV ให้ตรงกับตำแหน่ง sequence
        # สมมติว่าทำ sliding window sequentially: sequence i uses CSV rows [i : i+window]
        window = X_inv.shape[0]
        start_idx_csv = seq_index
        end_idx_csv_input = start_idx_csv + window

        # bounds check CSV
        if end_idx_csv_input >= len(self.data_csv):
            raise IndexError("CSV does not contain enough rows for this sequence index/window.")

        df_window_input = self.data_csv.iloc[start_idx_csv:end_idx_csv_input].reset_index(drop=True)
        df_window_output = self.data_csv.iloc[end_idx_csv_input:end_idx_csv_input + 1].reset_index(drop=True)

        print("\n=== [COMPARE CSV vs DECODED .PT] ===")
        print(f"--- Sequence index: {seq_index} ---")
        print("CSV INPUT (original) (first 10 rows):")
        print(df_window_input[self.input_col].head(10))
        print("\nDecoded INPUT (from .pt inverse) (first 10 rows):")
        print(pd.DataFrame(X_inv, columns=self.input_col).head(10))
        print("-" * 50)
        print("CSV OUTPUT (original):")
        print(df_window_output[self.output_col])
        print("\nDecoded OUTPUT (from .pt inverse):")
        print(pd.DataFrame([y_inv], columns=self.output_col))

        # --- Plot Input ---
        plt.figure(figsize=(12, 5))
        plt.title(f"Sequence #{seq_index} - Input Comparison", fontsize=14)

        t = np.arange(len(df_window_input))
        # plot CSV original (as line with markers)
        for col in self.input_col:
            plt.plot(t, df_window_input[col].values, linestyle="--", marker="o", alpha=0.6,
                     label=f"CSV {col}", zorder=1)
        # plot decoded sequence (may be multi-dim; plot each input column)
        df_decoded = pd.DataFrame(X_inv, columns=self.input_col)
        for col in df_decoded.columns:
            plt.plot(t, df_decoded[col].values, linestyle="-", marker=".", linewidth=2,
                     label=f"Decoded {col}", zorder=2)

        plt.xlabel("Time Step in Sequence")
        plt.ylabel("Input Value")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_fig:
            plt.savefig(save_fig + f"_input_seq{seq_index}.png", bbox_inches="tight")
        plt.show()

        # --- Plot Output (show as points because it's many-to-one) ---
        plt.figure(figsize=(6, 4))
        plt.title(f"Sequence #{seq_index} - Output Comparison", fontsize=14)

        # CSV true
        csv_out_vals = df_window_output[self.output_col].iloc[0].values
        # decoded
        decoded_vals = y_inv

        x_pos = np.arange(len(self.output_col))
        plt.scatter(x_pos - 0.05, csv_out_vals, label="CSV True", s=140, marker="o", edgecolors="black", zorder=2)
        plt.scatter(x_pos + 0.05, decoded_vals, label="Decoded (.pt)", s=110, marker="X", zorder=3)

        plt.xticks(x_pos, self.output_col, rotation=45)
        plt.ylabel("Output Value")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        if save_fig:
            plt.savefig(save_fig + f"_output_seq{seq_index}.png", bbox_inches="tight")
        plt.show()

    # ----------------------------------------------------------------------
    def compare_many_with_csv(self, start_idx=0, num_samples=5, show_outputs_as_points=True, save_fig=None):
        """
        เปรียบเทียบหลาย sequence กับ CSV
        - start_idx: sequence index ใน .pt ที่จะเริ่ม (สมมติ sliding window)
        - num_samples: จำนวน sequence ที่จะแสดง (ต่อเนื่อง)
        - show_outputs_as_points: ถ้า True จะแสดง output เป็นจุด (เหมาะกับ many-to-one)
        """
        n_total = len(self.X_pt)
        if start_idx < 0 or start_idx >= n_total:
            raise IndexError("start_idx out of range")
        end_idx = min(start_idx + num_samples, n_total)
        indices = list(range(start_idx, end_idx))
        window_size = self.X_pt.shape[1]  # input window length
        n_inputs = len(self.input_col)

        # --- Prepare CSV range to plot background original signal ---
        # We'll build an overall axis in CSV index space
        # For sequences start_idx..end_idx-1, earliest CSV idx used = start_idx
        # latest CSV idx used = (end_idx-1) + window_size - 1  (last input row of last sequence)
        csv_first = start_idx
        csv_last = (end_idx - 1) + window_size  # +1 because output sits at next step
        if csv_last > len(self.data_csv):
            csv_last = len(self.data_csv)
        csv_slice = self.data_csv.iloc[csv_first:csv_last].reset_index(drop=True)

        # --- Plot inputs: CSV background + each decoded sequence shifted into proper CSV index space ---
        plt.figure(figsize=(15, 6))
        plt.title(f"Input Comparison: Decoded Sequences vs. Original CSV (Samples {start_idx}–{end_idx-1})", fontsize=14)

        # plot CSV original for each input column (as faint line)
        global_x = np.arange(csv_first, csv_first + len(csv_slice))
        for col in self.input_col:
            plt.plot(global_x, csv_slice[col].values, label=f"CSV {col}", linestyle="--", alpha=0.6, zorder=1)

        # plot each decoded sequence in its CSV-aligned position
        for i in indices:
            X_inv = self.scaler_in.inverse_transform(self.X_pt[i])  # shape (window, n_inputs)
            seq_x = np.arange(i, i + X_inv.shape[0])  # CSV index positions
            # for multi-col input, plot each column offset (same color group)
            for col_idx, col in enumerate(self.input_col):
                plt.plot(seq_x, X_inv[:, col_idx], marker='.', linewidth=2, alpha=0.95, zorder=2)

        # legend helper
        plt.xlabel("Overall Time Step (CSV index)")
        plt.ylabel("Input Value")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend([f"CSV {c}" for c in self.input_col] + [f"Decoded sequences ({len(indices)})"], loc="upper right")
        if save_fig:
            plt.savefig(save_fig + f"_many_input_{start_idx}_{end_idx-1}.png", bbox_inches="tight")
        plt.show()

        # --- Plot outputs ---
        # decoded outputs
        y_inv_all = self.scaler_out.inverse_transform(self.y_pt[indices])
        # CSV outputs expected at indices + window_size
        output_start_idx = start_idx + window_size
        output_end_idx = output_start_idx + len(indices)
        csv_outputs_df = self.data_csv.iloc[output_start_idx:output_end_idx].reset_index(drop=True)

        if csv_outputs_df.shape[0] < len(indices):
            # partial overlap: warn but still plot available points
            print("Warning: CSV has fewer rows than requested output positions; plotting available overlap.")

        # If single-output, treat accordingly
        plt.figure(figsize=(10, 5))
        plt.title(f"Output Comparison (Samples {start_idx}–{end_idx-1})", fontsize=14)

        seq_positions = np.arange(start_idx, start_idx + len(indices))
        # plot CSV outputs trend if available
        for idx_out, col in enumerate(self.output_col):
            csv_vals = csv_outputs_df[col].values if col in csv_outputs_df.columns else np.array([])
            # plot trend line (if length matches)
            if len(csv_vals) == len(indices):
                plt.plot(seq_positions, csv_vals, label=f"CSV {col} (True)", linestyle="--", alpha=0.7, zorder=1)
                plt.scatter(seq_positions, csv_vals, label=None, color='orange', marker='o', s=80, edgecolors='black', zorder=2)

            # decoded outputs for this column
            decoded_col_vals = y_inv_all[:, idx_out]
            if show_outputs_as_points:
                plt.scatter(seq_positions, decoded_col_vals, label=f"Decoded {col}", marker='X', s=90, zorder=3)
            else:
                plt.plot(seq_positions, decoded_col_vals, label=f"Decoded {col}", linestyle='-', marker='.', zorder=3)

        plt.xlabel("Sequence Index")
        plt.ylabel("Output Value")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        if save_fig:
            plt.savefig(save_fig + f"_many_output_{start_idx}_{end_idx-1}.png", bbox_inches="tight")
        plt.show()

    def plot_grid_samples_formal(self, indices=None, inverse=True):
       
        if indices is None:
            indices = [0]
        n_samples = len(indices)
        
        # figsize ปรับตามจำนวน sample
        fig_width = 10
        fig_height = 2 * n_samples
        
        # --- CODE EDIT START ---
        # ใช้ constrained_layout=True เพื่อจัดการระยะห่างอัตโนมัติ
        fig, axes = plt.subplots(n_samples, 2, 
                                 figsize=(fig_width, fig_height), 
                                 constrained_layout=True)
        # --- CODE EDIT END ---
        
        # axes เป็น 2D array เสมอ
        if n_samples == 1:
            axes = np.array([axes])

        # font ขนาดมาตรฐาน
        title_font = {'fontsize': 8,
                      'fontweight': 'light',
                      'verticalalignment': 'bottom',
                      'horizontalalignment': 'center'}
        label_font = {'fontsize': 8,
                      'fontweight': 'normal'}

        for row_idx, sample_idx in enumerate(indices):
            self._check_index(sample_idx)
            X_sample = self.X_pt[sample_idx]
            y_sample = self.y_pt[sample_idx]

            if inverse:
                X_plot = self.scaler_in.inverse_transform(X_sample)
                y_plot = self.scaler_out.inverse_transform(y_sample.reshape(1, -1))[0]
            else:
                X_plot = X_sample
                y_plot = y_sample

            # --- Input plot ---
            ax_in = axes[row_idx, 0]
            for i, col_name in enumerate(self.input_col):
                ax_in.plot(range(len(X_plot)), X_plot[:, i], marker='.', label=col_name, linewidth=2)
            ax_in.set_title(f"Sample #{sample_idx} - Input", **title_font)
            ax_in.set_xlabel("Time Step", **label_font)
            ax_in.set_ylabel("Input Value", **label_font)
            ax_in.grid(True, linestyle='--', linewidth=0.5)
            ax_in.legend(fontsize=10)

            # --- Output plot ---
            ax_out = axes[row_idx, 1]
            x_pos = np.arange(len(self.output_col))
            ax_out.scatter(x_pos, y_plot, color='dodgerblue', marker='X', s=120, label='Output')
            ax_out.set_xticks(x_pos)
            ax_out.set_xticklabels(self.output_col, rotation=0, fontsize=8, horizontalalignment='center')
            ax_out.set_title(f"Sample #{sample_idx} - Output", **title_font)
            ax_out.set_ylabel("Output Value", **label_font)
            ax_out.grid(True, linestyle='--', linewidth=0.5)
            ax_out.legend(fontsize=10)
        
        # ไม่จำเป็นต้องใช้ tight_layout() เมื่อใช้ constrained_layout=True
        # plt.tight_layout() 
        plt.show()
        
if __name__ == "__main__":

    # --- กำหนดพาธไฟล์ที่ต้องการทดสอบ ---
    pt_path = r"D:\Project_end\New_world\my_project\data\processed\pwm_duty_0.56_freq_0.01_pwm.pt"
    csv_path = r"D:\Project_end\New_world\my_project\data\raw\pwm_duty_0.56_freq_0.01_pwm.csv"
    scale_path = r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip"

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
    # viewer.show_samples(indices=[0, 1, 2, 3], inverse=True)

    # viewer.compare_with_csv(seq_index=2)

    # viewer.compare_many_with_csv(start_idx=0, num_samples=10)
    
    viewer.plot_grid_samples_formal(indices=[0,1,2], inverse=True)