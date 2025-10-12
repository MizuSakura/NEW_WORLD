# src/data/preview_pt.py
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from src.data.scaling_loader import ScalingZipLoader
import matplotlib.pyplot as plt


class PreviewPTData:
    """
    ดูข้อมูลตัวอย่างจากไฟล์ .pt ที่ถูกแปลงจาก CSV (เวอร์ชันปรับปรุงการแสดงผล)
    ===================================================
    - แสดง input/output แบบ scaled และ inverse
    - เปรียบเทียบกับ CSV ต้นฉบับ พร้อม Plot ที่ชัดเจนขึ้น
    - รองรับ many samples พร้อมกัน
    """

    def __init__(self, pt_path, csv_path, scale_path, input_col, output_col):
        self.pt_path = Path(pt_path)
        self.csv_path = Path(csv_path)
        self.scale_path = Path(scale_path)
        self.input_col = input_col
        self.output_col = output_col

        # โหลด Scaler
        self.scaling_loader = ScalingZipLoader(scale_path)
        self.scaler_in = self.scaling_loader.scaler_in
        self.scaler_out = self.scaling_loader.scaler_out

        # โหลดข้อมูล
        self.data_pt = torch.load(pt_path)
        self.data_csv = pd.read_csv(csv_path)
        self.X_pt = self.data_pt["X"].numpy()
        self.y_pt = self.data_pt["y"].numpy()

    # ----------------------------------------------------------------------
    def show_sample(self, index=0, inverse=True):
        """ดูตัวอย่างเดียว"""
        X_sample = self.X_pt[index]
        y_sample = self.y_pt[index]

        print("=" * 60)
        print(f"[SAMPLE #{index}]")
        print(f"Input shape : {X_sample.shape}")
        print(f"Output shape: {y_sample.shape}")
        print("-" * 60)
        print("Scaled Input:\n", X_sample)
        print("Scaled Output:\n", y_sample)

        if inverse:
            X_inv = self.scaler_in.inverse_transform(X_sample)
            y_inv = self.scaler_out.inverse_transform(y_sample.reshape(1, -1))
            print("-" * 60)
            print("Inverse Input:\n", X_inv)
            print("Inverse Output:\n", y_inv)
        print("=" * 60)

    # ----------------------------------------------------------------------
    def show_samples(self, indices=[0, 1, 2], inverse=True):
        """ดูหลาย sample พร้อมกัน"""
        print(f"\n=== Showing {len(indices)} samples ===")
        for i in indices:
            self.show_sample(index=i, inverse=inverse)

    # ----------------------------------------------------------------------
    def compare_with_csv(self, seq_index=0):
        """เปรียบเทียบ sequence หนึ่งกับ CSV (ปรับปรุงการแสดงผล)"""
        X_sample = self.X_pt[seq_index]
        y_sample = self.y_pt[seq_index]

        X_inv = self.scaler_in.inverse_transform(X_sample)
        y_inv = self.scaler_out.inverse_transform(y_sample.reshape(1, -1))[0]

        df = self.data_csv[self.input_col + self.output_col]
        
        # เลือกช่วงข้อมูลจาก CSV ให้ตรงกับ X และ y ของ sample ที่เลือก
        # X คือข้อมูล input_sequence_length ตัว ก่อนหน้า y
        start_idx_csv = seq_index
        end_idx_csv_input = start_idx_csv + len(X_inv)
        
        df_window_input = df.iloc[start_idx_csv:end_idx_csv_input].reset_index(drop=True)
        # y คือข้อมูล ณ เวลาถัดไป (ที่โมเดลต้องทาย)
        df_window_output = df.iloc[end_idx_csv_input:end_idx_csv_input + 1].reset_index(drop=True)

        print("\n=== [COMPARE CSV vs DECODED .PT] ===")
        print(f"--- Sequence index: {seq_index} ---")
        print("CSV INPUT (original):")
        print(df_window_input[self.input_col])
        print("\nDecoded INPUT (from .pt inverse):")
        print(pd.DataFrame(X_inv, columns=self.input_col))
        print("-" * 50)
        print("CSV OUTPUT (original):")
        print(df_window_output[self.output_col])
        print("\nDecoded OUTPUT (from .pt inverse):")
        print(pd.DataFrame([y_inv], columns=self.output_col))

        # --- Plot Input ---
        plt.figure(figsize=(12, 5))
        plt.title(f"Sequence #{seq_index} - Input Comparison", fontsize=14)
        plt.plot(df_window_input.index, df_window_input[self.input_col], 
                 label="CSV Input (Original)", 
                 linestyle="--", marker='o', color='gray', alpha=0.7)
        plt.plot(df_window_input.index, X_inv, 
                 label="Decoded Input (.pt)", 
                 linewidth=2, marker='.', color='dodgerblue')
        
        plt.xlabel("Time Step in Sequence")
        plt.ylabel("Input Value")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

        # --- Plot Output ---
        plt.figure(figsize=(7, 4))
        plt.title(f"Sequence #{seq_index} - Output Comparison", fontsize=14)
        plt.scatter(0, df_window_output[self.output_col].values, 
                    label="CSV Output (True Value)", 
                    color='orange', s=150, marker='o', edgecolors='black')
        plt.scatter(0, y_inv, 
                    label="Decoded Output (.pt)", 
                    color='dodgerblue', s=100, marker='X')
        
        plt.ylabel("Output Value")
        plt.xticks([]) # ซ่อนตัวเลขแกน X
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()

    # ----------------------------------------------------------------------
    def compare_many_with_csv(self, start_idx=0, num_samples=5):
        """
        เปรียบเทียบหลาย sequence กับ CSV (ปรับปรุงการแสดงผล)
        แสดง input เป็นกราฟต่อเนื่อง, output เป็น scatter
        """
        end_idx = start_idx + num_samples
        indices = range(start_idx, end_idx)
        window_size = self.X_pt.shape[1] # หาขนาดของ input window

        # --- Plot Input ---
        plt.figure(figsize=(15, 6))
        plt.title(f"Input Comparison: Decoded Sequences vs. Original CSV (Samples {start_idx}–{end_idx-1})", fontsize=14)

        # พล็อตข้อมูล CSV จริงเป็นเส้นอ้างอิงพื้นหลัง
        total_length = window_size + num_samples - 1
        csv_input_range = self.data_csv[self.input_col].iloc[start_idx : start_idx + total_length]
        plt.plot(csv_input_range.index, csv_input_range.values, 
                 label='Original CSV Data', color='gray', linestyle='--', alpha=0.8, zorder=1)

        # วนลูปเพื่อพล็อต Decoded Input แต่ละ sequence
        for i in indices:
            X_inv = self.scaler_in.inverse_transform(self.X_pt[i])
            x_axis = range(i, i + len(X_inv))
            plt.plot(x_axis, X_inv, alpha=0.9, zorder=2)
        
        plt.plot([], [], color='dodgerblue', label=f'Decoded Sequences ({num_samples} samples)') # Label สำหรับกลุ่ม
        plt.xlabel("Overall Time Step (from CSV index)")
        plt.ylabel("Input Value")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

        # --- Plot Output ---
        y_inv_all = self.scaler_out.inverse_transform(self.y_pt[indices])
        
        # ข้อมูล y จาก CSV ที่สอดคล้องกัน (ตำแหน่ง = index + window_size)
        output_start_idx = start_idx + window_size
        output_end_idx = output_start_idx + num_samples
        csv_outputs = self.data_csv[self.output_col].iloc[output_start_idx:output_end_idx].values

        plt.figure(figsize=(10, 5))
        plt.title(f"Output Comparison (Samples {start_idx}–{end_idx-1})", fontsize=14)
        plt.plot(indices, csv_outputs, label="CSV Output Trend (True)", color="orange", linestyle='--', alpha=0.7)
        plt.scatter(indices, csv_outputs, label="CSV Output (True)", color="orange", marker='o', s=80, edgecolors='black')
        plt.scatter(indices, y_inv_all, label="Decoded Output (.pt)", color="dodgerblue", marker='X', s=80)
        
        plt.xlabel("Sequence Index")
        plt.ylabel("Output Value")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    viewer = PreviewPTData(
        pt_path=r"D:\Project_end\New_world\my_project\data\processed\pwm_duty_0.56_freq_0.01_pwm.pt",
        csv_path=r"D:\Project_end\New_world\my_project\data\raw\pwm_duty_0.56_freq_0.01_pwm.csv",
        scale_path=r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip",
        input_col=["DATA_INPUT"],
        output_col=["DATA_OUTPUT"]
    )

    # 1. ดูข้อมูลตัวอย่างแบบ Text
    viewer.show_samples(indices=[0, 1, 2, 3], inverse=True)

    # 2. เปรียบเทียบ 1 sample อย่างละเอียดพร้อมกราฟ
    viewer.compare_with_csv(seq_index=2)

    # 3. เปรียบเทียบหลายๆ sample เพื่อดูแนวโน้มโดยรวม
    viewer.compare_many_with_csv(start_idx=0, num_samples=10)