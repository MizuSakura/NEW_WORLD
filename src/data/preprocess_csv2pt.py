# src/data/preprocess_to_pt.py

from src.data.scaling_loader import ScalingZipLoader
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm

class convert_csv2pt:
    """
    PreprocessToPT
    ==============
    A utility class for converting large CSV files into .pt files (PyTorch tensors)
    with support for scaling and sequence creation.
    
    Features:
    ----------
    - Handles large CSVs efficiently via chunked reading.
    - Supports both per-file and merged saving.
    - Compatible with ScalingZipLoader for automatic scaling.
    - Can reload .pt data as TensorDataset for PyTorch training.
    """

    def __init__(self,
                folder_path,
                scale_path,
                save_dir,
                sequence_size=10,
                input_col='DATA_INPUT',
                output_col='DATA_OUTPUT',
                file_ext='.csv',
                chunksize=1000,
                allow_padding=True,
                pad_value=0.0,
                merge_all=False):
        

        self.folder_path = Path(folder_path)
        self.scale_path = Path(scale_path)
        self.save_dir = Path(save_dir)
        
        self.sequence_size = sequence_size
        self.input_col = input_col
        self.output_col = output_col
        self.file_ext = file_ext
        self.chunksize = chunksize
        self.allow_padding = allow_padding
        self.pad_value = pad_value
        self.merge_all = merge_all

        # Load scaler (auto-handled by ScalingZipLoader)
        self.scaling_loader = ScalingZipLoader(self.scale_path)
        self.scaler_input = self.scaling_loader.scaler_in
        self.scaler_output = self.scaling_loader.scaler_out

        self.sequence_map = self._index_sequences()

    def _index_sequences(self):
    
        mapping = []
        files = [f for f in os.listdir(self.folder_path) if f.endswith(self.file_ext)]

        for file_name in files:
            file_path = os.path.join(self.folder_path, file_name)

            # นับจำนวนแถวจริง ๆ (ลบ header)
            with open(file_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1

            # loop ตาม chunk
            for start_row in range(0, total_rows, self.chunksize):
                end_row = min(start_row + self.chunksize, total_rows)
                n_rows = end_row - start_row

                # ถ้า chunk น้อยกว่า sequence_size → skip
                if n_rows < self.sequence_size:
                    continue

                # จำนวน sequence ใน chunk
                num_seq = n_rows - self.sequence_size + 1  # ✅ ปลอดภัย

                for local_idx in range(num_seq):
                    mapping.append((file_name, start_row, local_idx))

        return mapping
    
    def __len__(self):
        return len(self.sequence_map)
    
    
    
    def __getitem__(self, idx):

        # 1️⃣ หาตำแหน่ง sequence จาก mapping
        file_name, start_row, local_idx = self.sequence_map[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # 2️⃣ โหลด chunk ของ CSV
        chunk = pd.read_csv(
            file_path,
            skiprows=range(1, start_row + 1),  # ข้าม header + แถวก่อนหน้า
            nrows=self.chunksize
        )

        # 3️⃣ scale input/output
        # รองรับหลาย column
        if isinstance(self.input_col, str):
            X_data = chunk[[self.input_col]].values
        else:
            X_data = chunk[self.input_col].values

        if isinstance(self.output_col, str):
            y_data = chunk[[self.output_col]].values
        else:
            y_data = chunk[self.output_col].values

        X_scaled = self.scaler_input.transform(X_data)
        y_scaled = self.scaler_output.transform(y_data)

        # 4️⃣ ตัด sequence
        start = local_idx
        end = local_idx + self.sequence_size
        X_seq = X_scaled[start:end]
        y_seq = y_scaled[end - 1]  # many-to-one

        return np.array(X_seq), np.array(y_seq)

    def process_all(self):
        """
        แปลง CSV → PT โดยใช้ mapping และ __getitem__
        - merge_all=True → save ไฟล์เดียว
        - merge_all=False → save per-file
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

        all_X, all_y = [], []
        current_file = None
        file_X, file_y = [], []

        # ใช้ tqdm แสดง progress bar
        for idx in tqdm(range(len(self)), desc="Processing sequences", unit="seq"):
            file_name, _, _ = self.sequence_map[idx]
            X_seq, y_seq = self[idx]  # ดึง sequence

            if self.merge_all:
                all_X.append(X_seq)
                all_y.append(y_seq)
            else:
                # แยก per-file
                if current_file != file_name:
                    if current_file is not None:
                        # save ไฟล์ก่อนหน้า
                        save_path = self.save_dir / f"{Path(current_file).stem}.pt"
                        torch.save({'X': torch.tensor(np.vstack(file_X), dtype=torch.float32),
                                    'y': torch.tensor(np.vstack(file_y), dtype=torch.float32)}, save_path)
                        print(f"[✅] Saved {save_path}")

                    # reset
                    current_file = file_name
                    file_X, file_y = [], []

                file_X.append(X_seq)
                file_y.append(y_seq)

        # save ไฟล์สุดท้าย (per-file)
        if not self.merge_all and current_file is not None:
            save_path = self.save_dir / f"{Path(current_file).stem}.pt"
            torch.save({'X': torch.tensor(np.vstack(file_X), dtype=torch.float32),
                        'y': torch.tensor(np.vstack(file_y), dtype=torch.float32)}, save_path)
            print(f"[✅] Saved {save_path}")

        # merge_all case
        if self.merge_all:
            save_path = self.save_dir / "merged_dataset.pt"
            torch.save({'X': torch.tensor(np.vstack(all_X), dtype=torch.float32),
                        'y': torch.tensor(np.vstack(all_y), dtype=torch.float32)}, save_path)
            print(f"[✅] Saved merged dataset → {save_path}")


if __name__ == "__main__":
    conver = convert_csv2pt(
        folder_path=r"D:\Project_end\New_world\my_project\data\raw",
        scale_path=r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip",
        save_dir=r"D:\Project_end\New_world\my_project\data\processed",
        sequence_size=10,
        input_col='DATA_INPUT',
        output_col='DATA_OUTPUT',
        chunksize=100000,
        allow_padding=True,
        pad_value=0.0,
        merge_all=False
    )
    print(len(conver))
    conver.process_all()