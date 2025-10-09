from src.data.scaling_loader import ScalingZipLoader
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils.logger import Logger
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os


class SequenceDataset(Dataset):

    def __init__(self, folder_path, scale_path, sequence_size,
                 input_col='DATA_INPUT', output_col='DATA_OUTPUT', 
                 file_ext='.csv', chunksize=1000,
                 allow_padding=True, pad_value=0.0):
        

        self.folder_path = Path(folder_path)
        self.scale_path = Path(scale_path)
        self.sequence_size = sequence_size
        self.input_col = input_col
        self.output_col = output_col
        self.file_ext = file_ext
        self.chunksize = chunksize
        self.allow_padding = allow_padding
        self.pad_value = pad_value

        
        self.scaling_loader = ScalingZipLoader(self.scale_path)
        self.scaler_input = self.scaling_loader.scaler_in  
        self.scaler_output = self.scaling_loader.scaler_out

        
        self.X, self.y = self.load_and_create_sequences(folder_path, file_ext)

    def load_and_create_sequences(self, folder_path, file_ext):
        X_all, y_all = [], []
        files = [f for f in os.listdir(folder_path) if f.endswith(file_ext)]

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            for chunk in pd.read_csv(file_path, chunksize=self.chunksize):
                # scale input/output
                inputs = self.scaler_input.transform(chunk[self.input_col].values)
                outputs = self.scaler_output.transform(chunk[self.output_col].values)

                X_seq, y_seq = self.create_sequences(inputs, outputs)

                if len(X_seq) > 0:
                    X_all.append(X_seq)
                    y_all.append(y_seq)

        return np.vstack(X_all), np.vstack(y_all)


    def create_sequences(self, X_data, y_data):
        Xs, ys = [], []
        data_len = len(X_data)
        
        if data_len < self.sequence_size and self.allow_padding:
            # pad แล้วสร้าง sequence เดียว
            X_pad = self.pad_or_truncate(X_data)
            y_pad = self.pad_or_truncate(y_data)

            return np.array([X_pad]), np.array([y_pad[-1]])
        
        for i in range(max(0, data_len - self.sequence_size)): 
            Xs.append(X_data[i:i + self.sequence_size])
            ys.append(y_data[i + self.sequence_size])

        return np.array(Xs), np.array(ys)
    
    def pad_or_truncate(self, seq):
        if len(seq) < self.sequence_size:
            pad_size = self.sequence_size - len(seq)
            pad = np.full((pad_size, seq.shape[1]), self.pad_value)
            seq = np.vstack((pad, seq))
        elif len(seq) > self.sequence_size:
            seq = seq[-self.sequence_size:]
        return seq

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LazyChunkedSequenceDataset(Dataset):
    def __init__(self, folder_path, scale_path, sequence_size,
                 input_col='DATA_INPUT', output_col='DATA_OUTPUT',
                 file_ext='.csv', chunksize=1000):
        """
        Dataset แบบ Lazy Loading (โหลดเฉพาะส่วนที่จำเป็นจากไฟล์ CSV)
        โดยตั้งชื่อพารามิเตอร์ให้เหมือน SequenceDataset เพื่อให้สลับใช้ได้ง่าย
        """

        self.folder_path = Path(folder_path)
        self.scale_path = Path(scale_path)
        self.sequence_size = sequence_size
        self.input_col = input_col
        self.output_col = output_col
        self.file_ext = file_ext
        self.chunksize = chunksize

        # โหลด Scaler จาก zip (เหมือน SequenceDataset)
        self.scaling_loader = ScalingZipLoader(self.scale_path)
        self.scaler_input = self.scaling_loader.scaler_in
        self.scaler_output = self.scaling_loader.scaler_out

        # สร้าง mapping สำหรับ sequence ทั้งหมด
        self.sequence_map = self._index_sequences()

    def _index_sequences(self):
        """
        สร้าง mapping ของ sequence ทั้งหมดในทุกไฟล์
        (index → (filename, start_row, local_index))
        """
        mapping = []
        files = [f for f in os.listdir(self.folder_path) if f.endswith(self.file_ext)]

        for file_name in files:
            file_path = os.path.join(self.folder_path, file_name)
            total_rows = sum(1 for _ in open(file_path)) - 1  # ลบ header

            for start_row in range(0, total_rows, self.chunksize):
                end_row = min(start_row + self.chunksize, total_rows)
                n_rows = end_row - start_row

                if n_rows <= self.sequence_size:
                    continue

                num_seq = n_rows - self.sequence_size
                for local_idx in range(num_seq):
                    mapping.append((file_name, start_row, local_idx))

        return mapping

    def __len__(self):
        return len(self.sequence_map)

    def __getitem__(self, idx):
        """
        ดึง sequence ตาม index โดยโหลดเฉพาะ chunk ที่จำเป็น
        """
        file_name, start_row, local_idx = self.sequence_map[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # อ่านเฉพาะ chunk ที่ต้องใช้
        chunk = pd.read_csv(
            file_path,
            skiprows=range(1, start_row + 1),  # ข้าม header + แถวก่อนหน้า
            nrows=self.chunksize
        )

        # Scale input/output
        X_data = self.scaler_input.transform(chunk[self.input_col].values.reshape(-1, 1)
                                             if isinstance(self.input_col, str)
                                             else chunk[self.input_col].values)
        y_data = self.scaler_output.transform(chunk[self.output_col].values.reshape(-1, 1)
                                              if isinstance(self.output_col, str)
                                              else chunk[self.output_col].values)

        # ตัดช่วง sequence
        start = local_idx
        end = local_idx + self.sequence_size
        X_seq = X_data[start:end]
        y_seq = y_data[end - 1]  # many-to-one

        return np.array(X_seq), np.array(y_seq)
        
if __name__ == "__main__":
    PATH_SCALE = r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip"
    FOLDER_FILE = r"D:\Project_end\New_world\my_project\data\raw"

    # เลือก Dataset แบบไหนที่ต้องการทดสอบ: "full" หรือ "lazy"
    dataset_type = "full"  # <-- เปลี่ยนเป็น "lazy" เพื่อทดสอบ LazyChunkedSequenceDataset

    if dataset_type == "full":
        print("Testing SequenceDataset (full load)...")
        dataset = SequenceDataset(
            folder_path=FOLDER_FILE,
            scale_path=PATH_SCALE,
            sequence_size=11,
            input_col=['DATA_INPUT'],
            output_col=["DATA_OUTPUT"],
            chunksize=1000,
            allow_padding=True,
            pad_value=0.0
        )
    elif dataset_type == "lazy":
        print("Testing LazyChunkedSequenceDataset (lazy load)...")
        dataset = LazyChunkedSequenceDataset(
            folder_path=FOLDER_FILE,
            scale_path=PATH_SCALE,
            sequence_size=11,
            input_col=['DATA_INPUT'],
            output_col=["DATA_OUTPUT"],
            chunksize=1000
        )
    else:
        raise ValueError("dataset_type must be 'full' or 'lazy'")

    # ทดสอบดึงข้อมูลตัวอย่าง
    X, y = dataset[0]
    print(f"X shape: {X.shape} | y shape: {y.shape} | dataset length: {len(dataset)}")
    print(f"First sample y: {y}")

    # สามารถทดสอบ DataLoader ได้ด้วย
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_X, batch_y = next(iter(dataloader))
    print(f"Batch X shape: {batch_X.shape} | Batch y shape: {batch_y.shape}")