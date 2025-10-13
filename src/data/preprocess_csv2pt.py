# src/data/preprocess_csv2pt.py
from src.data.scaling_loader import ScalingZipLoader
from pathlib import Path
import torch
import numpy as np
import pyarrow.csv as pv
import pyarrow.compute as pc
from tqdm import tqdm
import multiprocessing as mp
import os


class convert_csv2pt:
    def __init__(self, input_folder, output_folder, scale_path,
                 input_col, output_col, sequence_size=10, chunksize=100000,
                 num_workers=4, allow_padding=True, pad_value=0.0):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.scale_path = Path(scale_path)
        self.input_col = input_col
        self.output_col = output_col
        self.sequence_size = sequence_size
        self.chunksize = chunksize
        self.num_workers = num_workers
        self.allow_padding = allow_padding
        self.pad_value = pad_value

        # โหลด Scaler
        self.scaling_loader = ScalingZipLoader(scale_path)
        self.scaler_input = self.scaling_loader.scaler_in
        self.scaler_output = self.scaling_loader.scaler_out

        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    def _read_chunk(self, file_path, skip_rows):

        # ✅ ตั้งค่า options (เวอร์ชันใหม่ของ PyArrow)
        read_opts = pv.ReadOptions(skip_rows=skip_rows, autogenerate_column_names=False)
        parse_opts = pv.ParseOptions(delimiter=',')
        convert_opts = pv.ConvertOptions()

        # ✅ อ่านไฟล์ CSV ทั้งหมดที่เหลือ แล้วตัดเฉพาะ rows ที่ต้องการ
        table = pv.read_csv(
            file_path,
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts
        )

        # ✅ slice เฉพาะส่วนที่ต้องการ (จำลอง nrows)
        table = table.slice(0, self.chunksize)

        # ✅ แปลงกลับเป็น pandas DataFrame
        return table.to_pandas()

    # ----------------------------------------------------------------------
    def _process_file(self, file_name):
        file_path = self.input_folder / file_name
        total_rows = sum(1 for _ in open(file_path)) - 1
        num_chunks = max(1, total_rows // self.chunksize)

        X_total, y_total = [], []
        for i in range(num_chunks):
            start = i * self.chunksize
            df = self._read_chunk(file_path, start)

            # Extract & Scale
            X_data = df[self.input_col].to_numpy()
            y_data = df[self.output_col].to_numpy()

            if X_data.ndim == 1:
                X_data = X_data.reshape(-1, 1)
            if y_data.ndim == 1:
                y_data = y_data.reshape(-1, 1)

            X_scaled = self.scaler_input.transform(X_data)
            y_scaled = self.scaler_output.transform(y_data)

            # Convert to sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
            if len(X_seq) > 0:
                X_total.append(X_seq)
                y_total.append(y_seq)

        # รวมทั้งหมดแล้วบันทึกเป็น .pt
        X_total = np.vstack(X_total)
        y_total = np.vstack(y_total)
        torch.save({'X': torch.tensor(X_total, dtype=torch.float32),
                    'y': torch.tensor(y_total, dtype=torch.float32)},
                   self.output_folder / f"{file_name.replace('.csv', '.pt')}")

        return file_name

    # ----------------------------------------------------------------------
    def create_sequences(self, X_data, y_data):
        Xs, ys = [], []
        n = len(X_data)

        if n < self.sequence_size and self.allow_padding:
            X_pad = self.pad_or_truncate(X_data)
            y_pad = self.pad_or_truncate(y_data)
            return np.array([X_pad]), np.array([y_pad[-1]])

        for i in range(max(0, n - self.sequence_size)):
            Xs.append(X_data[i:i + self.sequence_size])
            ys.append(y_data[i + self.sequence_size - 1])
        return np.array(Xs), np.array(ys)

    # ----------------------------------------------------------------------
    def pad_or_truncate(self, seq):
        if len(seq) < self.sequence_size:
            pad_size = self.sequence_size - len(seq)
            pad = np.full((pad_size, seq.shape[1]), self.pad_value)
            seq = np.vstack((pad, seq))
        elif len(seq) > self.sequence_size:
            seq = seq[-self.sequence_size:]
        return seq

    # ----------------------------------------------------------------------
    def process_all(self):
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        print(f"[INFO] Found {len(csv_files)} CSV files")

        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap(self._process_file, csv_files),
                      total=len(csv_files),
                      desc="Converting CSV → PT"))


# =============================================================================
# RUN EXAMPLE
# =============================================================================
if __name__ == "__main__":
    converter = convert_csv2pt(
        input_folder=r"D:\Project_end\New_world\my_project\data\raw",
        output_folder=r"D:\Project_end\New_world\my_project\data\processed",
        scale_path=r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip",
        input_col=['DATA_INPUT'],
        output_col=['DATA_OUTPUT'],
        sequence_size=10,
        chunksize=200000,
        num_workers=6,
    )

    converter.process_all()
