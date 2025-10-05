# src/utils/scaling_reference_combined.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from pathlib import Path
import zipfile
import os
import yaml
import io

class GlobalScalingReference:
    def __init__(self,
                 data_dir,
                 dataset_name="RC_Tank_Env_Training",
                 input_features=None,
                 output_features=None,
                 scaler_type="MinMaxScaler",
                 chunk_size=10000,
                 save_dir=None):

        # ✅ ตั้งค่าเส้นทาง
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.input_features = input_features or ["PWM_duty", "Prev_output"]
        self.output_features = output_features or ["Tank_level"]
        self.chunk_size = chunk_size
        self.scaler_type = scaler_type

        # ✅ โฟลเดอร์บันทึกหลัก
        self.save_dir = Path(save_dir) if save_dir else (self.data_dir / "scaling_reference")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ✅ ตรวจสอบไฟล์ CSV
        self.csv_files = list(self.data_dir.glob("*.csv"))
        if not self.csv_files:
            raise FileNotFoundError(f"❌ ไม่พบไฟล์ CSV ใน {self.data_dir}")

        # ✅ เตรียม Scaler
        scaler_cls = MinMaxScaler if scaler_type == "MinMaxScaler" else StandardScaler
        self.scaler_in = scaler_cls()
        self.scaler_out = scaler_cls()

    def fit_from_folder(self):
        """อ่านไฟล์ทั้งหมดในโฟลเดอร์และทำการ fit scaler แบบ incremental"""
        print("🚀 Starting Global Scaler Fitting Process ...")
        print(f"📁 Processing {len(self.csv_files)} files from: {self.data_dir}\n")

        for i, csv_path in enumerate(self.csv_files, 1):
            print(f"  [{i}/{len(self.csv_files)}] Reading {csv_path.name}")
            with pd.read_csv(csv_path, chunksize=self.chunk_size) as reader:
                for chunk in reader:
                    # ตรวจสอบคอลัมน์
                    if not all(col in chunk.columns for col in self.input_features + self.output_features):
                        raise KeyError(f"❌ Missing columns in {csv_path.name}")

                    X = chunk[self.input_features].values
                    y = chunk[self.output_features].values
                    self.scaler_in.partial_fit(X)
                    self.scaler_out.partial_fit(y)

        # ✅ สร้าง metadata
        self.metadata = {
            "dataset_name": self.dataset_name,
            "scaler_type": self.scaler_type,
            "num_files": len(self.csv_files),
            "input_features": self.input_features,
            "output_features": self.output_features,
            "input_min": self.scaler_in.data_min_.tolist(),
            "input_max": self.scaler_in.data_max_.tolist(),
            "output_min": self.scaler_out.data_min_.tolist(),
            "output_max": self.scaler_out.data_max_.tolist()
        }

        print("\n✅ Scaler fitting complete!")
        print(f"  Input range:  {self.scaler_in.data_min_} → {self.scaler_in.data_max_}")
        print(f"  Output range: {self.scaler_out.data_min_} → {self.scaler_out.data_max_}")
        return self.metadata

    def save_all_to_zip(self):
        """บีบอัด Scaler และ metadata เป็น ZIP เดียว"""
        zip_path = self.save_dir / f"{self.dataset_name}_scalers.zip"
        print(f"\n💾 Creating single ZIP archive: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # ✅ เขียน input_scaler.pkl และ output_scaler.pkl ลงใน zip โดยไม่ต้องสร้างไฟล์จริง
            for name, scaler in [("input_scaler.pkl", self.scaler_in), ("output_scaler.pkl", self.scaler_out)]:
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                zipf.writestr(name, buffer.getvalue())

            # ✅ เขียน metadata.yaml ลง zip
            yaml_str = yaml.dump(self.metadata, allow_unicode=True, sort_keys=False)
            zipf.writestr("metadata.yaml", yaml_str)

        print(f"📦 ZIP created successfully at: {zip_path}")
        return zip_path

    def run(self):
        """รันกระบวนการทั้งหมด"""
        self.fit_from_folder()
        zip_path = self.save_all_to_zip()
        return {"metadata": self.metadata, "zip": zip_path}


if __name__ == "__main__":
    FOLDER_DATA =r"D:\Project_end\New_world\my_project\data\raw"
    FOLDER_SAVE_SCALE = r"D:\Project_end\New_world\my_project\config"
    NAME_FILE = "RC_Tank_Env_Training2"
    COLUMN_INPUT = "DATA_INPUT"
    COLUMN_OUTPUT = "DATA_OUTPUT"
    SCALER_TYPE = "MinMaxScaler"
    CHUNK_SIZE = 10000

    scaler_ref = GlobalScalingReference(
        data_dir = FOLDER_DATA,
        save_dir = FOLDER_SAVE_SCALE,
        dataset_name = NAME_FILE,
        input_features =[COLUMN_INPUT],
        output_features =[COLUMN_OUTPUT],
        scaler_type = SCALER_TYPE,
        chunk_size = CHUNK_SIZE
    )

    result = scaler_ref.run()

    print("\n🎯 Summary:")
    print(pd.DataFrame([result["metadata"]]))
