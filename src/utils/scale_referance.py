# src/utils/scale_referance.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class ScalingReference:
    def __init__(self, data_dir="data", dataset_name="RC_Tank_Env_Training",
                 input_features=None, output_features=None,
                 scaler_type="MinMaxScaler", window_size=30):

        self.base_dir = Path(__file__).resolve().parents[2]

        # ✅ รองรับ absolute path หรือ relative path
        data_dir = Path(data_dir)
        self.data_dir = data_dir if data_dir.is_absolute() else self.base_dir / data_dir

        self.dataset_name = dataset_name
        self.input_features = input_features or ["PWM_duty", "Prev_output"]
        self.output_features = output_features or ["Tank_level"]
        self.scaler_type = scaler_type
        self.window_size = window_size

        # เตรียมที่เก็บผลลัพธ์
        self.save_dir = self.base_dir / "scaling_reference" / dataset_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ✅ ตรวจสอบไฟล์ CSV ให้ยืดหยุ่น (รองรับ data/ และ root)
        self.csv_file = self.data_dir / f"{dataset_name}.csv"
        if not self.csv_file.exists():
            alt_path = self.data_dir / "data" / f"{dataset_name}.csv"
            if alt_path.exists():
                self.csv_file = alt_path
            else:
                raise FileNotFoundError(f"❌ ไม่พบไฟล์: {self.csv_file} หรือ {alt_path}")

        # โหลดข้อมูล
        self.df = pd.read_csv(self.csv_file)

        # เตรียม scaler
        scaler_cls = MinMaxScaler if scaler_type == "MinMaxScaler" else StandardScaler
        self.scaler_in = scaler_cls()
        self.scaler_out = scaler_cls()

    def fit_scalers(self):
        """Fit ข้อมูลจาก CSV"""
        X = self.df[self.input_features].values
        y = self.df[self.output_features].values

        self.scaler_in.fit(X)
        self.scaler_out.fit(y)

        # สร้าง metadata สำหรับบันทึก
        self.metadata = {
            "input_min": self.scaler_in.data_min_.tolist(),
            "input_max": self.scaler_in.data_max_.tolist(),
            "output_min": self.scaler_out.data_min_.tolist(),
            "output_max": self.scaler_out.data_max_.tolist(),
        }
        return self.metadata

    def save(self):
        """บันทึก scaler และ metadata"""
        in_path = self.save_dir / "input_scaler.pkl"
        out_path = self.save_dir / "output_scaler.pkl"
        meta_path = self.save_dir / "metadata.csv"

        joblib.dump(self.scaler_in, in_path)
        joblib.dump(self.scaler_out, out_path)
        pd.DataFrame([self.metadata]).to_csv(meta_path, index=False)

        return {"input": in_path, "output": out_path, "metadata": meta_path}

    def run(self):
        """ทำทั้ง fit และ save"""
        self.fit_scalers()
        return self.save()
