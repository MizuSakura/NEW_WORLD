import joblib
import yaml
import zipfile
import io
import warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class ScalingZipLoader:
    def __init__(self, zip_path):
        self.zip_path = Path(zip_path)
        self.scaler_in = None
        self.scaler_out = None
        self.metadata = None
        self._load_from_zip()

    def _get_file_from_zip(self, zipf, name_keyword):
        for info in zipf.infolist():
            if name_keyword in info.filename:
                with zipf.open(info) as f:
                    return f.read()
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ที่มีชื่อ '{name_keyword}' ใน zip")

    def _safe_load_yaml(self, zipf, filename="metadata.yaml"):
        try:
            with zipf.open(filename) as f:
                return yaml.safe_load(f)
        except KeyError:
            warnings.warn(f"⚠️ ไม่พบ {filename} ใน zip — metadata จะเป็น None", UserWarning)
            return None

    def _load_from_zip(self):
        if not self.zip_path.exists():
            raise FileNotFoundError(f"❌ ไม่พบไฟล์ ZIP ที่ {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as zipf:
            self.scaler_in = joblib.load(io.BytesIO(self._get_file_from_zip(zipf, "input_scaler")))
            self.scaler_out = joblib.load(io.BytesIO(self._get_file_from_zip(zipf, "output_scaler")))
            self.metadata = self._safe_load_yaml(zipf)

    def transform_input(self, X):
        return self.scaler_in.transform(X)

    def inverse_output(self, y_scaled):
        return self.scaler_out.inverse_transform(y_scaled)

    def is_loaded(self):
        return all([self.scaler_in is not None, self.scaler_out is not None])

    def summary(self):
        print("📦 ScalingZipLoader Summary:")
        if self.metadata:
            for k, v in self.metadata.items():
                print(f"  {k}: {v}")
        else:
            print("  ⚠️ ไม่มีข้อมูล metadata")

# =====================================================
# 🔽 ส่วนทดสอบการทำงาน (Test Section)
# =====================================================
if __name__ == "__main__":
    print("🧪 เริ่มการทดสอบ ScalingZipLoader ...")

    test_zip_path = Path(r"D:\Project_end\New_world\my_project\config\RC_Tank_Env_Training2_scalers.zip")

    # 🧭 ถามผู้ใช้ก่อนว่าอยากทำอะไร
    if test_zip_path.exists():
        print(f"📂 พบไฟล์อยู่แล้วที่: {test_zip_path}")
        choice = input("ต้องการ (r) อ่านไฟล์ หรือ (w) เขียนทับใหม่? [r/w]: ").strip().lower()
    else:
        choice = "w"

    if choice == "w":
        print("✏️ สร้างไฟล์ zip ใหม่...")

        # ข้อมูลจำลอง
        X = np.array([[0], [1], [2], [3], [4]], dtype=float)
        scaler_in = MinMaxScaler(feature_range=(-1, 1)).fit(X)
        scaler_out = MinMaxScaler(feature_range=(0, 10)).fit(X)
        metadata = {
            "input_feature_range": "(-1, 1)",
            "output_feature_range": "(0, 10)",
            "description": "Test scaler zip file",
        }

        # สร้าง zip ตัวอย่าง
        with zipfile.ZipFile(test_zip_path, "w") as zipf:
            buffer_in = io.BytesIO()
            joblib.dump(scaler_in, buffer_in)
            zipf.writestr("input_scaler.pkl", buffer_in.getvalue())

            buffer_out = io.BytesIO()
            joblib.dump(scaler_out, buffer_out)
            zipf.writestr("output_scaler.pkl", buffer_out.getvalue())

            zipf.writestr("metadata.yaml", yaml.safe_dump(metadata))

        print(f"✅ สร้างไฟล์ zip ใหม่ที่: {test_zip_path.resolve()}")

    elif choice == "r":
        print("📖 อ่านข้อมูลจาก zip ที่มีอยู่...")

    else:
        print("⚠️ ตัวเลือกไม่ถูกต้อง — จะทำการอ่านไฟล์แทน")
    
    # โหลด zip ด้วย ScalingZipLoader
    loader = ScalingZipLoader(test_zip_path)

    # แสดงข้อมูล summary
    loader.summary()

    # ตัวอย่าง transform & inverse
    X_test = np.array([[1.5], [3.0]])
    X_scaled = loader.transform_input(X_test)
    y_inverse = loader.inverse_output(X_scaled)

    print("\n🔹 ตัวอย่างการแปลงข้อมูล:")
    print(f"  ข้อมูลต้นฉบับ:\n{X_test}")
    print(f"  หลัง scale ด้วย input_scaler:\n{X_scaled}")
    print(f"  inverse ด้วย output_scaler:\n{y_inverse}")

    print("\n🎉 การทดสอบเสร็จสมบูรณ์")