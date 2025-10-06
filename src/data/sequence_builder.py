# src/utils/sequence_builder.py
import numpy as np

def create_sequences(data, window_size):
    """
    สร้าง input-output sequence สำหรับ LSTM
    ------------------------------------------
    Args:
        data : np.ndarray (N, n_features)
            ข้อมูลทั้งหมดที่รวม input + output ไว้
            เช่น [[PWM, Tank_Level], ...]
        window_size : int
            จำนวน timestep ต่อหนึ่ง sequence (เช่น 30)
    
    Returns:
        X : np.ndarray, shape (num_seq, window_size, n_features)
            ลำดับข้อมูล input
        y : np.ndarray, shape (num_seq,)
            ค่าผลลัพธ์ที่ต้องการทำนาย (เช่น Tank_Level ถัดไป)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 1])  # ใช้คอลัมน์ที่ 1 (output)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # 🧪 ตัวอย่างการทดสอบการทำงาน
    print("🧩 Testing create_sequences() ...")

    # จำลองข้อมูล [input, output]
    time = np.arange(0, 10, 0.1)
    data_input = np.sin(time)
    data_output = np.cos(time)
    data = np.column_stack((data_input, data_output))

    X, y = create_sequences(data, window_size=5)

    print(f"✅ X shape: {X.shape}  | y shape: {y.shape}")
    print("🔍 Example X[0]:", X[0])
    print("🔍 Example y[0]:", y[0])
