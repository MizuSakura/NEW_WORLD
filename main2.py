import sys
import subprocess
from fastapi import FastAPI, HTTPException
from pathlib import Path
import os

PYTHON_EXECUTABLE = sys.executable 

# --- Path ที่นาย Hardcode ---
# **สำคัญ:** เช็คให้แน่ใจว่า Path 2 บรรทัดนี้ถูกต้องตามที่อยู่ไฟล์ของนายจริงๆ!
RESCALE_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Rescal_data.py")
TRAIN_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Model_LSTM.py")

# --- หา "จุดยืน" ที่ถูกต้อง ---
SCRIPT_WORKING_DIR = RESCALE_SCRIPT_PATH.parent

app = FastAPI(title="AI Factory Controller")

def run_script_and_capture(script_path: Path):
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"FATAL: Hardcoded script path does not exist! Check your path: {script_path}")
    
    try:
        # สร้าง environment variables ใหม่ เพื่อบังคับ UTF-8
        script_env = os.environ.copy()
        script_env["PYTHONUTF8"] = "1"

        # ใช้ .run() เพื่อรอจนเสร็จและดักจับทุกอย่าง
        procesdos = subprocess.run(
            [PYTHON_EXECUTABLE, "-u", str(script_path.name)],
            cwd=str(SCRIPT_WORKING_DIR),
            capture_output=True,
            text=True,
            check=False,
            env=script_env # <--- ไม้ตายสุดท้าย!
        )
        
        full_log = f"--- SCRIPT OUTPUT ---\n{process.stdout}\n\n--- SCRIPT ERROR ---\n{process.stderr}"
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=full_log)
        
        return {"message": "Script finished successfully!", "log": full_log}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastAPI Internal Error: {str(e)}")


@app.post("/start-scaling")
async def trigger_scaling():
    return run_script_and_capture(RESCALE_SCRIPT_PATH)

@app.post("/start-training")
async def trigger_training():
    return run_script_and_capture(TRAIN_SCRIPT_PATH)


#บังคับให้ Python interpreter ที่ถูกเรียกใช้ UTF-8 เท่านั้น!
#PYTHONUTF8="1" 


# full log = f"--- SCRIPT OUTPUT ---\n{process.stdout}\n\n--- SCRIPT ERROR ---\n{process.stderr}" คือการรวมผลลัพธ์และข้อผิดพลาดจากการรันสคริปต์เข้าด้วยกันในตัวแปรเดียว
#ถ้า process.returncode != 0: คือการตรวจสอบว่าการรันสคริปต์ล้มเหลวหรือไม่ โดยถ้าล้มเหลวจะมีการส่งกลับ HTTP 500 พร้อมกับ log ที่รวมผลลัพธ์และข้อผิดพลาด