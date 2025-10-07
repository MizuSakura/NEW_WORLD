import sys
import subprocess
from fastapi import FastAPI
from pathlib import Path
import os

PYTHON_EXECUTABLE = sys.executable 

# --- Path ที่นาย Hardcode ---
RESCALE_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Rescal_data.py")
TRAIN_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Model_LSTM.py")
SCRIPT_WORKING_DIR = RESCALE_SCRIPT_PATH.parent

# --- สร้างโฟลเดอร์สำหรับเก็บ Log ---
LOG_DIR = SCRIPT_WORKING_DIR.parent.parent / "logs" # ย้าย logs ไปไว้ข้างนอก
LOG_DIR.mkdir(exist_ok=True) 

app = FastAPI(title="AI Factory Controller")

def start_background_process(script_path: Path, log_name: str):
    if not script_path.exists():
        return {"error": f"Script not found: {script_path}"}
    
    # กำหนดไฟล์ Log ที่จะเขียน
    log_out_path = LOG_DIR / f"{log_name}_output.log"
    log_err_path = LOG_DIR / f"{log_name}_error.log"
    
    # เปิดไฟล์ Log
    log_out = open(log_out_path, "w")
    log_err = open(log_err_path, "w")

    # เตรียม Environment
    script_env = os.environ.copy()
    script_env["PYTHONUTF8"] = "1"

    # สั่งเริ่มทำงานเบื้องหลัง!
    subprocess.Popen(
        [PYTHON_EXECUTABLE, "-u", str(script_path.name)],
        cwd=str(SCRIPT_WORKING_DIR),
        stdout=log_out,
        stderr=log_err,
        env=script_env
    )
    
    # ตอบกลับทันที! ไม่ต้องรอ!
    return {"message": f"Process '{log_name}' started in background. Check the 'logs' folder for progress."}


@app.post("/start-scaling")
async def trigger_scaling():
    return start_background_process(RESCALE_SCRIPT_PATH, "scaling")

@app.post("/start-training")
async def trigger_training():
    return start_background_process(TRAIN_SCRIPT_PATH, "training")

# เปลี่ยนจาก subprocess.run() เป็น subprocess.Popen() เพื่อให้ทำงานแบบเบื้องหลัง
# เพิ่มการจัดการ Log โดยแยกไฟล์สำหรับ stdout และ stderr
# เพิ่มโฟลเดอร์ logs เพื่อเก็บไฟล์ Log
