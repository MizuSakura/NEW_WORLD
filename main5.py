import sys
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
import os
import asyncio

print("--- INSIDE THE SCRIPT ---")
print(f"My Current Working Directory is: {os.getcwd()}")
print(f"My sys.path is: {sys.path}")
# --- Configuration (เหมือนเดิม) ---
PYTHON_EXECUTABLE = sys.executable 
SIMULATION_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\NEW_WORLD\NEW_WORLD\src\environment\data_simulation.py")
RESCALE_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\NEW_WORLD\NEW_WORLD\src\utils\scale_referance.py")
#TRAIN_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Model_LSTM.py")
# SCRIPT_WORKING_DIR = RESCALE_SCRIPT_PATH.parent
# LOG_DIR = SCRIPT_WORKING_DIR.parent.parent / "logs"
# LOG_DIR.mkdir(exist_ok=True) 

PROJECT_ROOT = RESCALE_SCRIPT_PATH.parent.parent.parent # C:\...\NEW_WORLD\NEW_WORLD
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True) 


app = FastAPI(title="AI Factory Controller")

# --- ฟังก์ชัน "แอบดูไฟล์" (ใหม่!) ---
async def tail_log_file(websocket: WebSocket, log_path: Path):
    """ฟังก์ชันนี้จะคอยเฝ้าดูไฟล์ Log และส่งข้อมูลใหม่ๆ ผ่าน WebSocket"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # ไปที่ท้ายไฟล์
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    # ถ้าไม่มีบรรทัดใหม่ ก็รอแป๊บนึง
                    await asyncio.sleep(0.5)
                    continue
                # ส่งบรรทัดใหม่ไปให้ C#
                await websocket.send_text(line.strip())
    except FileNotFoundError:
        await websocket.send_text(f"ERROR: Log file not found at {log_path}")
    except Exception as e:
        await websocket.send_text(f"ERROR: An error occurred while reading log: {e}")


# --- WebSocket Endpoint (ใหม่!) ---
@app.websocket("/ws/log")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # รอรับข้อความแรกจาก C# เพื่อบอกว่าจะให้ดูไฟล์ไหน
    log_type = await websocket.receive_text()
    
    # log_file_path = None
    # if log_type == "scaling":
    #     log_file_path = LOG_DIR / "scaling_output.log"
    # elif log_type == "training":
    #     log_file_path = LOG_DIR / "training_output.log"
    # else:
    #     await websocket.send_text("Error: Invalid log type specified.")
    #     await websocket.close()
    #     return
    log_files = {
        "simulation": LOG_DIR / "simulation_output.log",
        "scaling": LOG_DIR / "scaling_output.log",
        "training": LOG_DIR / "training_output.log"
    }
    
    log_file_path = log_files.get(log_type)

    if not log_file_path:
        await websocket.send_text(f"Error: Invalid log type '{log_type}' specified.")
        await websocket.close()
        return



    # เริ่ม "แอบดู" ไฟล์ที่ C# ร้องขอ
    try:
        await tail_log_file(websocket, log_file_path)
    except WebSocketDisconnect:
        print(f"Client disconnected from log: {log_type}")
    finally:
        print("WebSocket connection closed.")


# --- HTTP Endpoints (เหมือนเดิม แต่สะอาดขึ้น) ---
# ในไฟล์ main.py
# ในไฟล์ main.py
def start_background_process(script_path: Path, log_name: str):
    if not script_path.exists():
        return {"error": f"Script not found: {script_path}"}
    
    # "จุดยืน" คือโถงกลางของห้องสมุด (ถูกต้องแล้ว)
    working_dir = PROJECT_ROOT

    # **แก้ไข:** สร้าง "แผนที่เดินทาง" จากจุดยืนไปยังไฟล์เป้าหมาย!
    script_relative_path = script_path.relative_to(working_dir)

       # --- เพิ่ม 2 บรรทัดนี้เข้าไป! ---
    print(f"DEBUG: Current Working Directory (cwd) = {working_dir}")
    print(f"DEBUG: Command to run = python -u {script_relative_path}")
    # --- แค่นี้พอ! ---

    log_out = open(LOG_DIR / f"{log_name}_output.log", "w", encoding='utf-8')
    log_err = open(LOG_DIR / f"{log_name}_error.log", "w", encoding='utf-8')

    script_env = os.environ.copy()
    script_env["PYTHONUTF8"] = "1"

    subprocess.Popen(
        [PYTHON_EXECUTABLE, "-u", str(script_relative_path)], # <--- **ใช้ "แผนที่เดินทาง" นี้!**
        cwd=str(working_dir),
        stdout=log_out,
        stderr=log_err,
        env=script_env
    )
    
    return {"message": f"Process '{log_name}' started. Monitoring log file..."}


@app.post("/start-simulation")
async def trigger_simulation():
    return start_background_process(SIMULATION_SCRIPT_PATH, "simulation")

@app.post("/start-scaling")
async def trigger_scaling():
    return start_background_process(RESCALE_SCRIPT_PATH, "scaling")

# @app.post("/start-training")
# async def trigger_training():
#     return start_background_process(TRAIN_SCRIPT_PATH, "training")


# add debug print to show current working directory and command to run
# print inside the script