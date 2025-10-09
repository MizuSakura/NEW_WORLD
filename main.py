import sys
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
import os
import asyncio
from pydantic import BaseModel # เครื่องมือสร้างแบบ form

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

# --- "แบบฟอร์ม" รับข้อมูลจาก C# (ใหม่!) ---
class SimulationParams(BaseModel):
    R: float = 1.5
    C: float = 2.0
    dt: float = 0.01
    control_mode: str = 'voltage'
    setpoint_level: float = 5.0
    level_max: float = 10.0
    max_action_volt: float = 24.0
    max_action_current: float = 5.0
    time_sim: float = 30.0
    signal_type: str = 'pwm'
    amplitude: float = 12.0
    duty: float = 0.5
    freq: float = 1.0

class BatchSimulationParams(BaseModel):
    # พารามิเตอร์ Env พื้นฐาน
    R: float = 1.5
    C: float = 2.0
    dt: float = 0.01
    setpoint_level: float = 5.0
    time_sim: float = 30.0
    amplitude: float = 12.0
    # พารามิเตอร์สำหรับ Batch
    duty_start: float = 0.1
    duty_end: float = 1.0
    duty_steps: int = 10
    freq_start: float = 0.1
    freq_end: float = 2.0
    freq_steps: int = 5

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
def start_background_process(script_path: Path, log_name: str, args_dict: dict | None = None):
    if not script_path.exists():
        return {"error": f"Script not found: {script_path}"}
    
    # "จุดยืน" คือโถงกลางของห้องสมุด (ถูกต้องแล้ว)
    working_dir = PROJECT_ROOT

    # **แก้ไข:** สร้าง "แผนที่เดินทาง" จากจุดยืนไปยังไฟล์เป้าหมาย!
    script_relative_path = script_path.relative_to(working_dir)
    module_path = ".".join(script_relative_path.with_suffix("").parts)

    # --- หัวใจของการแก้ไข! ---
    # สร้าง "รายการคำสั่งพิเศษ" จากข้อมูลที่ได้รับ
    command = [PYTHON_EXECUTABLE, "-u", "-m", module_path]
    if args_dict:
        for key, value in args_dict.items():
            # เช็คก่อนว่าค่าที่ได้เป็น boolean รึเปล่า
            if isinstance(value, bool):
                # ถ้าเป็น True... ก็แค่ "ดีดสวิตช์"
                if value:
                    command.append(f"--{key}")
                # ถ้าเป็น False... ก็ "ไม่ต้องไปยุ่งกับมัน"
            else:
                # ถ้าเป็นค่าอื่นๆ ก็ใส่ตามปกติ
                command.append(f"--{key}")
                command.append(str(value))
    


    log_out = open(LOG_DIR / f"{log_name}_output.log", "w", encoding='utf-8')
    log_err = open(LOG_DIR / f"{log_name}_error.log", "w", encoding='utf-8')

    script_env = os.environ.copy()
    script_env["PYTHONUTF8"] = "1"

    subprocess.Popen(
        command, # <-- ใช้ "รายการคำสั่งพิเศษ" ที่สร้างขึ้นใหม่!
        cwd=str(working_dir),
        stdout=log_out,
        stderr=log_err,
        env=script_env
    )
    
    return {"message": f"Process '{log_name}' started. Monitoring log file..."}



# @app.post("/start-simulation")
# async def trigger_simulation(params: SimulationParams): # <-- รับ "แบบฟอร์ม" จาก C#!
#     # แปลง "แบบฟอร์ม" เป็น Dictionary แล้วส่งต่อ
#     return start_background_process(SIMULATION_SCRIPT_PATH, "simulation", args_dict=params.model_dump())

@app.post("/start-simulation")
async def trigger_simulation(params: SimulationParams):
    # เพิ่ม --batch_mode False เข้าไป เพื่อให้แน่ใจว่าทำงานถูกโหมด
    args = params.model_dump()
    args['batch_mode'] = False
    return start_background_process(SIMULATION_SCRIPT_PATH, "simulation", args_dict=args)

# --- ประตูสำหรับ "จัดเลี้ยง" (ใหม่!) ---
@app.post("/start-simulation-batch")
async def trigger_batch_simulation(params: BatchSimulationParams):
    # แปลง "แบบฟอร์ม" เป็น Dictionary
    args = params.model_dump()
    # เพิ่ม --batch_mode True เพื่อเปิดใช้งาน "โหมดจัดเลี้ยง"
    args['batch_mode'] = True
    return start_background_process(SIMULATION_SCRIPT_PATH, "simulation", args_dict=args)


@app.post("/start-scaling")
async def trigger_scaling():
    # (ฟังก์ชันนี้ยังทำงานเหมือนเดิม เพราะยังไม่มีพารามิเตอร์)
    return start_background_process(RESCALE_SCRIPT_PATH, "scaling")





# add_class BatchSimulationParams(BaseModel):_for batch sim parameters
# add_endpoint /start-simulation-batch to handle batch simulation requests

# แก้ args_dict ใน start_background_process เพื่อรองรับ boolean flags
# เพิ่ม --batch_mode ใน args_dict เมื่อเรียก /start-simulation-batch

