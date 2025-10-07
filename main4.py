import sys
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
import os
import asyncio

# --- Configuration (เหมือนเดิม) ---
PYTHON_EXECUTABLE = sys.executable 
RESCALE_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Rescal_data.py")
TRAIN_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Model_LSTM.py")
SCRIPT_WORKING_DIR = RESCALE_SCRIPT_PATH.parent
LOG_DIR = SCRIPT_WORKING_DIR.parent.parent / "logs"
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
    
    log_file_path = None
    if log_type == "scaling":
        log_file_path = LOG_DIR / "scaling_output.log"
    elif log_type == "training":
        log_file_path = LOG_DIR / "training_output.log"
    else:
        await websocket.send_text("Error: Invalid log type specified.")
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
def start_background_process(script_path: Path, log_name: str):
    if not script_path.exists():
        return {"error": f"Script not found: {script_path}"}
    
    log_out = open(LOG_DIR / f"{log_name}_output.log", "w", encoding='utf-8')
    log_err = open(LOG_DIR / f"{log_name}_error.log", "w", encoding='utf-8')

    script_env = os.environ.copy()
    script_env["PYTHONUTF8"] = "1"

    subprocess.Popen(
        [PYTHON_EXECUTABLE, "-u", str(script_path.name)],
        cwd=str(SCRIPT_WORKING_DIR),
        stdout=log_out,
        stderr=log_err,
        env=script_env
    )
    
    return {"message": f"Process '{log_name}' started. Monitoring log file..."}

@app.post("/start-scaling")
async def trigger_scaling():
    return start_background_process(RESCALE_SCRIPT_PATH, "scaling")

@app.post("/start-training")
async def trigger_training():
    return start_background_process(TRAIN_SCRIPT_PATH, "training")

# add websocket for real-time log streaming
# client (C#) จะเชื่อมต่อมาที่ /ws/log และส่งข้อความ "scaling" หรือ "training" เพื่อเลือกไฟล์ log ที่ต้องการดู
# ฟังก์ชัน tail_log_file จะคอยอ่านไฟล์ log และส่งบรรทัดใหม่ๆ ผ่าน WebSocket ทันที
# ใช้ asyncio เพื่อให้สามารถรอและส่งข้อมูลแบบไม่บล็อก
# ส่วน HTTP endpoints ยังคงใช้สำหรับเริ่มกระบวนการ scaling และ training เหมือนเดิม
# แต่ตอนนี้จะสะอาดขึ้นเพราะแยกหน้าที่การทำงานชัดเจน