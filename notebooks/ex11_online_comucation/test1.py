"""
MQTT Autonomous + Remote Override Controller
--------------------------------------------
- ไม่มี MQTT command -> ทำงานอัตโนมัติ
- มี MQTT command -> ทำตาม payload ทันที
- ถ้า command หายไปเกิน TIMEOUT -> กลับสู่โหมดอัตโนมัติ

Author: Research Assistant
"""

import time
import json
import paho.mqtt.client as mqtt

# ======================================================
# MQTT CONFIG
# ======================================================
BROKER = "100.85.77.73"
PORT = 1883
USER, PASS = "yessuskhonpui", "246810"
TOPIC = "RL/COMMAND_CONTROL/ACTION"
CLIENT_ID = "python-subscriber-01"

# ======================================================
# SYSTEM STATE
# ======================================================
current_command = None
last_command_time = 0.0
COMMAND_TIMEOUT = 5.0   # seconds

# ======================================================
# MQTT CALLBACKS
# ======================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker")
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to topic: {TOPIC}")
    else:
        print("❌ MQTT connection failed, rc =", rc)

def on_message(client, userdata, msg):
    global current_command, last_command_time

    payload = msg.payload.decode()
    try:
        current_command = json.loads(payload)
    except json.JSONDecodeError:
        current_command = payload

    last_command_time = time.time()
    print("📥 Receive command:", current_command)

# ======================================================
# CONTROL LOGIC
# ======================================================
def autonomous_task():
    """
    Default behavior when no MQTT command is active
    """
    print("🤖 AUTO MODE : ระบบทำงานตามปกติ")

def execute_command(cmd):
    """
    Execute command from MQTT payload
    """
    print("🎮 REMOTE MODE :", cmd)

    # ----- ตัวอย่างการตีความ payload -----
    if isinstance(cmd, dict):
        action = cmd.get("action", None)
        value = cmd.get("value", None)
        print(f"    Action = {action}, Value = {value}")

    elif isinstance(cmd, str):
        if cmd == "STOP":
            print("    หยุดระบบ")
        elif cmd == "LEFT":
            print("    หมุนซ้าย")
        elif cmd == "RIGHT":
            print("    หมุนขวา")

# ======================================================
# MAIN
# ======================================================
def main():
    # ---------- MQTT SETUP ----------
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.username_pw_set(USER, PASS)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()   # ⭐ สำคัญ: non-blocking

    print("🚀 System started")

    # ---------- MAIN LOOP ----------
    while True:
        now = time.time()

        if current_command is not None and (now - last_command_time) < COMMAND_TIMEOUT:
            execute_command(current_command)
        else:
            autonomous_task()

        time.sleep(0.1)

# ======================================================
if __name__ == "__main__":
    main()
