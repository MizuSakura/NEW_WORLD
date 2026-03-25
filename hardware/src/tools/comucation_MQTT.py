#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/tools/comucation_MQTT.py
from __future__ import print_function
import time
import json
import paho.mqtt.client as mqtt
import yaml
from pathlib import Path

# ======================================================
# CONFIG LOADER (Fixed for Python 3.6.9 & Folder Structure)
# ======================================================
def load_config():
    current_file = Path(__file__).resolve()
    # Path calculation based on: hardware/src/utils/your_file.py
    # parents[3] leads to my_project/
    config_path = current_file.parent.parent.parent / "config" / "hardware.yaml"
    
    # Check fallback path if not found
    if not config_path.exists():
        config_path = current_file.parent.parent.parent / "hardware" / "config" / "hardware.yaml"
        
    if not config_path.exists():
        raise FileNotFoundError("Could not find hardware.yaml")

    # Python 3.6.9 open() works best with str(Path)
    with open(str(config_path), "r") as f:
        return yaml.safe_load(f)

_cfg = load_config()

BROKER    = _cfg["mqtt"]["broker"]
PORT      = _cfg["mqtt"]["port"]
CLIENT_ID = _cfg["mqtt"]["client_id"]
TOPIC     = _cfg["mqtt"]["topics"]["action"]
USER      = _cfg["mqtt"]["username"]
PASS      = _cfg["mqtt"]["password"]

# ======================================================
# SYSTEM STATE
# ======================================================
current_command = None
last_command_time = time.time()

# ======================================================
# MQTT CALLBACKS
# ======================================================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker")
        client.subscribe(TOPIC)
        # Use .format() instead of f-string
        print("📡 Subscribed to topic: {}".format(TOPIC))
    else:
        print("❌ MQTT connection failed, rc = {}".format(rc))

def on_message(client, userdata, msg):
    global current_command, last_command_time

    # Explicitly decode bytes to string (Required for Python 3.6)
    payload = msg.payload.decode("utf-8")
    try:
        current_command = json.loads(payload)
    except Exception:
        current_command = payload

    last_command_time = time.time()
    print("📥 Receive command: {}".format(current_command))

def main():
    # ---------- MQTT SETUP ----------
    # Paho-mqtt 1.x (standard on 3.6.9) uses this signature
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.username_pw_set(USER, PASS)

    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting to Broker: {}...".format(BROKER))
    try:
        client.connect(BROKER, PORT, keepalive=30)
        
        # Start the background thread to handle messages
        client.loop_start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        print("❌ Fatal Error: {}".format(e))

if __name__ == "__main__":
    main()
