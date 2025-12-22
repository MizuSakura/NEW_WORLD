import time
import json
import paho.mqtt.client as mqtt

BROKER = "100.85.77.73"
PORT = 1883
USER, PASS = "yessuskhonpui", "246810"
TOPIC = "RL/COMMAND_CONTROL/ACTION"
CLIENT_ID = "python-subscriber-01"

# ======================================================
# SYSTEM STATE
# ======================================================
current_command = None

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

def main():

     # ---------- MQTT SETUP ----------
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.username_pw_set(USER, PASS)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start() 