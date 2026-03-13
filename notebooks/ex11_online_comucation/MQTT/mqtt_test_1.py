import json
import time
import random
import threading
import queue
import paho.mqtt.client as mqtt

# ===============================
# CONFIG
# ===============================
BROKER = "100.85.77.73"
PORT = 1883

TRANSITION_TOPIC = "project/rl/nvidia01/edge/telemetry/transition"
WEIGHT_TOPIC = "project/rl/nvidia01/sync/weight_update"

USER = "yessuskhonpui"
PASS = "246810"

# ===============================
# QUEUE (Cloud Side)
# ===============================
transition_queue = queue.Queue(maxsize=10000)
replay_buffer = []

# ===============================
# EDGE MQTT CLIENT
# ===============================
edge_client = mqtt.Client(
    client_id="edge-sim",
    protocol=mqtt.MQTTv311,
    callback_api_version=mqtt.CallbackAPIVersion.VERSION1
)

def on_connect_edge(client, userdata, flags, rc):
    if rc == 0:
        print("[EDGE] Connected to broker")
        client.subscribe(WEIGHT_TOPIC, qos=1)
    else:
        print("[EDGE] Connection failed:", rc)

def on_weight_update(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"[EDGE] 🔄 New weight received: {data}")
    except Exception as e:
        print("[EDGE] Weight parse error:", e)

edge_client.username_pw_set(USER, PASS)
edge_client.on_connect = on_connect_edge
edge_client.on_message = on_weight_update
edge_client.connect(BROKER, PORT)
edge_client.loop_start()

# ===============================
# CLOUD MQTT CLIENT
# ===============================
cloud_client = mqtt.Client(
    client_id="cloud-sim",
    protocol=mqtt.MQTTv311,
    callback_api_version=mqtt.CallbackAPIVersion.VERSION1
)

def on_connect_cloud(client, userdata, flags, rc):
    if rc == 0:
        print("[CLOUD] Connected to broker")
        client.subscribe(TRANSITION_TOPIC, qos=0)
    else:
        print("[CLOUD] Connection failed:", rc)

def on_transition(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        transition_queue.put_nowait(data)
        print(f"[CLOUD] 📥 Received ID={data['id']}")
    except queue.Full:
        print("[CLOUD] ⚠️ Queue full, dropping transition")
    except Exception as e:
        print("[CLOUD] Parse error:", e)

cloud_client.username_pw_set(USER, PASS)
cloud_client.on_connect = on_connect_cloud
cloud_client.on_message = on_transition
cloud_client.connect(BROKER, PORT)
cloud_client.loop_start()

# ===============================
# EDGE SIMULATOR (50Hz)
# ===============================
def edge_simulator():
    transition_id = 0

    while True:
        transition = {
            "id": transition_id,
            "state": [random.random(), random.random()],
            "action": random.randint(0, 10),
            "reward": random.random(),
            "next_state": [random.random(), random.random()],
            "done": False
        }

        edge_client.publish(
            TRANSITION_TOPIC,
            json.dumps(transition),
            qos=0
        )

        print(f"[EDGE] 🚀 Sent ID={transition_id}")

        transition_id += 1
        time.sleep(0.02)  # 50Hz

# ===============================
# CLOUD TRAINER THREAD
# ===============================
def trainer():
    global replay_buffer

    while True:
        try:
            data = transition_queue.get(timeout=1)
            replay_buffer.append(data)
        except queue.Empty:
            pass

        if len(replay_buffer) >= 100:
            print(f"[CLOUD] 🧠 Training... buffer={len(replay_buffer)}")

            time.sleep(0.5)  # simulate training

            weight_msg = {
                "version": time.time()
            }

            cloud_client.publish(
                WEIGHT_TOPIC,
                json.dumps(weight_msg),
                qos=1
            )

            replay_buffer.clear()

# ===============================
# START THREADS
# ===============================
threading.Thread(target=edge_simulator, daemon=True).start()
threading.Thread(target=trainer, daemon=True).start()

print("System started... Press Ctrl+C to stop")

while True:
    time.sleep(1)