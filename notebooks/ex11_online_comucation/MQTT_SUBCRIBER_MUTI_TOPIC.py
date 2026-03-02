import json
import paho.mqtt.client as mqtt

BROKER = "100.85.77.73"
PORT = 1883
USER, PASS = "yessuskhonpui", "246810"

CLIENT_ID = "jetson-nvidia01-subscriber"

TOPICS = [
    ("project/rl/nvidia01/edge/telemetry/transition", 0),
    ("project/rl/nvidia01/edge/telemetry/status", 0),
    ()
]

# ===============================
# Callback: On Connect
# ===============================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(TOPICS)
        print("Subscribed to multiple topics")
    else:
        print("Connection failed:", rc)

# ===============================
# Callback: On Message
# ===============================
def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()

    print(f"\n[Incoming] Topic: {topic}")

    # ---------------------------
    # Transition Data
    # ---------------------------
    if topic.endswith("transition"):
        try:
            data = json.loads(payload)
            print("Transition Received:")
            print(data)

            # ตัวอย่าง expected structure
            # {
            #   "state": [...],
            #   "action": ...,
            #   "reward": ...,
            #   "next_state": [...],
            #   "done": false
            # }

            # TODO: push to replay buffer

        except Exception as e:
            print("Transition parsing error:", e)

    # ---------------------------
    # Status Data
    # ---------------------------
    elif topic.endswith("status"):
        print("Status Update:")
        print(payload)

        # เช่น:
        # "online"
        # "training"
        # "error"

    else:
        print("Unhandled topic")

# ===============================
# Create Client
# ===============================
client = mqtt.Client(client_id=CLIENT_ID)
client.username_pw_set(USER, PASS)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)

print("Starting MQTT loop...")
client.loop_forever()