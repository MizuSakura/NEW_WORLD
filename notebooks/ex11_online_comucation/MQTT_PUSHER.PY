import paho.mqtt.client as mqtt
import time
import json
import numpy as np 

BROKER = "100.85.77.73"
PORT = 1883
USER, PASS = "yessuskhonpui", "246810"
TOPIC = "RL/COMMAND_CONTROL/ACTION"
CLIENT_ID = "COMPUTER_CONTROL"

client = mqtt.Client(client_id=CLIENT_ID)
client.username_pw_set(USER, PASS)
client.connect(BROKER, PORT)

while True:
    action = {
        "value": np.random.uniform(0,10),
        "mode": "auto",
        "timestamp": time.time()
    }

    client.publish(
        TOPIC,
        json.dumps(action),
        qos=1,
        retain=False
    )

    print("Send action:", action)
    time.sleep(2)
