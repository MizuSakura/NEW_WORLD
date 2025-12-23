import paho.mqtt.client as mqtt

BROKER = "100.85.77.73"
PORT = 1883
USER, PASS = "yessuskhonpui", "246810"
TOPIC = "RL/COMMAND_CONTROL/ACTION"
CLIENT_ID = "python-subscriber-01"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(TOPIC, qos=1)
    else:
        print("Connection failed:", rc)

def on_message(client, userdata, msg):
    action = msg.payload.decode()
    print(f"Receive action -> {action}")
    # ตรงนี้คือจุดที่คุณเอา action ไปใช้จริง
    # เช่น ส่งไป PLC / Modbus / Plant

client = mqtt.Client(client_id=CLIENT_ID)
client.username_pw_set(USER, PASS)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
