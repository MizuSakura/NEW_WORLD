import paho.mqtt.client as mqtt
import time
import json
import numpy as np 
from src.agent.SAC_Agent import SACAgent
import gymnasium as gym
import numpy as np
from pathlib import Path
from src.environment.Apply_real_env import Real_env_remote
import time


env = Real_env_remote(ip_host="192.168.1.100",
                    port=502,
                    min_action = 0,
                    max_action = 10,
                    setpoint= 5,
                    delay_of_action = 0.2,
                    address_sensor=1,
                    address_actuator=1025,
                    )
state = env.reset()

state_dim = env.state_dim
action_dim = env.action_dim
min_action = env.min_action
max_action = env.max_action

print("State dim:", state_dim)
print("Action dim:", action_dim)
print("Action range:", min_action, max_action)

agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    min_action=np.array([min_action]),
    max_action=np.array([max_action]),
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.4,
    logger_status=True
)

final_model_path = r"D:\Project_end\New_world\my_project\models\checkpoint\sac_checkpoint.pt"
agent.load_model(path=final_model_path)