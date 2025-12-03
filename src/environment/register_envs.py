#my_project\src\environment\register_envs.py
import gymnasium as gym
from gymnasium.envs.registration import register

# 1) Register your custom env to gym
register(
    id="RCTankEnv-v0",
    entry_point="src.environment.RCTankEnv_gym:RCTankEnv",   # path:module:ClassName
    max_episode_steps=1000,
)