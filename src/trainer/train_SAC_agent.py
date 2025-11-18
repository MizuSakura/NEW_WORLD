#my_project\src\trainer\train_SAC_agent.py
from src.agent.SAC_Agent import SACAgent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.environment.RC_Tank_env import RCTankEnv

# สร้าง environment
#env = gym.make("Pendulum-v1", render_mode="human")
env = RCTankEnv()
state, _ = env.reset()
state_dim = state.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low
max_action = env.action_space.high

print("State dim:", state_dim)
print("Action dim:", action_dim)
print("Action range:", min_action, max_action)


# สร้าง agent
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    min_action=min_action,
    max_action=max_action,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.4,
    logger_status= True
)

# ===== Training Parameters =====
episodes = 500
max_steps = 200
batch_size = 1080

rewards_history = []

# ===== Training Loop =====
for ep in range(episodes):

    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # 1. เลือก action
        action = agent.select_action(state)
       

        # 2. Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. เก็บลง Replay Buffer
        agent.replay_buffer.push(
            state, action, reward, next_state, float(done)
        )

        # 4. อัปเดต network
        agent.update(batch_size)
        
        state = next_state
        episode_reward += reward

        if done:
            break

    rewards_history.append(episode_reward)
    print(f"Episode {ep+1}/{episodes} Reward: {episode_reward:.2f}")
agent.save_model(path=rf"D:\Project_end\New_world\my_project\notebooks\Test_RCTankEnv.pt")