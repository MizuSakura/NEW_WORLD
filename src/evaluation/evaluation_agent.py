import numpy as np
import matplotlib.pyplot as plt
from src.agent.SAC_Agent import SACAgent
import gymnasium as gym
from pathlib import Path
from src.environment.RCTankEnv_gym import RCTankEnv


def test_agent(env, agent, episodes=20, max_steps=1000):
    returns = []
    trajectories = []   # เก็บ state/action/reward เพื่อนำไป plot

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        states = []
        actions = []
        rewards = []

        for step in range(max_steps):
            # ใช้ deterministic เพื่อดู performance จริง
            action = agent.select_action(state, deterministic=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()

            # เก็บข้อมูลสำหรับ plot
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            episode_reward += reward
            state = next_state

            if done:
                break

        returns.append(episode_reward)
        trajectories.append({
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
        })

        print(f"[TEST] Episode {ep+1}/{episodes} Reward = {episode_reward:.2f}| status train:{done} ")

    return returns, trajectories

env = RCTankEnv(render_mode="human")

state, _ = env.reset()
state_dim = state.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low
max_action = env.action_space.high

print("State dim:", state_dim)
print("Action dim:", action_dim)
print("Action range:", min_action, max_action)

agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    min_action=min_action,
    max_action=max_action,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.4,
    logger_status=True
)

final_model_path = r"D:\Project_end\New_world\my_project\models\sac_checkpoint.pt"
agent.load_model(path=final_model_path)

test_returns, test_traj = test_agent(env, agent, episodes=5, max_steps=200)
plt.figure(figsize=(12,4))
plt.plot(test_traj[1]["rewards"], label="Reward per step")
plt.title("Reward curve (Test Episode 1)")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(test_traj[1]["actions"], label="Action", alpha=0.7)
plt.title("Action output (Test Episode 1)")
plt.xlabel("Step")
plt.ylabel("Action Value")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(test_traj[1]["states"])
plt.title("State trajectory (Test Episode 1)")
plt.xlabel("Step")
plt.ylabel("State Value")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(test_returns, marker="o")
plt.title("Total test return per episode")
plt.xlabel("Test Episode")
plt.ylabel("Total Return")
plt.grid(True)
plt.show()
