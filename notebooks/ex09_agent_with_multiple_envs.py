"""
Example: Using Trained Agent with Different Environments
=====================================================
แสดงวิธีการใช้ Agent ที่ฝึกแล้วกับ environment ต่างๆ

สามารถใช้กับ:
1. LSTM World Model Environment
2. RCTankEnv (Gymnasium)
3. Environment อื่นๆ ที่ compatible กับ Gymnasium interface
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.agent.SAC_Agent import SACAgent
from src.trainer.train_SAC_on_LSTM import LSTMWorldModelEnv, evaluate_agent_on_lstm_env
from src.environment.RCTankEnv_gym import RCTankEnv
from src.environment.noise_manager import NoiseManager


# ================================================================
# Configuration
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGENT_MODEL_PATH = r"D:\Project_end\New_world\my_project\models\checkpoint\Autosave.pt"

LSTM_MODEL_PATH = r"D:\Project_end\New_world\my_project\models\checkpoint\lstm_world_model.pt"
LSTM_SCALER_ZIP = r"D:\Project_end\New_world\my_project\config\test1_scalers.zip"


# ================================================================
# 1️⃣ Loading Trained Agent
# ================================================================
print("[1] Loading Trained Agent...")
print(f"  Using device: {DEVICE}")

# You need to know agent dimensions from training
state_dim = 12  # ตรวจสอบจากการฝึก
action_dim = 1

# Create agent with same configuration as training
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    min_action=0.0,
    max_action=10.0,  # Adjust based on your environment
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    device=DEVICE,
    logger_status=False,
)

# Load trained model
agent.load_model(path=AGENT_MODEL_PATH)
print(f"  ✅ Agent loaded from: {AGENT_MODEL_PATH}")


# ================================================================
# 2️⃣ Use Agent with LSTM World Model Environment
# ================================================================
print("\n[2] Evaluating on LSTM World Model Environment...")
lstm_rewards, lstm_trajectories = evaluate_agent_on_lstm_env(
    model_path=LSTM_MODEL_PATH,
    scaler_zip=LSTM_SCALER_ZIP,
    agent=agent,
    num_episodes=5,
    max_steps=500,
    device=DEVICE,
    verbose=True
)
print(f"  Average LSTM Reward: {np.mean(lstm_rewards):.3f}")


# ================================================================
# 3️⃣ Use Agent with RCTankEnv (Real Environment)
# ================================================================
print("\n[3] Evaluating on RCTankEnv (Real Environment)...")

noise_manager = NoiseManager(enabled=False)
env = RCTankEnv(
    render_mode=None,
    noise_manager=noise_manager
)

# Get environment dimensions
test_state, _ = env.reset()
env_state_dim = test_state.shape[0]
env_action_dim = env.action_space.shape[0]

print(f"  RCTankEnv State dim: {env_state_dim}")
print(f"  RCTankEnv Action dim: {env_action_dim}")

# Evaluate on RCTankEnv
rc_tank_rewards = []
rc_tank_trajectories = []

for ep in range(5):
    state, _ = env.reset()
    episode_reward = 0.0
    states, actions, rewards = [], [], []
    
    for step in range(500):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        states.append(state.copy())
        actions.append(action)
        rewards.append(reward)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    rc_tank_rewards.append(episode_reward)
    rc_tank_trajectories.append({
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
    })
    
    print(f"  Episode {ep+1}/5: Reward = {episode_reward:.2f}")

env.close()
print(f"  Average RCTankEnv Reward: {np.mean(rc_tank_rewards):.3f}")


# ================================================================
# 📊 Comparison & Visualization
# ================================================================
print("\n[4] Generating Comparison Plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# LSTM Rewards
axes[0, 0].plot(lstm_rewards, marker="o")
axes[0, 0].set_title("LSTM Env: Episode Rewards")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].grid(True)

# RCTank Rewards
axes[0, 1].plot(rc_tank_rewards, marker="s")
axes[0, 1].set_title("RCTankEnv: Episode Rewards")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Reward")
axes[0, 1].grid(True)

# Comparison
axes[0, 2].bar(["LSTM", "RCTank"], [np.mean(lstm_rewards), np.mean(rc_tank_rewards)])
axes[0, 2].set_title("Average Reward Comparison")
axes[0, 2].set_ylabel("Average Reward")
axes[0, 2].grid(True, alpha=0.3)

# LSTM Sample Trajectory
if len(lstm_trajectories) > 0:
    axes[1, 0].plot(lstm_trajectories[0]["rewards"])
    axes[1, 0].set_title("LSTM: Sample Trajectory Rewards")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].grid(True)

# RCTank Sample Trajectory
if len(rc_tank_trajectories) > 0:
    axes[1, 1].plot(rc_tank_trajectories[0]["rewards"])
    axes[1, 1].set_title("RCTank: Sample Trajectory Rewards")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(True)

# Actions Distribution
if len(rc_tank_trajectories) > 0:
    axes[1, 2].hist(rc_tank_trajectories[0]["actions"], bins=20)
    axes[1, 2].set_title("RCTank: Action Distribution")
    axes[1, 2].set_xlabel("Action Value")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("agent_evaluation_comparison.png", dpi=100)
print("  📊 Comparison plot saved to: agent_evaluation_comparison.png")
plt.show()


# ================================================================
# 📝 Summary
# ================================================================
print("\n" + "="*60)
print("✨ Agent Evaluation Complete!")
print("="*60)
print(f"\nSummary:")
print(f"  LSTM Env Average Reward:    {np.mean(lstm_rewards):.3f}")
print(f"  RCTankEnv Average Reward:   {np.mean(rc_tank_rewards):.3f}")
print(f"\nAgent can be reused with any Gymnasium-compatible environment!")
print("="*60)
