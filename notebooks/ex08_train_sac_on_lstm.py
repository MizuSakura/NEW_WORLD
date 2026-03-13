"""
Example: Train SAC Agent on LSTM World Model
=====================================================
ตัวอย่างการใช้ train_SAC_on_LSTM.py เพื่อฝึก Agent บน LSTM World Model

ขั้นตอนการทำงาน:
1. โหลด LSTM Model checkpoint
2. โหลด Scaler files
3. สร้าง SAC Agent
4. ฝึก Agent บน LSTM World Model Environment
5. บันทึก Agent
6. ประเมิน Agent
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.agent.SAC_Agent import SACAgent
from src.trainer.train_SAC_on_LSTM import (
    LSTMWorldModelEnv,
    train_agent_on_lstm_env,
    evaluate_agent_on_lstm_env
)


# ================================================================
# 📋 Configuration
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"D:\Project_end\New_world\my_project\models\checkpoint\lstm_world_model.pt"
SCALER_ZIP = r"D:\Project_end\New_world\my_project\config\test1_scalers.zip"

CHECKPOINT_DIR = Path(r"D:\Project_end\New_world\my_project\models\checkpoint\lstm_training")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 1️⃣ | Create LSTM Environment (สำหรับเช็ค dimension)
# ================================================================
print("[1/4] Creating LSTM Environment...")
env = LSTMWorldModelEnv(
    model_path=MODEL_PATH,
    scaler_zip=SCALER_ZIP,
    reward_type="adaptive",
    device=DEVICE
)

print(f"  State dimension: {env.observation_space.shape}")
print(f"  Action dimension: {env.action_space.shape}")
print(f"  Action range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = float(env.action_space.low[0])
max_action = float(env.action_space.high[0])

env.close()

# ================================================================
# 2️⃣ | Create SAC Agent
# ================================================================
print("\n[2/4] Creating SAC Agent...")
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    min_action=min_action,
    max_action=max_action,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    replay_capacity=100000,
    device=DEVICE,
    logger_status=False,
    simple_layers_actor=2,
    simple_hidden_actor=256,
    simple_layers_critic=2,
    simple_hidden_critic=256,
)
print(f"  Using device: {agent.device}")

# ================================================================
# 3️⃣ | Train Agent on LSTM World Model
# ================================================================
print("\n[3/4] Training Agent on LSTM World Model...")
agent, episode_rewards, episode_lengths = train_agent_on_lstm_env(
    model_path=MODEL_PATH,
    scaler_zip=SCALER_ZIP,
    agent=agent,
    num_episodes=100,
    max_steps=500,
    batch_size=64,
    update_interval=1,
    save_interval=10,
    save_dir=CHECKPOINT_DIR,
    device=DEVICE,
    verbose=True
)

# ================================================================
# 4️⃣ | Evaluate Agent
# ================================================================
print("\n[4/4] Evaluating Trained Agent...")
eval_rewards, trajectories = evaluate_agent_on_lstm_env(
    model_path=MODEL_PATH,
    scaler_zip=SCALER_ZIP,
    agent=agent,
    num_episodes=10,
    max_steps=500,
    device=DEVICE,
    verbose=True
)

# ================================================================
# 📊 Save Final Model & Plot Results
# ================================================================
final_model_path = CHECKPOINT_DIR / "agent_final.pt"
agent.save_model(path=str(final_model_path))
print(f"\n✅ Model saved to: {final_model_path}")

# Plot training results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Episode rewards
axes[0, 0].plot(episode_rewards)
axes[0, 0].set_title("Training: Episode Rewards")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Total Reward")
axes[0, 0].grid(True)

# Episode lengths
axes[0, 1].plot(episode_lengths)
axes[0, 1].set_title("Training: Episode Lengths")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Steps")
axes[0, 1].grid(True)

# Evaluation rewards
axes[1, 0].plot(eval_rewards, marker="o")
axes[1, 0].set_title("Evaluation: Episode Rewards")
axes[1, 0].set_xlabel("Eval Episode")
axes[1, 0].set_ylabel("Total Reward")
axes[1, 0].grid(True)

# Sample trajectory
if len(trajectories) > 0:
    axes[1, 1].plot(trajectories[0]["rewards"])
    axes[1, 1].set_title("Sample Trajectory: Rewards")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(CHECKPOINT_DIR / "training_results.png", dpi=100)
print(f"📊 Results figure saved to: {CHECKPOINT_DIR / 'training_results.png'}")
plt.show()

print("\n" + "="*60)
print("✨ Training Complete!")
print("="*60)
