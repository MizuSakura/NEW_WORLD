#my_project\src\trainer\train_SAC_agent.py

from src.agent.SAC_Agent import SACAgent
import gymnasium as gym
import numpy as np
from pathlib import Path
from src.environment.RC_Tank_env import RC_Tank_Env
from src.environment.RCTankEnv_gym import RCTankEnv


# ======================================================
# 1) Create environment
# ======================================================
# env = gym.make("Pendulum-v1", render_mode="human")
env = RCTankEnv(render_mode="human")

state, _ = env.reset()
state_dim = state.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low
max_action = env.action_space.high

print("State dim:", state_dim)
print("Action dim:", action_dim)
print("Action range:", min_action, max_action)


# ======================================================
# 2) Create SAC agent
# ======================================================
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


# ======================================================
# 3) Training Hyperparameters
# ======================================================
episodes = 10000
max_steps = 200
batch_size = 1080

rewards_history = []

checkpoint_path = Path(r"D:\Project_end\New_world\my_project\models\sac_checkpoint.pt")
autosave_every = 10  # Save checkpoint every N episodes


# ======================================================
# 4) Auto-Load Checkpoint (if exists)
# ======================================================
start_episode = 1

if checkpoint_path.exists():
    print("\n[Trainer] Found checkpoint. Loading...")
    start_episode = agent.load_checkpoint(checkpoint_path) + 1
    print(f"[Trainer] Resuming training from episode {start_episode}\n")
else:
    print("\n[Trainer] No checkpoint found. Starting from episode 1\n")


# ======================================================
# 5) Training Loop
# ======================================================
for ep in range(start_episode, episodes + 1):

    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):

        # -----------------------------------------
        # (1) Select action
        # -----------------------------------------
        action = agent.select_action(state)

        # -----------------------------------------
        # (2) Step environment
        # -----------------------------------------
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()

        done = terminated or truncated

        # -----------------------------------------
        # (3) Store transition
        # -----------------------------------------
        agent.replay_buffer.push(state, action, reward, next_state, float(done))

        # -----------------------------------------
        # (4) SAC update
        # -----------------------------------------
        agent.update(batch_size)

        state = next_state
        episode_reward += reward

        if done:
            break

    rewards_history.append(episode_reward)
    print(f"Episode {ep}/{episodes} | Reward = {episode_reward:.2f}")

    # ==================================================
    # Auto-Save checkpoint every N episodes
    # ==================================================
    if ep % autosave_every == 0:
        agent.save_checkpoint(ep, checkpoint_path)


# ======================================================
# 6) Save final model (for evaluation purposes)
# ======================================================
final_model_path = r"D:\Project_end\New_world\my_project\models\Test_RCTankEnv.pt"
agent.save_model(path=final_model_path)

env.close()

print("\n[Trainer] Training finished.")
print(f"[Trainer] Final model saved to: {final_model_path}")
