#my_project\src\trainer\train_SAC_real_agent.py
from src.agent.SAC_Agent import SACAgent
import gymnasium as gym
import numpy as np
from pathlib import Path
from  src.utils.logger_pyarrow import EpisodeLogger
from src.environment.Apply_real_env import Real_env_remote
import time

def print_realtime_status(ep, step, state, action, reward, setpoint):
    state_str  = np.array2string(np.array(state), precision=3)
    action_val = float(action[0]) if hasattr(action, "__len__") else float(action)

    msg = (
        f"\r[EP {ep:04d} | STEP {step:04d}] "
        f"SP: {setpoint:.2f} | "
        f"STATE: {state_str} | "
        f"ACTION: {action_val:6.3f} | "
        f"REWARD: {reward:7.3f}"
    )
    print(msg, end="", flush=True)

Logger = EpisodeLogger(folder=r"D:\Project_end\New_world\my_project\logs\episode_train_real",filename="episode_")
env = Real_env_remote(ip_host="192.168.1.100",
                    port=502,
                    min_action = 0,
                    max_action = 24,
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
    logger_status=True,
    simple_layers_actor=2,
    simple_hidden_actor=256,
    advanced_hidden_size_actor=None,
    simple_layers_critic= 2,
    simple_hidden_critic= 256,
    advanced_hidden_sizes_critic=None,
    critic_encoder=False,
    logger_path=r"D:\Project_end\New_world\my_project\logs\agent_train_real",
    file_name_log = "optimized_"
)

episodes = 1000
max_steps = 2000
batch_size = 1080
Logging_status = True
rewards_history = []

checkpoint_path = Path(r"D:\Project_end\New_world\my_project\models\sac_checkpoint_real.pt")
autosave_every = 10  # Save checkpoint every N episodes
delay_time_reset = 1

start_episode = 1

if checkpoint_path.exists():
    print("\n[Trainer] Found checkpoint. Loading...")
    start_episode = agent.load_checkpoint(checkpoint_path) + 1
    print(f"[Trainer] Resuming training from episode {start_episode}\n")
else:
    print("\n[Trainer] No checkpoint found. Starting from episode 1\n")

for ep in range(start_episode, episodes + 1):
    state ,info= env.reset()
    current_setpoint = info.get("setpoint", None)
    episode_reward = 0
    time.sleep(delay_time_reset)

    for step in range(max_steps):

        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action = action)
        print_realtime_status(
        ep=ep,
        step=step,
        state=state,
        action=action,
        reward=reward,
        setpoint=current_setpoint
    )

        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        if Logging_status:
            Logger.log(episode=ep,setpoint=current_setpoint,step=step,state=state,action=action,reward=reward,next_state=next_state,done=done)

        agent.update(batch_size)

        state = next_state
        episode_reward += reward

        if done:
            break

    rewards_history.append(episode_reward)
    print(f"Episode {ep}/{episodes} | Reward = {episode_reward:.2f} | status train:{done} ")
    Logger.save()
    Logger.clear()
    agent.logger.save()
    agent.logger.clear()

    # ==================================================
    # Auto-Save checkpoint every N episodes
    # ==================================================
    if ep % autosave_every == 0:
        agent.save_checkpoint(ep, checkpoint_path)


# ======================================================
# 6) Save final model (for evaluation purposes)
# ======================================================
final_model_path = r"D:\Project_end\New_world\my_project\models\Test_train_real.pt"
agent.save_model(path=final_model_path)
