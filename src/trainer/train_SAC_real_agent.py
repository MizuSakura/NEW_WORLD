# my_project/src/trainer/train_SAC_real_agent.py

from src.agent.SAC_Agent import SACAgent
import numpy as np
from pathlib import Path
import time

from src.utils.logger_pyarrow import EpisodeLogger
from src.environment.Apply_real_env import Real_env_remote

# ======================================================
# Realtime status (utility)
# ======================================================
def print_realtime_status(ep, step, state, action, reward, setpoint):
    state_str = np.array2string(np.array(state), precision=3)
    action_val = float(action[0]) if hasattr(action, "__len__") else float(action)

    msg = (
        f"\r[EP {ep:04d} | STEP {step:04d}] "
        f"SP: {setpoint:.2f} | "
        f"STATE: {state_str} | "
        f"ACTION: {action_val:6.3f} | "
        f"REWARD: {reward:7.3f}"
    )
    print(msg, end="", flush=True)

def env_setup_real(
    ip_host,
    port,
    min_action,
    max_action,
    setpoint,
    delay_of_action,
    address_sensor,
    address_actuator,
):
    env = Real_env_remote(
        ip_host=ip_host,
        port=port,
        min_action=min_action,
        max_action=max_action,
        setpoint=setpoint,
        delay_of_action=delay_of_action,
        address_sensor=address_sensor,
        address_actuator=address_actuator,
    )

    state,_ = env.reset()

    print("State dim:", env.state_dim)
    print("Action dim:", env.action_dim)
    print("Action range:", env.min_action, env.max_action)

    return env, env.state_dim, env.action_dim, min_action, max_action

# ======================================================
# Training logic (REAL)
# ======================================================
def train_Agent(
    env,
    agent,
    logger,
    EPISODES,
    MAX_STEPS,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    AUTO_SAVE_EVERY,
    DELAY_TIME_RESET=1.0,
    LOGGIN_STATUS_EP=True,
    FINAL_MODEL_PATH=None,
):
    # ---------- Resume ----------
    start_episode = 1
    if CHECKPOINT_PATH.exists():
        print("\n[Trainer] Found checkpoint. Loading...")
        start_episode = agent.load_checkpoint(CHECKPOINT_PATH) + 1
        print(f"[Trainer] Resuming training from episode {start_episode}\n")
    else:
        print("\n[Trainer] No checkpoint found. Starting from episode 1\n")

    # ---------- Training loop ----------
    for ep in range(start_episode, EPISODES + 1):

        state, info = env.reset()
        current_setpoint = info.get("setpoint", None)
        episode_reward = 0.0

        time.sleep(DELAY_TIME_RESET)

        for step in range(MAX_STEPS):

            # (1) Select action
            action = agent.select_action(state)

            # (2) Environment step (REAL)
            next_state, reward, done, info = env.step(action=action)

            print_realtime_status(
                ep=ep,
                step=step,
                state=state,
                action=action,
                reward=reward,
                setpoint=current_setpoint,
            )

            # (3) Store transition
            agent.replay_buffer.push(
                state, action, reward, next_state, float(done)
            )

            if LOGGIN_STATUS_EP:
                logger.log(
                    episode=ep,
                    setpoint=current_setpoint,
                    step=step,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

            # (4) SAC update
            agent.update(BATCH_SIZE)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(
            f"\nEpisode {ep}/{EPISODES} | "
            f"Reward = {episode_reward:.2f}"
        )

        logger.save()
        logger.clear()
        agent.logger.save()
        agent.logger.clear()

        # ---------- Auto-save ----------
        if ep % AUTO_SAVE_EVERY == 0:
            agent.save_checkpoint(ep, CHECKPOINT_PATH)

    # ---------- Save final model ----------
    if FINAL_MODEL_PATH is not None:
        agent.save_model(FINAL_MODEL_PATH)

    print("\n[Trainer] Training finished.")

if __name__ == "__main__":

    # ==============================
    # REAL ENV CONFIG
    # ==============================
    IP_HOST = "192.168.1.100"
    PORT = 502
    MIN_ACTION = 0
    MAX_ACTION = 10
    SETPOINT = 5
    DELAY_OF_ACTION = 0.2
    ADDRESS_SENSOR = 1
    ADDRESS_ACTUATOR = 1025

    # CONFIG AGENT 
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.4
    LOGGER_STATUS = True

    SIMPLE_LAYERS_ACTOR = 2
    SIMPLE_HIDDEN_ACTOR = 256
    ADVANCED_HIDDEN_SIZE_ACTOR = None

    SIMPLE_LAYERS_CRITIC = 2
    SIMPLE_HIDDEN_CRITIC = 256
    ADVANCED_HIDDEN_SIZE_CRITIC = None
    CRITIC_ENCODE = False
    LOGGER_PATH_AGENT = r"D:\Project_end\New_world\my_project\logs\agent\RC_Tank"
    LOGGER_FILE_NAME_AGENT = "optimized_"

    # ==============================
    # TRAINING CONFIG
    # ==============================

    EPISODES = 1000
    MAX_STEPS = 100
    BATCH_SIZE = 1080
    AUTO_SAVE_EVERY = 5
    DELAY_TIME_RESET = 1.0

    CHECKPOINT_PATH = Path(
        r"D:\Project_end\New_world\my_project\models\checkpoint\sac_checkpoint_real.pt"
    )
    FINAL_MODEL_PATH = (
        r"D:\Project_end\New_world\my_project\models\Test_train_real.pt"
    )

     # ==============================
    # LOGGER
    # ==============================
    logger = EpisodeLogger(
        folder=r"D:\Project_end\New_world\my_project\logs\episode_train_real",
        filename="episode_",
    )

    # ==============================
    # ENVIRONMENT
    # ==============================
    env, state_dim, action_dim, min_action, max_action  = env_setup_real(ip_host=IP_HOST, port=PORT, min_action=MIN_ACTION,
        max_action=MAX_ACTION, setpoint=SETPOINT, delay_of_action=DELAY_OF_ACTION, address_sensor=ADDRESS_SENSOR,
        address_actuator=ADDRESS_ACTUATOR,
    )

    # ==============================
    # AGENT
    # ==============================
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        min_action=np.array([min_action]),
        max_action=np.array([max_action]),
        lr=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        logger_status=LOGGER_STATUS,
        simple_layers_actor=SIMPLE_LAYERS_ACTOR,
        simple_hidden_actor=SIMPLE_HIDDEN_ACTOR,
        advanced_hidden_size_actor=ADVANCED_HIDDEN_SIZE_ACTOR,
        simple_layers_critic=SIMPLE_LAYERS_CRITIC,
        simple_hidden_critic=SIMPLE_HIDDEN_CRITIC,
        advanced_hidden_sizes_critic=ADVANCED_HIDDEN_SIZE_CRITIC,
        critic_encoder=CRITIC_ENCODE,
        logger_path=LOGGER_PATH_AGENT,
        file_name_log=LOGGER_FILE_NAME_AGENT
    )

    # ==============================
    # TRAIN
    # ==============================
    train_Agent(
        env=env,
        agent=agent,
        logger=logger,
        EPISODES=EPISODES,
        MAX_STEPS=MAX_STEPS,
        BATCH_SIZE=BATCH_SIZE,
        CHECKPOINT_PATH=CHECKPOINT_PATH,
        AUTO_SAVE_EVERY=AUTO_SAVE_EVERY,
        DELAY_TIME_RESET=DELAY_TIME_RESET,
        LOGGIN_STATUS_EP=True,
        FINAL_MODEL_PATH=FINAL_MODEL_PATH,
    )