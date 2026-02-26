# my_project/src/trainer/train_SAC_agent.py

from src.agent.SAC_Agent import SACAgent
import gymnasium as gym
import numpy as np
from pathlib import Path
import src.environment.register_envs
from src.utils.logger_pyarrow import EpisodeLogger
from src.environment.noise_manager import (
    NoiseManager,
    GaussianNoise,
    BoundedGaussianNoise,
    OUNoise,
    ScheduledNoise,
    NormalCurveScheduler,
)

# ======================================================
# Environment setup
# ======================================================
def env_setup(name_env="RCTankEnv-v0", render_mode="human"):
    scheduler = NormalCurveScheduler(
        peak=3000,   # จุดพีค noise
        std=1500,         # ความกว้างโค้ง
        max_scale=1.0
    )

    action_noise = ScheduledNoise(
        noise=OUNoise(
            mu=0.0,
            theta=0.15,
            sigma=0.25,   # sigma สูงสุด
            dt=0.1,
        ),
        scheduler=scheduler
    )

    process_noise = ScheduledNoise(
        GaussianNoise(sigma=0.02),
        scheduler
    )

    noise_manager = NoiseManager(
        action_noise=action_noise,
        process_noise=process_noise,
        sensor_noise=BoundedGaussianNoise(
            sigma=0.02,
            clip=0.05
        ),
        enabled=True,
    )


    env = gym.make(
        name_env,
        render_mode=render_mode,
        noise_manager=noise_manager,  
    )

    state, _ = env.reset()
    state_dim = state.shape[0]

    action_dim = env.action_space.shape[0]
    min_action = env.action_space.low
    max_action = env.action_space.high

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Action range:", min_action, max_action)

    return env, state_dim, action_dim, min_action, max_action


# ======================================================
# Training logic
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
    LOGGIN_STATUS_EP=True,
    FINAL_MODEL_PATH=None
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

        if hasattr(env, "noise_manager") and env.noise_manager is not None:
            env.noise_manager.on_episode_start(ep)

        current_setpoint = info.get("setpoint", None)
        episode_reward = 0.0

        for step in range(MAX_STEPS):

            action = agent.select_action(state)

            if hasattr(env, "noise_manager") and env.noise_manager is not None:
                env.noise_manager.step()

            next_state, reward, terminated, truncated, info = env.step(action)
            env.render()

            done = terminated or truncated
            current_setpoint = info.get("setpoint", current_setpoint)

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
                    done=done
                )

            # (4) SAC update
            agent.update(BATCH_SIZE)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(
            f"Episode {ep}/{EPISODES} | "
            f"Reward = {episode_reward:.2f}"
        )

        logger.save()
        logger.clear()
        agent.logger.save()
        agent.logger.clear()

        # ---------- Auto-save ----------
        if ep % AUTO_SAVE_EVERY == 0:
            agent.save_checkpoint(ep, CHECKPOINT_PATH)
    agent.save_model(FINAL_MODEL_PATH)

    env.close()
    print("\n[Trainer] Training finished.")
if __name__ == "__main__":

    # env config
    NAME_ENV = "RCTankEnv-v0"
    RENDER_MODE = "human"
    FOLDER_LOGGER = r"D:\Project_end\New_world\my_project\logs\episode"
    FILE_NAME_LOGGER = "episode_"

    # training config
    EPISODES = 10000
    MAX_STEPS = 200
    BATCH_SIZE = 1080
    LOGGIN_STATUS_EP = True

    # agent config
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
    FILE_NAME_AUTO_SAVE = "Autosave"

    # auto save
    CHECKPOINT_PATH = Path(
        r"D:\Project_end\New_world\my_project\models\checkpoint\Autosave.pt"
    )
    AUTO_SAVE_EVERY = 1
    FINAL_MODEL_PATH =  r"D:\Project_end\New_world\my_project\models\Test_histrory.pt"

    # logger
    logger = EpisodeLogger(
        folder=FOLDER_LOGGER,
        filename=FILE_NAME_LOGGER
    )

    # environment
    env, state_dim, action_dim, min_action, max_action = env_setup(
        name_env=NAME_ENV,
        render_mode=RENDER_MODE
    )

    # agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        min_action=min_action,
        max_action=max_action,
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

    # train
    train_Agent(
        env=env,
        agent=agent,
        logger=logger,
        EPISODES=EPISODES,
        MAX_STEPS=MAX_STEPS,
        BATCH_SIZE=BATCH_SIZE,
        CHECKPOINT_PATH=CHECKPOINT_PATH,
        AUTO_SAVE_EVERY=AUTO_SAVE_EVERY,
        LOGGIN_STATUS_EP=LOGGIN_STATUS_EP
        ,FINAL_MODEL_PATH=FINAL_MODEL_PATH
    )
