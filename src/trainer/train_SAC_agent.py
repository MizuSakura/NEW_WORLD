# my_project/src/trainer/train_SAC_agent.py
import os
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
def env_setup(name_env="RCTankEnv-v0", render_mode="human",
              noise_cfg=None):
    """
    สร้าง environment พร้อม NoiseManager
    noise_cfg — dict จาก rl_params.yaml section noise
                ถ้าไม่ส่งมา ใช้ค่า default
    """
    if noise_cfg is None:
        noise_cfg = {}

    ou    = noise_cfg.get("ou_noise",  {})
    gauss = noise_cfg.get("gaussian",  {})
    sens  = noise_cfg.get("sensor_noise", {})
    sched = noise_cfg.get("scheduler", {})

    scheduler = NormalCurveScheduler(
        peak      = sched.get("peak",      3000),
        std       = sched.get("std",       1500),
        max_scale = sched.get("max_scale", 1.0),
    )

    action_noise = ScheduledNoise(
        noise=OUNoise(
            mu    = ou.get("mu",    0.0),
            theta = ou.get("theta", 0.15),
            sigma = ou.get("sigma", 0.25),
            dt    = ou.get("dt",    0.1),
        ),
        scheduler=scheduler
    )

    process_noise = ScheduledNoise(
        GaussianNoise(sigma=gauss.get("sigma", 0.02)),
        scheduler
    )

    noise_manager = NoiseManager(
        action_noise  = action_noise,
        process_noise = process_noise,
        sensor_noise  = BoundedGaussianNoise(
            sigma = sens.get("sigma", 0.02),
            clip  = sens.get("clip",  0.05),
        ),
        enabled = noise_cfg.get("enabled", True),
    )

    render = render_mode
    if os.environ.get("DASHBOARD_MODE") == "1":
        render = None
    env = gym.make(
        name_env,
        render_mode=render,
        noise_manager=noise_manager,
    )

    state, _ = env.reset()
    state_dim  = state.shape[0]
    action_dim = env.action_space.shape[0]
    min_action = env.action_space.low
    max_action = env.action_space.high

    print("State dim:",   state_dim)
    print("Action dim:",  action_dim)
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
        episode_reward   = 0.0

        for step in range(MAX_STEPS):

            action = agent.select_action(state)

            if hasattr(env, "noise_manager") and env.noise_manager is not None:
                env.noise_manager.step()

            next_state, reward, terminated, truncated, info = env.step(action)
            env.render()

            done             = terminated or truncated
            current_setpoint = info.get("setpoint", current_setpoint)

            agent.replay_buffer.push(
                state, action, reward, next_state, float(done)
            )

            if LOGGIN_STATUS_EP:
                logger.log(
                    episode  = ep,
                    setpoint = current_setpoint,
                    step     = step,
                    state    = state,
                    action   = action,
                    reward   = reward,
                    next_state = next_state,
                    done     = done
                )

            agent.update(BATCH_SIZE)
            if step % 5 == 0:   
                print(
                    f"STEP|{ep}|{step}|"
                    f"{state[0]:.4f}|"
                    f"{float(action[0]):.4f}|"
                    f"{reward:.4f}|"
                    f"{current_setpoint:.4f}",
                    flush=True
                )

            state          = next_state
            episode_reward += reward

            if done:
                break

        print(
        f"Episode {ep}/{EPISODES} | "
        f"Reward = {episode_reward:.2f} | "
        f"Level = {state[0]:.3f} | "
        f"Setpoint = {current_setpoint:.3f}",
        flush=True
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

def _resolve_path(name: str, default_dir: str) -> Path:
    """
    ถ้าใส่แค่ชื่อไฟล์ (ไม่มี folder)   -> ใส่ใน default_dir
    ถ้าใส่ relative path (models/xxx)   -> ต่อจาก PROJECT_ROOT
    ถ้าใส่ absolute path                -> ใช้ตรงๆ
    เติม .pt ถ้าไม่มีนามสกุล
    """
    p = Path(name)

    # เติม .pt ถ้าไม่มีนามสกุล
    if p.suffix == "":
        p = p.with_suffix(".pt")

    # absolute path — ใช้ตรงๆ
    if p.is_absolute():
        return p

    # แค่ชื่อไฟล์ (ไม่มี folder) — ใส่ใน default_dir
    if len(p.parts) == 1:
        return PROJECT_ROOT / default_dir / p

    # relative path — ต่อจาก PROJECT_ROOT
    return PROJECT_ROOT / p


# ======================================================
# Main — อ่านค่าจาก rl_params.yaml
# ======================================================
if __name__ == "__main__":

    import sys
    import yaml

    # ── โหลด rl_params.yaml ──────────────────────────────────
    PROJECT_ROOT = Path(__file__).parent.parent.parent  # my_project/
    RL_CONFIG_PATH = PROJECT_ROOT / "src" / "API" / "config" / "rl_params.yaml"

    if RL_CONFIG_PATH.exists():
        print(f"[Config] Loading from {RL_CONFIG_PATH}")
        with open(RL_CONFIG_PATH, "r", encoding="utf-8") as f:
            rl_cfg = yaml.safe_load(f)
    else:
        print(f"[Config] rl_params.yaml not found — using defaults")
        rl_cfg = {}

    # ── แยก section ──────────────────────────────────────────
    train_cfg  = rl_cfg.get("training", {})
    sac_cfg    = rl_cfg.get("sac",      {})
    noise_cfg  = rl_cfg.get("noise",    {})
    logger_cfg = rl_cfg.get("logger",   {})

    actor_cfg  = sac_cfg.get("actor",  {})
    critic_cfg = sac_cfg.get("critic", {})

    # ── Training parameters ───────────────────────────────────
    NAME_ENV         = train_cfg.get("env_name",        "RCTankEnv-v0")
    RENDER_MODE      = train_cfg.get("render_mode",     "human")
    EPISODES         = train_cfg.get("episodes",        10000)
    MAX_STEPS        = train_cfg.get("max_steps",       200)
    BATCH_SIZE       = train_cfg.get("batch_size",      1080)
    AUTO_SAVE_EVERY  = train_cfg.get("auto_save_every", 1)
    LOGGIN_STATUS_EP = True

    

    CHECKPOINT_PATH  = _resolve_path(
        train_cfg.get("checkpoint_path",  "Autosave"),
        "models/checkpoint"
    )
    FINAL_MODEL_PATH = str(_resolve_path(
        train_cfg.get("final_model_path", "Test_history"),
        "models"
    ))
    # ── SAC parameters ────────────────────────────────────────
    LEARNING_RATE = sac_cfg.get("learning_rate", 3e-4)
    GAMMA         = sac_cfg.get("gamma",         0.99)
    TAU           = sac_cfg.get("tau",           0.005)
    ALPHA         = sac_cfg.get("alpha",         0.4)

    SIMPLE_LAYERS_ACTOR      = actor_cfg.get("layers",  2)
    SIMPLE_HIDDEN_ACTOR      = actor_cfg.get("hidden",  256)
    ADVANCED_HIDDEN_SIZE_ACTOR = None

    SIMPLE_LAYERS_CRITIC       = critic_cfg.get("layers",  2)
    SIMPLE_HIDDEN_CRITIC       = critic_cfg.get("hidden",  256)
    ADVANCED_HIDDEN_SIZE_CRITIC = None
    CRITIC_ENCODE              = critic_cfg.get("encoder", False)

    # ── Logger parameters ─────────────────────────────────────
    FOLDER_LOGGER          = logger_cfg.get("episode_folder",
                                str(PROJECT_ROOT / "logs" / "episode"))
    FILE_NAME_LOGGER       = logger_cfg.get("episode_filename", "episode_")
    LOGGER_PATH_AGENT      = logger_cfg.get("agent_folder",
                                str(PROJECT_ROOT / "logs" / "agent" / "RC_Tank"))
    LOGGER_FILE_NAME_AGENT = logger_cfg.get("agent_filename", "optimized_")

    # ── Print config summary ──────────────────────────────────
    print("\n" + "="*50)
    print("[Config] Training Parameters")
    print("="*50)
    print(f"  Env           : {NAME_ENV}")
    print(f"  Episodes      : {EPISODES}")
    print(f"  Max Steps     : {MAX_STEPS}")
    print(f"  Batch Size    : {BATCH_SIZE}")
    print(f"  Learning Rate : {LEARNING_RATE}")
    print(f"  Gamma         : {GAMMA}")
    print(f"  Alpha         : {ALPHA}")
    print(f"  Actor layers  : {SIMPLE_LAYERS_ACTOR} x {SIMPLE_HIDDEN_ACTOR}")
    print(f"  Critic layers : {SIMPLE_LAYERS_CRITIC} x {SIMPLE_HIDDEN_CRITIC}")
    print(f"  Noise enabled : {noise_cfg.get('enabled', True)}")
    print(f"  Checkpoint    : {CHECKPOINT_PATH}")
    print("="*50 + "\n")

    # ── Logger ────────────────────────────────────────────────
    logger = EpisodeLogger(
        folder   = FOLDER_LOGGER,
        filename = FILE_NAME_LOGGER
    )

    # ── Environment ───────────────────────────────────────────
    env, state_dim, action_dim, min_action, max_action = env_setup(
        name_env    = NAME_ENV,
        render_mode = RENDER_MODE,
        noise_cfg   = noise_cfg,
    )

    # ── Agent ─────────────────────────────────────────────────
    agent = SACAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        min_action   = min_action,
        max_action   = max_action,
        lr           = LEARNING_RATE,
        gamma        = GAMMA,
        tau          = TAU,
        alpha        = ALPHA,
        logger_status = True,
        simple_layers_actor        = SIMPLE_LAYERS_ACTOR,
        simple_hidden_actor        = SIMPLE_HIDDEN_ACTOR,
        advanced_hidden_size_actor = ADVANCED_HIDDEN_SIZE_ACTOR,
        simple_layers_critic         = SIMPLE_LAYERS_CRITIC,
        simple_hidden_critic         = SIMPLE_HIDDEN_CRITIC,
        advanced_hidden_sizes_critic = ADVANCED_HIDDEN_SIZE_CRITIC,
        critic_encoder = CRITIC_ENCODE,
        logger_path    = LOGGER_PATH_AGENT,
        file_name_log  = LOGGER_FILE_NAME_AGENT,
    )

    # ── Train ─────────────────────────────────────────────────
    train_Agent(
        env             = env,
        agent           = agent,
        logger          = logger,
        EPISODES        = EPISODES,
        MAX_STEPS       = MAX_STEPS,
        BATCH_SIZE      = BATCH_SIZE,
        CHECKPOINT_PATH = CHECKPOINT_PATH,
        AUTO_SAVE_EVERY = AUTO_SAVE_EVERY,
        LOGGIN_STATUS_EP = LOGGIN_STATUS_EP,
        FINAL_MODEL_PATH = FINAL_MODEL_PATH,
    )