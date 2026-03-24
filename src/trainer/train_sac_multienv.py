# my_project/src/trainer/train_SAC_multienv.py
"""
Multi-Environment SAC Trainer  (AsyncVectorEnv + Domain Randomization)
-----------------------------------------------------------------------
- รัน N env parallel บน CPU ด้วย gymnasium.vector.AsyncVectorEnv
- ทุก env share replay buffer เดียวกัน (PER + N-step ของเดิม)
- Domain Randomization: สุ่ม R, C ต่างกันทุก worker → robust sim-to-real
- stdout protocol เหมือน train_SAC_agent.py ทุกบรรทัด
  → server._stream_log / dashboard ไม่ต้องแก้เลย
- num_envs อ่านจาก rl_params.yaml  training.num_envs  (default 8)

Usage (รันตรงๆ):
    python -m src.trainer.train_SAC_multienv
"""

import yaml
import numpy as np
from pathlib import Path

import gymnasium as gym
import src.environment.register_envs      # noqa: F401  register RCTankEnv-v0

from src.agent.SAC_Agent import SACAgent
from src.utils.logger_pyarrow import EpisodeLogger
from src.environment.noise_manager import (
    NoiseManager,
    GaussianNoise,
    BoundedGaussianNoise,
    OUNoise,
    ScheduledNoise,
    NormalCurveScheduler,
)


# ======================================================================
# Helpers
# ======================================================================

def _resolve_path(name: str, default_dir: str, project_root: Path) -> Path:
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(".pt")
    if p.is_absolute():
        return p
    if len(p.parts) == 1:
        return project_root / default_dir / p
    return project_root / p


def _build_noise(noise_cfg: dict, seed: int) -> NoiseManager:
    """สร้าง NoiseManager 1 ชุดสำหรับ 1 worker"""
    ou    = noise_cfg.get("ou_noise",     {})
    gauss = noise_cfg.get("gaussian",     {})
    sens  = noise_cfg.get("sensor_noise", {})
    sched = noise_cfg.get("scheduler",    {})

    scheduler = NormalCurveScheduler(
        peak      = sched.get("peak",      3000),
        std       = sched.get("std",       1500),
        max_scale = sched.get("max_scale", 1.0),
    )
    action_noise = ScheduledNoise(
        noise=OUNoise(
            mu    = ou.get("mu",    0.0),
            theta = ou.get("theta", 0.15),
            sigma = ou.get("sigma", 0.10),
            dt    = ou.get("dt",    0.1),
        ),
        scheduler=scheduler,
    )
    process_noise = ScheduledNoise(
        GaussianNoise(sigma=gauss.get("sigma", 0.01)),
        scheduler,
    )
    return NoiseManager(
        action_noise  = action_noise,
        process_noise = process_noise,
        sensor_noise  = BoundedGaussianNoise(
            sigma = sens.get("sigma", 0.02),
            clip  = sens.get("clip",  0.05),
        ),
        enabled = noise_cfg.get("enabled", True),
    )


def _make_env_fn(env_name: str, noise_cfg: dict, rank: int):
    """
    Factory function สำหรับ AsyncVectorEnv — 1 fn ต่อ 1 worker

    Domain Randomization ต่อ worker:
        R ∈ [1.0, 2.5]  Ω   (nominal 1.5)
        C ∈ [1.5, 3.0]  F   (nominal 2.0)
    seed แตกต่างกันทุก worker → setpoint / init state สุ่มอิสระ
    → agent เจอ distribution ของ dynamics กว้างขึ้น
    → robust กับ hardware จริงที่มี parameter drift
    """
    def _init():
        rng = np.random.default_rng(rank * 1000 + 42)
        R   = float(rng.uniform(1.0, 2.5))
        C   = float(rng.uniform(1.5, 3.0))

        env = gym.make(
            env_name,
            render_mode        = "human",   # ปิด GUI เสมอในโหมด multi-env
            noise_manager      = _build_noise(noise_cfg, seed=rank),
            R                  = R,
            C                  = C,
            save_episode_image = False,
        )
        env.reset(seed=rank)
        return env
    return _init


def env_setup_single(env_name: str, noise_cfg: dict):
    """
    สร้าง env 1 ตัวชั่วคราวเพื่อดึง state_dim / action_dim / bounds
    แล้วปิดทิ้ง — VectorEnv จะสร้าง worker ใหม่
    """
    env = _make_env_fn(env_name, noise_cfg, rank=0)()
    obs, _ = env.reset()
    state_dim  = obs.shape[0]
    action_dim = env.action_space.shape[0]
    min_action = env.action_space.low.copy()
    max_action = env.action_space.high.copy()
    env.close()
    return state_dim, action_dim, min_action, max_action


# ======================================================================
# Training loop
# ======================================================================

def train_multienv(
    env_name:         str,
    noise_cfg:        dict,
    agent:            SACAgent,
    logger:           EpisodeLogger,
    num_envs:         int,
    episodes:         int,
    max_steps:        int,
    batch_size:       int,
    checkpoint_path:  Path,
    auto_save_every:  int,
    final_model_path: str,
    log_every_steps:  int = 5,
):
    # ── Resume ────────────────────────────────────────────────────
    start_episode = 1
    if checkpoint_path.exists():
        print("[Trainer] Found checkpoint. Loading...", flush=True)
        start_episode = agent.load_checkpoint(checkpoint_path) + 1
        print(f"[Trainer] Resuming training from episode {start_episode}\n",
              flush=True)
    else:
        print("[Trainer] No checkpoint found. Starting from episode 1\n",
              flush=True)

    print(
        f"[MultiEnv] Starting | num_envs={num_envs} | "
        f"episodes={episodes} | max_steps={max_steps} | batch={batch_size}",
        flush=True,
    )

    # ── Build AsyncVectorEnv ──────────────────────────────────────
    # AsyncVectorEnv รัน env แต่ละ worker ใน subprocess แยกกัน
    # → ไม่มี GIL blocking → CPU cores ทำงาน parallel จริง
    vec_env = gym.vector.AsyncVectorEnv(
        [_make_env_fn(env_name, noise_cfg, rank=i) for i in range(num_envs)]
    )

    obs_vec, infos = vec_env.reset()    # shape: (num_envs, state_dim)

    print(f"[MultiEnv] State dim : {obs_vec.shape[1]}", flush=True)
    print(f"[MultiEnv] Action dim: {vec_env.single_action_space.shape[0]}\n",
          flush=True)

    # ── Per-env accounting ────────────────────────────────────────
    env_ep_rewards = np.zeros(num_envs, dtype=np.float64)
    env_steps      = np.zeros(num_envs, dtype=np.int32)
    env_setpoints  = np.zeros(num_envs, dtype=np.float64)

    global_ep  = start_episode
    total_step = 0

    # ── Main loop ─────────────────────────────────────────────────
    try:
        while global_ep <= episodes:

            # 1) Forward pass: select actions ทุก env (1 GPU batch)
            actions = agent.select_action_batch(obs_vec)    # (num_envs, 1)

            # 2) Step ทุก env พร้อมกัน (async subprocess)
            next_obs, rewards, terms, truncs, infos = vec_env.step(actions)
            total_step += num_envs

            # 3) Parse setpoints per env
            sp_raw = infos.get("setpoint", None)
            if isinstance(sp_raw, np.ndarray):
                env_setpoints = sp_raw.astype(np.float64)
            elif sp_raw is not None:
                env_setpoints[:] = float(sp_raw)

            # 4) Push ทุก transition → shared replay buffer
            #    buffer เดียวกับ single-env trainer — ไม่ต้องแก้อะไร
            for i in range(num_envs):
                done_i = float(terms[i] or truncs[i])
                agent.replay_buffer.push(
                    obs_vec[i],
                    actions[i],
                    float(rewards[i]),
                    next_obs[i],
                    done_i,
                )
                env_ep_rewards[i] += rewards[i]
                env_steps[i]      += 1

            # 5) Learner: 1 gradient step per global step
            #    (num_envs transitions ต่อ 1 update → buffer เต็มเร็วขึ้น)
            agent.update(batch_size)

            # 6) STEP log → server parses "STEP|ep|step|level|action|reward|sp"
            step_rep = int(env_steps[0])
            if step_rep % log_every_steps == 0:
                print(
                    f"STEP|{global_ep}|{step_rep}|"
                    f"{float(next_obs[0][0]):.4f}|"
                    f"{float(actions[0][0]):.4f}|"
                    f"{float(rewards[0]):.4f}|"
                    f"{float(env_setpoints[0]):.4f}",
                    flush=True,
                )

            # 7) Episode done per env
            for i in range(num_envs):
                if not (terms[i] or truncs[i]):
                    continue

                ep_reward = float(env_ep_rewards[i])

                # server parses regex: "Episode X/Y | Reward = Z"
                print(
                    f"Episode {global_ep}/{episodes} | "
                    f"Reward = {ep_reward:.2f} | "
                    f"Worker = {i} | "
                    f"Steps = {int(env_steps[i])} | "
                    f"Level = {float(next_obs[i][0]):.3f} | "
                    f"Setpoint = {float(env_setpoints[i]):.3f}",
                    flush=True,
                )

                logger.save()
                logger.clear()
                agent.logger.save()
                agent.logger.clear()

                if global_ep % auto_save_every == 0:
                    agent.save_checkpoint(global_ep, checkpoint_path)

                env_ep_rewards[i] = 0.0
                env_steps[i]      = 0
                global_ep        += 1

                if global_ep > episodes:
                    break

            obs_vec = next_obs

    finally:
        vec_env.close()
        agent.save_model(final_model_path)
        print("\n[Trainer] Training finished.", flush=True)


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":

    PROJECT_ROOT   = Path(__file__).resolve().parents[2]
    RL_CONFIG_PATH = PROJECT_ROOT / "src" / "API" / "config" / "rl_params.yaml"

    if RL_CONFIG_PATH.exists():
        print(f"[Config] Loading from {RL_CONFIG_PATH}", flush=True)
        with open(RL_CONFIG_PATH, "r", encoding="utf-8") as f:
            rl_cfg = yaml.safe_load(f)
    else:
        print("[Config] rl_params.yaml not found — using defaults", flush=True)
        rl_cfg = {}

    train_cfg  = rl_cfg.get("training", {})
    sac_cfg    = rl_cfg.get("sac",      {})
    noise_cfg  = rl_cfg.get("noise",    {})
    logger_cfg = rl_cfg.get("logger",   {})
    actor_cfg  = sac_cfg.get("actor",   {})
    critic_cfg = sac_cfg.get("critic",  {})

    NAME_ENV        = train_cfg.get("env_name",        "RCTankEnv-v0")
    EPISODES        = train_cfg.get("episodes",        10000)
    MAX_STEPS       = train_cfg.get("max_steps",       200)
    BATCH_SIZE      = train_cfg.get("batch_size",      1080)
    AUTO_SAVE_EVERY = train_cfg.get("auto_save_every", 1)
    NUM_ENVS        = train_cfg.get("num_envs",        8)

    CHECKPOINT_PATH  = _resolve_path(
        train_cfg.get("checkpoint_path",  "Autosave"),
        "models/checkpoint", PROJECT_ROOT,
    )
    FINAL_MODEL_PATH = str(_resolve_path(
        train_cfg.get("final_model_path", "Test_history"),
        "models", PROJECT_ROOT,
    ))

    LEARNING_RATE   = sac_cfg.get("learning_rate",  3e-4)
    GAMMA           = sac_cfg.get("gamma",           0.99)
    TAU             = sac_cfg.get("tau",             0.005)
    ALPHA           = sac_cfg.get("alpha",           0.05)
    REPLAY_CAPACITY = sac_cfg.get("replay_capacity", 200000)
    BUFFER_TYPE     = sac_cfg.get("buffer_type",     "nstep_per")
    N_STEP          = sac_cfg.get("n_step",          3)
    PER_ALPHA       = sac_cfg.get("per_alpha",       0.6)
    PER_BETA        = sac_cfg.get("per_beta",        0.4)

    SIMPLE_LAYERS_ACTOR  = actor_cfg.get("layers",  2)
    SIMPLE_HIDDEN_ACTOR  = actor_cfg.get("hidden",  256)
    SIMPLE_LAYERS_CRITIC = critic_cfg.get("layers", 2)
    SIMPLE_HIDDEN_CRITIC = critic_cfg.get("hidden", 256)
    CRITIC_ENCODER       = critic_cfg.get("encoder", False)

    FOLDER_LOGGER          = logger_cfg.get("episode_folder",
                                str(PROJECT_ROOT / "logs" / "episode"))
    FILE_NAME_LOGGER       = logger_cfg.get("episode_filename", "episode_")
    LOGGER_PATH_AGENT      = logger_cfg.get("agent_folder",
                                str(PROJECT_ROOT / "logs" / "agent" / "RC_Tank"))
    LOGGER_FILE_NAME_AGENT = logger_cfg.get("agent_filename", "optimized_")

    print("\n" + "="*55, flush=True)
    print("[Config] Multi-Env SAC Training", flush=True)
    print("="*55, flush=True)
    print(f"  Env           : {NAME_ENV}",            flush=True)
    print(f"  Num envs      : {NUM_ENVS}  (parallel)", flush=True)
    print(f"  Episodes      : {EPISODES}",             flush=True)
    print(f"  Max steps     : {MAX_STEPS}",            flush=True)
    print(f"  Batch size    : {BATCH_SIZE}",           flush=True)
    print(f"  Alpha (SAC)   : {ALPHA}",                flush=True)
    print(f"  Buffer type   : {BUFFER_TYPE}",          flush=True)
    print(f"  Checkpoint    : {CHECKPOINT_PATH}",      flush=True)
    print("="*55 + "\n",                               flush=True)

    logger = EpisodeLogger(folder=FOLDER_LOGGER, filename=FILE_NAME_LOGGER)

    state_dim, action_dim, min_action, max_action = env_setup_single(
        NAME_ENV, noise_cfg
    )
    print(f"[Setup] state_dim={state_dim}  action_dim={action_dim}", flush=True)
    print(f"[Setup] action range=[{min_action[0]:.1f}, {max_action[0]:.1f}]\n",
          flush=True)

    agent = SACAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        min_action   = min_action,
        max_action   = max_action,
        lr           = LEARNING_RATE,
        gamma        = GAMMA,
        tau          = TAU,
        alpha        = ALPHA,
        replay_capacity = REPLAY_CAPACITY,
        buffer_type     = BUFFER_TYPE,
        n_step          = N_STEP,
        per_alpha       = PER_ALPHA,
        per_beta        = PER_BETA,
        logger_status   = True,
        simple_layers_actor        = SIMPLE_LAYERS_ACTOR,
        simple_hidden_actor        = SIMPLE_HIDDEN_ACTOR,
        advanced_hidden_size_actor = None,
        simple_layers_critic         = SIMPLE_LAYERS_CRITIC,
        simple_hidden_critic         = SIMPLE_HIDDEN_CRITIC,
        advanced_hidden_sizes_critic = None,
        critic_encoder = CRITIC_ENCODER,
        logger_path    = LOGGER_PATH_AGENT,
        file_name_log  = LOGGER_FILE_NAME_AGENT,
    )

    train_multienv(
        env_name         = NAME_ENV,
        noise_cfg        = noise_cfg,
        agent            = agent,
        logger           = logger,
        num_envs         = NUM_ENVS,
        episodes         = EPISODES,
        max_steps        = MAX_STEPS,
        batch_size       = BATCH_SIZE,
        checkpoint_path  = CHECKPOINT_PATH,
        auto_save_every  = AUTO_SAVE_EVERY,
        final_model_path = FINAL_MODEL_PATH,
    )