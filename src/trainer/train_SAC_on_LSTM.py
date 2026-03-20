# my_project/src/trainer/train_SAC_on_LSTM.py
"""
SAC on LSTM Environment
-----------------------
ใช้ LSTM ที่ train ไว้แล้วเป็น environment model
SAC train บน LSTM แทน hardware จริง

Flow:
    1. โหลด LSTM model (.pth)
    2. โหลด scaler zip
    3. LSTMEnv — gym-like env ที่ใช้ LSTM ทำนาย next level
    4. Train SAC เหมือน offline ปกติ
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import torch
import yaml
from pathlib import Path
from collections import deque

from src.agent.SAC_Agent import SACAgent
from src.environment.reward_function_control import Reward_manager
from src.data.scaling_loader import ScalingZipLoader
from src.utils.logger_pyarrow import EpisodeLogger
from src.environment.noise_manager import (
    NoiseManager, GaussianNoise, BoundedGaussianNoise,
    OUNoise, ScheduledNoise, NormalCurveScheduler,
)
from src.environment.state_builder import StateBuilder

# ======================================================
# LSTM Environment
# ======================================================
class LSTMEnv:
    """
    Gym-like environment ที่ใช้ LSTM ทำนาย next level
    State = [level] + action_history(10) + [setpoint] = 12 dims
    Action = [pump_output] ∈ [min_action, max_action]
    """

    def __init__(
        self,
        lstm_model,
        scaler,
        sequence_size: int = 30,
        min_action: float = 0.0,
        max_action: float = 10.0,
        action_history_len: int = 10,
        max_steps: int = 200,
        device=None,
        noise_manager=None,
    ):
        self.model          = lstm_model
        self.scaler         = scaler
        self.sequence_size  = sequence_size
        self.min_action     = min_action
        self.max_action     = max_action
        self.action_history_len = action_history_len
        self.max_steps      = max_steps
        self.device         = device or torch.device("cpu")
        self.noise_manager  = noise_manager

        # StateBuilder — canonical state เหมือน RCTankEnv_gym
        _rl_cfg_path = Path(__file__).resolve().parents[2] / "src" / "API" / "config" / "rl_params.yaml"
        if _rl_cfg_path.exists():
            self.state_builder = StateBuilder.from_yaml(_rl_cfg_path)
        else:
            self.state_builder = StateBuilder({})

        self.state_dim  = self.state_builder.state_dim
        self.action_dim = 1

        # Sequence buffer สำหรับ LSTM input
        self._input_buffer = deque(maxlen=sequence_size)
        self._action_history = deque(maxlen=action_history_len)
        self._reward_manager = Reward_manager(buffer_size=5)
        self._step_count = 0
        self._level = 0.0
        self._setpoint = 5.0


    def _predict_next_level(self, action: float) -> float:
        """ส่ง [action, level] เข้า LSTM แล้วได้ next level"""
        raw_input = np.array([[action, self._level]])

        # Scale input
        scaled = self.scaler.scaler_in.transform(raw_input)  # (1, 2) หรือ (1, 1)
        self._input_buffer.append(scaled[0])

        # ถ้า buffer ยังไม่เต็ม — pad ด้วย zero
        seq = list(self._input_buffer)
        while len(seq) < self.sequence_size:
            seq.insert(0, np.zeros_like(seq[0]))

        # สร้าง tensor (1, seq_len, features)
        x = torch.tensor(
            np.array(seq), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_scaled = self.model(x).cpu().numpy()  # (1, 1)

        # Inverse scale
        next_level = self.scaler.scaler_out.inverse_transform(y_scaled)[0][0]
        return float(np.clip(next_level, self.min_action, self.max_action))

    def reset(self) -> tuple[np.ndarray, dict]:
        # Random setpoint
        self._setpoint = float(np.random.uniform(
            self.min_action, self.max_action
        ))

        # Initial level — random เล็กน้อย
        self._level = float(np.random.uniform(
            self.min_action, self.max_action * 0.3
        ))

        # Clear buffers
        # Clear buffers
        self._input_buffer.clear()

        self._reward_manager.reset(
            init_setpoint = self._setpoint,
            init_state    = self._level,
            init_action   = self._level,
        )
        self._step_count = 0

        # reset StateBuilder
        self.state_builder.reset(
            level    = self._level,
            action   = self._level,
            setpoint = self._setpoint,
            dt       = 0.1,
        )
        state = self.state_builder.get_state()

        return state, {"setpoint": self._setpoint}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = float(np.clip(np.array(action).item(), self.min_action, self.max_action))

        # Noise บน action
        if self.noise_manager:
            noisy_action = self.noise_manager.apply_action_noise(action)
            noisy_action = float(np.clip(noisy_action, self.min_action, self.max_action))
        else:
            noisy_action = action

        # LSTM ทำนาย next level
        next_level = self._predict_next_level(noisy_action)

        # Process noise บน level
        if self.noise_manager:
            next_level = self.noise_manager.apply_process_noise(next_level)
            next_level = float(np.clip(next_level, self.min_action, self.max_action))

        self._level = next_level
        self._step_count += 1

        # Reward
        self._reward_manager.update(
            setpoint = self._setpoint,
            state    = self._level,
            action   = action,
        )
        reward = self._reward_manager.reward_continuous_control()

        # Done
        error       = abs(self._setpoint - self._level)
        truncated   = self._step_count >= self.max_steps
        terminated  = error < 0.05

        info = {
            "setpoint": self._setpoint,
            "level":    self._level,
            "error":    error,
        }

        state = self.state_builder.update(
        level    = self._level,
        action   = action,
        setpoint = self._setpoint,)
        return state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


# ======================================================
# Load LSTM model
# ======================================================
def load_lstm_model(model_path: Path, device):
    """โหลด LSTM model จาก .pth ไฟล์"""
    from src.models.lstm_model import VanillaLSTM_MODEL, DeepLSTM_MODEL, BiLSTM_MODEL

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type  = checkpoint.get("model_type",  "DeepLSTM")
    input_dim   = checkpoint.get("input_dim",   2)
    output_dim  = checkpoint.get("output_dim",  1)
    hidden_dim  = checkpoint.get("hidden_dim",  128)
    num_layers  = checkpoint.get("num_layers",  2)
    fc_units    = checkpoint.get("fc_units",    [64, 32])

    MODEL_MAP = {
        "VanillaLSTM": VanillaLSTM_MODEL,
        "DeepLSTM":    DeepLSTM_MODEL,
        "BiLSTM":      BiLSTM_MODEL,
    }

    cls   = MODEL_MAP.get(model_type, DeepLSTM_MODEL)
    model = cls(
        input_dim  = input_dim,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        output_dim = output_dim,
        fc_units   = fc_units,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[LSTM] Loaded {model_type} | input={input_dim} output={output_dim} hidden={hidden_dim}")
    return model


# ======================================================
# Training logic (same as train_SAC_agent)
# ======================================================
def train_Agent(
    env, agent, logger,
    EPISODES, MAX_STEPS, BATCH_SIZE,
    CHECKPOINT_PATH, AUTO_SAVE_EVERY,
    LOGGIN_STATUS_EP=True,
    FINAL_MODEL_PATH=None,
):
    start_episode = 1
    if CHECKPOINT_PATH.exists():
        print("\n[Trainer] Found checkpoint. Loading...")
        start_episode = agent.load_checkpoint(CHECKPOINT_PATH) + 1
        print(f"[Trainer] Resuming from episode {start_episode}\n")
    else:
        print("\n[Trainer] No checkpoint. Starting from episode 1\n")

    for ep in range(start_episode, EPISODES + 1):
        state, info = env.reset()

        if hasattr(env, "noise_manager") and env.noise_manager:
            if hasattr(env.noise_manager, "on_episode_start"):
                env.noise_manager.on_episode_start(ep)

        current_setpoint = info.get("setpoint", None)
        episode_reward   = 0.0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)

            if hasattr(env, "noise_manager") and env.noise_manager:
                if hasattr(env.noise_manager, "step"):
                    env.noise_manager.step()

            next_state, reward, terminated, truncated, info = env.step(action)
            done             = terminated or truncated
            current_setpoint = info.get("setpoint", current_setpoint)

            agent.replay_buffer.push(
                state, action, reward, next_state, float(done)
            )

            if LOGGIN_STATUS_EP:
                logger.log(
                    episode    = ep,
                    setpoint   = current_setpoint,
                    step       = step,
                    state      = state,
                    action     = action,
                    reward     = reward,
                    next_state = next_state,
                    done       = done,
                )

            agent.update(BATCH_SIZE)
            state          = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {ep}/{EPISODES} | Reward = {episode_reward:.2f}")

        logger.save()
        logger.clear()
        agent.logger.save()
        agent.logger.clear()

        if ep % AUTO_SAVE_EVERY == 0:
            agent.save_checkpoint(ep, CHECKPOINT_PATH)

    agent.save_model(FINAL_MODEL_PATH)
    env.close()
    print("\n[Trainer] Training finished.")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    PROJECT_ROOT   = Path(__file__).resolve().parents[2]
    RL_CONFIG_PATH = PROJECT_ROOT / "src" / "API" / "config" / "rl_params.yaml"

    if RL_CONFIG_PATH.exists():
        with open(RL_CONFIG_PATH, "r", encoding="utf-8") as f:
            rl_cfg = yaml.safe_load(f)
    else:
        rl_cfg = {}

    train_cfg    = rl_cfg.get("training",     {})
    sac_cfg      = rl_cfg.get("sac",          {})
    noise_cfg    = rl_cfg.get("noise",        {})
    logger_cfg   = rl_cfg.get("logger",       {})
    lstm_sac_cfg = rl_cfg.get("lstm_sac_env", {})

    actor_cfg  = sac_cfg.get("actor",  {})
    critic_cfg = sac_cfg.get("critic", {})

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── LSTM Model path ───────────────────────────────────────
    lstm_model_path = PROJECT_ROOT / lstm_sac_cfg.get(
        "lstm_model_path", "models/lstm_model_2_s.pth"
    )
    scaler_zip_path = PROJECT_ROOT / lstm_sac_cfg.get(
        "scaler_zip", "config/AutoScaler_DATA_INPUT_DATA_OUTPUT_scalers.zip"
    )

    print(f"[Config] LSTM model : {lstm_model_path}")
    print(f"[Config] Scaler zip : {scaler_zip_path}")

    # ── โหลด LSTM + Scaler ────────────────────────────────────
    lstm_model = load_lstm_model(lstm_model_path, device)
    scaler     = ScalingZipLoader(scaler_zip_path)

    # ── Noise ─────────────────────────────────────────────────
    ou    = noise_cfg.get("ou_noise",  {})
    sched = noise_cfg.get("scheduler", {})
    gauss = noise_cfg.get("gaussian",  {})
    sens  = noise_cfg.get("sensor_noise", {})

    scheduler = NormalCurveScheduler(
        peak      = sched.get("peak",      3000),
        std       = sched.get("std",       1500),
        max_scale = sched.get("max_scale", 1.0),
    )
    noise_manager = NoiseManager(
        action_noise  = ScheduledNoise(
            OUNoise(mu=ou.get("mu",0.0), theta=ou.get("theta",0.15),
                    sigma=ou.get("sigma",0.25), dt=ou.get("dt",0.1)),
            scheduler
        ),
        process_noise = ScheduledNoise(
            GaussianNoise(sigma=gauss.get("sigma", 0.02)), scheduler
        ),
        sensor_noise  = BoundedGaussianNoise(
            sigma=sens.get("sigma",0.02), clip=sens.get("clip",0.05)
        ),
        enabled = noise_cfg.get("enabled", True),
    ) if noise_cfg.get("enabled", True) else None

    # ── LSTMEnv ───────────────────────────────────────────────
    env = LSTMEnv(
        lstm_model    = lstm_model,
        scaler        = scaler,
        sequence_size = lstm_sac_cfg.get("sequence_size", 30),
        min_action    = lstm_sac_cfg.get("min_action",    0.0),
        max_action    = lstm_sac_cfg.get("max_action",    10.0),
        max_steps     = train_cfg.get("max_steps",        200),
        device        = device,
        noise_manager = noise_manager,
    )

    state_dim  = env.state_dim
    action_dim = env.action_dim
    min_action = np.array([env.min_action])
    max_action = np.array([env.max_action])

    print(f"\n[Env] state_dim={state_dim} action_dim={action_dim}")
    print(f"[Env] action range: [{env.min_action}, {env.max_action}]")

    # ── Training parameters ───────────────────────────────────
    EPISODES        = train_cfg.get("episodes",        10000)
    MAX_STEPS       = train_cfg.get("max_steps",       200)
    BATCH_SIZE      = train_cfg.get("batch_size",      1080)
    AUTO_SAVE_EVERY = train_cfg.get("auto_save_every", 1)

    def _resolve_path(name, default_dir):
        p = Path(name)
        if p.suffix == "": p = p.with_suffix(".pt")
        if p.is_absolute(): return p
        if len(p.parts) == 1: return PROJECT_ROOT / default_dir / p
        return PROJECT_ROOT / p

    CHECKPOINT_PATH  = _resolve_path(
        lstm_sac_cfg.get("checkpoint_path",
            train_cfg.get("checkpoint_path", "sac_on_lstm")),
        "models/checkpoint"
    )
    FINAL_MODEL_PATH = str(_resolve_path(
        lstm_sac_cfg.get("final_model_path",
            train_cfg.get("final_model_path", "sac_on_lstm_final")),
        "models"
    ))

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "="*50)
    print("[Config] SAC on LSTM Parameters")
    print("="*50)
    print(f"  LSTM model    : {lstm_model_path.name}")
    print(f"  Episodes      : {EPISODES}")
    print(f"  Max Steps     : {MAX_STEPS}")
    print(f"  Batch Size    : {BATCH_SIZE}")
    print(f"  Learning Rate : {sac_cfg.get('learning_rate', 3e-4)}")
    print(f"  Checkpoint    : {CHECKPOINT_PATH}")
    print("="*50 + "\n")

    # ── Logger ────────────────────────────────────────────────
    logger = EpisodeLogger(
        folder   = logger_cfg.get("episode_folder",
                       str(PROJECT_ROOT / "logs" / "episode")),
        filename = logger_cfg.get("episode_filename", "lstm_sac_"),
    )

    # ── SAC Agent ─────────────────────────────────────────────
    agent = SACAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        min_action   = min_action,
        max_action   = max_action,
        lr           = sac_cfg.get("learning_rate", 3e-4),
        gamma        = sac_cfg.get("gamma",         0.99),
        tau          = sac_cfg.get("tau",           0.005),
        alpha        = sac_cfg.get("alpha",         0.4),
        logger_status = True,
        simple_layers_actor        = actor_cfg.get("layers",  2),
        simple_hidden_actor        = actor_cfg.get("hidden",  256),
        advanced_hidden_size_actor = None,
        simple_layers_critic         = critic_cfg.get("layers",  2),
        simple_hidden_critic         = critic_cfg.get("hidden",  256),
        advanced_hidden_sizes_critic = None,
        critic_encoder = critic_cfg.get("encoder", False),
        logger_path    = logger_cfg.get("agent_folder",
                             str(PROJECT_ROOT / "logs" / "agent" / "RC_Tank")),
        file_name_log  = logger_cfg.get("agent_filename", "lstm_sac_"),
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
        FINAL_MODEL_PATH = FINAL_MODEL_PATH,
    )