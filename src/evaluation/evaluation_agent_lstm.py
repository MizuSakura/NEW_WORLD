# my_project/src/evaluation/evaluation_agent_lstm.py
"""
Evaluation Agent — LSTM Environment
-------------------------------------
อ่านค่าจาก eval_params.yaml section lstm
ใช้ LSTMEnv (จาก train_SAC_on_LSTM.py) เป็น environment
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

from src.agent.SAC_Agent import SACAgent
from src.data.scaling_loader import ScalingZipLoader
from src.trainer.train_SAC_on_LSTM import LSTMEnv
from src.environment.noise_manager import NoiseManager


# ======================================================
# Helper
# ======================================================
def _resolve_path(name: str, root: Path, suffix: str = ".pt") -> Path:
    """
    Resolve model path ตามลำดับ:
    1. absolute path
    2. rc_models (server store) — ชื่อไฟล์อย่างเดียว
    3. relative to project root
    """
    _RC_MODELS = Path(r"E:\server_Project\SER_VER_STORE\rc_models")
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(suffix)
    if p.is_absolute():
        return p
    # ชื่อไฟล์อย่างเดียว → ลอง rc_models ก่อน
    if len(p.parts) == 1:
        rc = _RC_MODELS / p
        if rc.exists(): return rc
    else:
        # relative path → ลอง rc_models/filename ก่อน
        rc = _RC_MODELS / p.name
        if rc.exists(): return rc
    return root / p


def _load_lstm_model(lstm_model_path: Path, device):
    """
    โหลด LSTM model จาก .pth — รองรับ VanillaLSTM / DeepLSTM / BiLSTM
    model ถูก save ด้วย torch.save(model, path) (full model)
    """
    model = torch.load(lstm_model_path, map_location=device)
    model.eval()
    print(f"[LSTM] Loaded: {lstm_model_path.name} → {type(model).__name__}")
    return model


# ======================================================
# Evaluation logic (reuse pattern จาก gym version)
# ======================================================
def test_agent_lstm(env, agent, episodes=10, max_steps=500, deterministic=True):
    returns      = []
    trajectories = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        states, actions, rewards, levels, setpoints = [], [], [], [], []

        for step in range(max_steps):
            action = agent.select_action(state, deterministic=deterministic)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(float(np.array(action).item()))
            rewards.append(reward)
            levels.append(info.get("level", env._level))
            setpoints.append(info.get("setpoint", env._setpoint))

            episode_reward += reward
            state = next_state

            if done:
                break

        returns.append(episode_reward)
        trajectories.append({
            "states":    np.array(states),
            "actions":   np.array(actions),
            "rewards":   np.array(rewards),
            "levels":    np.array(levels),
            "setpoints": np.array(setpoints),
        })
        print(f"[EVAL] Episode {ep+1}/{episodes} | Reward = {episode_reward:.2f}")

    return returns, trajectories


def save_plots(trajectories, returns, plot_folder: Path, ep_idx: int = 0):
    plot_folder.mkdir(parents=True, exist_ok=True)
    traj = trajectories[min(ep_idx, len(trajectories) - 1)]

    # Level vs Setpoint
    plt.figure(figsize=(12, 4))
    plt.plot(traj["levels"],    label="Level",    color="#3b82f6")
    plt.plot(traj["setpoints"], label="Setpoint", color="#10b981", linestyle="--")
    plt.title(f"Level vs Setpoint (Episode {ep_idx+1})")
    plt.xlabel("Step"); plt.ylabel("Value")
    plt.grid(True); plt.legend()
    plt.savefig(plot_folder / "level_setpoint.png")
    plt.close()

    # Action
    plt.figure(figsize=(12, 4))
    plt.plot(traj["actions"], label="Action", color="#ef4444", alpha=0.8)
    plt.title(f"Action output (Episode {ep_idx+1})")
    plt.xlabel("Step"); plt.ylabel("Action Value")
    plt.grid(True); plt.legend()
    plt.savefig(plot_folder / "action_output.png")
    plt.close()

    # Reward per step
    plt.figure(figsize=(12, 4))
    plt.plot(traj["rewards"], label="Reward", color="#a855f7")
    plt.title(f"Reward per step (Episode {ep_idx+1})")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.grid(True); plt.legend()
    plt.savefig(plot_folder / "reward_curve.png")
    plt.close()

    # Total return per episode
    plt.figure(figsize=(6, 4))
    plt.plot(returns, marker="o", color="#f59e0b")
    plt.title("Total return per episode")
    plt.xlabel("Episode"); plt.ylabel("Total Return")
    plt.grid(True)
    plt.savefig(plot_folder / "total_returns.png")
    plt.close()

    print(f"[EVAL] Plots saved to {plot_folder}")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    PROJECT_ROOT     = Path(__file__).resolve().parents[2]
    EVAL_CONFIG_PATH = PROJECT_ROOT / "src" / "API" / "config" / "eval_params.yaml"

    with open(EVAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    cfg     = eval_cfg.get("lstm", {})
    env_cfg = cfg.get("env", {})

    MODEL_PATH      = _resolve_path(cfg.get("model_path",      "models/checkpoint/sac_on_lstm.pt"),  PROJECT_ROOT)
    LSTM_MODEL_PATH = _resolve_path(cfg.get("lstm_model_path", "models/lstm_model_2_s.pth"),         PROJECT_ROOT, suffix=".pth")
    SCALER_ZIP      = _resolve_path(cfg.get("scaler_zip",      "config/AutoScaler_DATA_INPUT_DATA_OUTPUT_scalers.zip"), PROJECT_ROOT, suffix=".zip")

    EPISODES      = cfg.get("episodes",      10)
    MAX_STEPS     = cfg.get("max_steps",     500)
    DETERMINISTIC = cfg.get("deterministic", True)
    SAVE_PLOT     = cfg.get("save_plot",     True)
    PLOT_FOLDER   = PROJECT_ROOT / cfg.get("plot_folder", "logs/eval/lstm")

    SEQUENCE_SIZE = env_cfg.get("sequence_size", 30)
    MIN_ACTION    = float(env_cfg.get("min_action", 0.0))
    MAX_ACTION    = float(env_cfg.get("max_action", 10.0))

    print(f"\n[Config] SAC Model  : {MODEL_PATH}")
    print(f"[Config] LSTM Model : {LSTM_MODEL_PATH}")
    print(f"[Config] Scaler ZIP : {SCALER_ZIP}")
    print(f"[Config] Episodes   : {EPISODES} | Max Steps: {MAX_STEPS}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── โหลด LSTM + Scaler ───────────────────────────────
    lstm_model = _load_lstm_model(LSTM_MODEL_PATH, device)
    scaler     = ScalingZipLoader(SCALER_ZIP)

    # ── สร้าง LSTMEnv ─────────────────────────────────────
    noise_manager = NoiseManager(enabled=False)
    env = LSTMEnv(
        lstm_model    = lstm_model,
        scaler        = scaler,
        sequence_size = SEQUENCE_SIZE,
        min_action    = MIN_ACTION,
        max_action    = MAX_ACTION,
        max_steps     = MAX_STEPS,
        device        = device,
        noise_manager = noise_manager,
    )

    state_dim  = env.state_dim
    action_dim = env.action_dim
    min_action = np.array([MIN_ACTION])
    max_action = np.array([MAX_ACTION])

    print(f"State dim   : {state_dim}")
    print(f"Action dim  : {action_dim}")
    print(f"Action range: {MIN_ACTION} ~ {MAX_ACTION}\n")

    # ── โหลด SAC Agent ────────────────────────────────────
    agent = SACAgent(
        state_dim  = state_dim,
        action_dim = action_dim,
        min_action = min_action,
        max_action = max_action,
        logger_status = False,
    )
    agent.load_model(path=MODEL_PATH)

    # ── Evaluate ──────────────────────────────────────────
    returns, trajectories = test_agent_lstm(
        env, agent,
        episodes      = EPISODES,
        max_steps     = MAX_STEPS,
        deterministic = DETERMINISTIC,
    )

    print(f"\n[EVAL] Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

    if SAVE_PLOT:
        save_plots(trajectories, returns, PLOT_FOLDER)
