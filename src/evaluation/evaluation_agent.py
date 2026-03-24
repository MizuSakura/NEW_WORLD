# my_project/src/evaluation/evaluation_agent.py
"""
Evaluation Agent — Gym Environment
-----------------------------------
อ่านค่าจาก eval_params.yaml section gym

DASHBOARD_MODE=1  → ปิด pygame render, ปิด plt popup
                    print EVAL_STEP|... ทุก step → server parse → dashboard
DASHBOARD_MODE=0  → render pygame + plt popup ตามปกติ
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import os
import sys
import json
import numpy as np
import yaml
from pathlib import Path

DASHBOARD_MODE = os.environ.get("DASHBOARD_MODE", "0") == "1"

# ── matplotlib backend ────────────────────────────────────
import matplotlib
if DASHBOARD_MODE:
    matplotlib.use("Agg")   # ปิด GUI backend
import matplotlib.pyplot as plt

from src.agent.SAC_Agent import SACAgent
from src.environment.noise_manager import NoiseManager
import src.environment.register_envs
import gymnasium as gym


# ======================================================
# Helper
# ======================================================
# โฟลเดอร์ rc_models บน server (fallback)
_RC_MODELS_DIR = Path(r"E:\server_Project\SER_VER_STORE\rc_models")

def _resolve_path(name: str, default_dir: str, root: Path) -> Path:
    """
    Resolve model path ตามลำดับ:
    1. absolute path
    2. rc_models (server store)
    3. default_dir ใน project root
    4. relative to project root
    """
    p = Path(name)
    if p.suffix == "": p = p.with_suffix(".pt")

    # 1. absolute path
    if p.is_absolute():
        return p

    # 2. ชื่อไฟล์อย่างเดียว → ลอง rc_models ก่อน
    if len(p.parts) == 1:
        rc = _RC_MODELS_DIR / p
        if rc.exists(): return rc
        return root / default_dir / p

    # 3. relative path → ลอง rc_models / filename ก่อน
    rc = _RC_MODELS_DIR / p.name
    if rc.exists(): return rc

    return root / p


def _print_step(ep: int, step: int, level: float, action: float,
                reward: float, setpoint: float):
    """
    Print structured step data → server parse
    throttle: ทุก 5 steps เพื่อไม่ให้ WebSocket ล้น
    """
    if step % 5 == 0:
        print(
            f"EVAL_STEP|{ep}|{step}|{level:.4f}|{action:.4f}"
            f"|{reward:.4f}|{setpoint:.4f}",
            flush=True
        )


# ======================================================
# Evaluation logic
# ======================================================
def test_agent(env, agent, episodes=10, max_steps=500, deterministic=True):
    returns      = []
    trajectories = []

    for ep in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        states, actions, rewards, levels, setpoints = [], [], [], [], []

        setpoint = info.get("setpoint", 5.0)

        for step in range(max_steps):
            action = agent.select_action(state, deterministic=deterministic)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # render เฉพาะเมื่อไม่อยู่ใน DASHBOARD_MODE
            if not DASHBOARD_MODE:
                env.render()

            level      = float(env.level) if hasattr(env, "level") else float(state[0])
            action_val = float(np.array(action).item())
            setpoint   = float(info.get("setpoint", setpoint))

            states.append(state)
            actions.append(action_val)
            rewards.append(float(reward))
            levels.append(level)
            setpoints.append(setpoint)

            episode_reward += reward
            state = next_state

            # broadcast step ไป dashboard
            if DASHBOARD_MODE:
                _print_step(ep + 1, step + 1, level, action_val,
                            float(reward), setpoint)

            # if done:
            #     break

        returns.append(episode_reward)
        trajectories.append({
            "states":    np.array(states),
            "actions":   np.array(actions),
            "rewards":   np.array(rewards),
            "levels":    np.array(levels),
            "setpoints": np.array(setpoints),
        })
        print(f"[EVAL] Episode {ep+1}/{episodes} | Reward = {episode_reward:.2f}",
              flush=True)

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
    plt.savefig(plot_folder / "level_setpoint.png", bbox_inches="tight")
    plt.close()

    # Action
    plt.figure(figsize=(12, 4))
    plt.plot(traj["actions"], label="Action", color="#ef4444", alpha=0.8)
    plt.title(f"Action output (Episode {ep_idx+1})")
    plt.xlabel("Step"); plt.ylabel("Action Value")
    plt.grid(True); plt.legend()
    plt.savefig(plot_folder / "action_output.png", bbox_inches="tight")
    plt.close()

    # Reward per step
    plt.figure(figsize=(12, 4))
    plt.plot(traj["rewards"], label="Reward", color="#a855f7")
    plt.title(f"Reward per step (Episode {ep_idx+1})")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.grid(True); plt.legend()
    plt.savefig(plot_folder / "reward_curve.png", bbox_inches="tight")
    plt.close()

    # Total return per episode
    plt.figure(figsize=(6, 4))
    plt.plot(returns, marker="o", color="#f59e0b")
    plt.title("Total return per episode")
    plt.xlabel("Episode"); plt.ylabel("Total Return")
    plt.grid(True)
    plt.savefig(plot_folder / "total_returns.png", bbox_inches="tight")
    plt.close()

    print(f"[EVAL] Plots saved to {plot_folder}", flush=True)


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    PROJECT_ROOT     = Path(__file__).resolve().parents[2]
    EVAL_CONFIG_PATH = PROJECT_ROOT / "src" / "API" / "config" / "eval_params.yaml"

    with open(EVAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    cfg = eval_cfg.get("gym", {})

    MODEL_PATH    = _resolve_path(
        cfg.get("model_path", "models/checkpoint/Autosave.pt"),
        "models/checkpoint", PROJECT_ROOT
    )
    ENV_NAME      = cfg.get("env_name",      "RCTankEnv-v0")
    # DASHBOARD_MODE → force render_mode=None เพื่อปิด pygame
    RENDER_MODE   = None if DASHBOARD_MODE else cfg.get("render_mode", "human")
    EPISODES      = cfg.get("episodes",      10)
    MAX_STEPS     = cfg.get("max_steps",     500)
    DETERMINISTIC = cfg.get("deterministic", True)
    SAVE_PLOT     = cfg.get("save_plot",     True)
    PLOT_FOLDER   = PROJECT_ROOT / cfg.get("plot_folder", "logs/eval/gym")

    print(f"[Config] Model      : {MODEL_PATH}", flush=True)
    print(f"[Config] Env        : {ENV_NAME}", flush=True)
    print(f"[Config] Episodes   : {EPISODES} | Max Steps: {MAX_STEPS}", flush=True)
    print(f"[Config] Dashboard  : {DASHBOARD_MODE}", flush=True)

    # ── Environment ───────────────────────────────────────
    noise_manager = NoiseManager(enabled=False)
    env = gym.make(ENV_NAME,
                   render_mode=RENDER_MODE,
                   noise_manager=noise_manager)

    state, _ = env.reset()
    state_dim  = state.shape[0]
    action_dim = env.action_space.shape[0]
    min_action = env.action_space.low
    max_action = env.action_space.high

    print(f"[Config] State dim  : {state_dim}", flush=True)
    print(f"[Config] Action dim : {action_dim}", flush=True)
    print(f"[Config] Action     : {min_action} ~ {max_action}", flush=True)

    # ── Agent ─────────────────────────────────────────────
    agent = SACAgent(
        state_dim  = state_dim,
        action_dim = action_dim,
        min_action = min_action,
        max_action = max_action,
        logger_status = False,
    )
    agent.load_model(path=MODEL_PATH)

    # ── Evaluate ──────────────────────────────────────────
    returns, trajectories = test_agent(
        env, agent,
        episodes      = EPISODES,
        max_steps     = MAX_STEPS,
        deterministic = DETERMINISTIC,
    )

    returns_list = [float(r) for r in returns]
    mean_r = float(np.mean(returns_list))
    std_r  = float(np.std(returns_list))
    best_ep = int(np.argmax(returns_list))

    print(f"[EVAL] Mean return: {mean_r:.2f} \u00b1 {std_r:.2f}", flush=True)

    # ── ส่ง EVAL_RESULT ไป server เพื่อเก็บ history ──────────
    if DASHBOARD_MODE:
        result = {
            "mode":   "gym",
            "config": {
                "episodes":      EPISODES,
                "max_steps":     MAX_STEPS,
                "model_path":    str(MODEL_PATH),
                "env_name":      ENV_NAME,
                "deterministic": DETERMINISTIC,
            },
            "stats": {
                "mean":    round(mean_r, 4),
                "std":     round(std_r,  4),
                "min":     round(float(np.min(returns_list)), 4),
                "max":     round(float(np.max(returns_list)), 4),
                "best_ep": best_ep + 1,
            },
            "returns": returns_list,
            "trajectories": [
                {
                    "ep":       i + 1,
                    "levels":    traj["levels"].tolist(),
                    "actions":   traj["actions"].tolist(),
                    "rewards":   traj["rewards"].tolist(),
                    "setpoints": traj["setpoints"].tolist(),
                }
                for i, traj in enumerate(trajectories)
            ],
        }
        print("EVAL_RESULT|" + json.dumps(result, ensure_ascii=False), flush=True)

    if SAVE_PLOT:
        save_plots(trajectories, returns_list, PLOT_FOLDER)

    env.close()

