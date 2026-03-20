# my_project/src/evaluation/evaluation_agent_real.py
"""
Evaluation Agent — Real Hardware
---------------------------------
อ่านค่าจาก eval_params.yaml section real
ใช้ Real_env_remote (Apply_real_env.py) เป็น environment
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.agent.SAC_Agent import SACAgent
from src.environment.Apply_real_env import Real_env_remote


# ======================================================
# Helper
# ======================================================
def _resolve_path(name: str, root: Path, suffix: str = ".pt") -> Path:
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(suffix)
    if p.is_absolute():
        return p
    return root / p


# ======================================================
# Evaluation logic
# ======================================================
def test_agent_real(env, agent, episodes=5, max_steps=500, deterministic=True):
    returns      = []
    trajectories = []

    for ep in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        states, actions, rewards, levels, setpoints = [], [], [], [], []

        setpoint = info.get("setpoint", env.setpoint)

        for step in range(max_steps):
            action = agent.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = env.step(action=action)

            level = info.get("raw_level", 0.0)

            states.append(state)
            actions.append(float(np.array(action).item()))
            rewards.append(reward)
            levels.append(level)
            setpoints.append(info.get("setpoint", setpoint))

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
    plt.title(f"Level vs Setpoint — Real Hardware (Episode {ep_idx+1})")
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
    plt.title("Total return per episode — Real Hardware")
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
    RL_CONFIG_PATH   = PROJECT_ROOT / "src" / "API" / "config" / "rl_params.yaml"

    with open(EVAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    cfg     = eval_cfg.get("real", {})
    env_cfg = cfg.get("env", {})

    MODEL_PATH    = _resolve_path(
        cfg.get("model_path", "models/checkpoint/sac_checkpoint_real.pt"),
        PROJECT_ROOT
    )
    EPISODES      = cfg.get("episodes",      5)
    MAX_STEPS     = cfg.get("max_steps",     500)
    DETERMINISTIC = cfg.get("deterministic", True)
    SAVE_PLOT     = cfg.get("save_plot",     True)
    PLOT_FOLDER   = PROJECT_ROOT / cfg.get("plot_folder", "logs/eval/real")

    IP_HOST          = env_cfg.get("ip_host",          "192.168.1.100")
    PORT             = env_cfg.get("port",              502)
    MIN_ACTION       = float(env_cfg.get("min_action",  0.0))
    MAX_ACTION       = float(env_cfg.get("max_action",  10.0))
    SETPOINT         = float(env_cfg.get("setpoint",    5.0))
    DELAY_OF_ACTION  = float(env_cfg.get("delay_of_action", 0.2))
    ADDRESS_SENSOR   = env_cfg.get("address_sensor",   1)
    ADDRESS_ACTUATOR = env_cfg.get("address_actuator",  1025)

    print(f"\n[Config] Model   : {MODEL_PATH}")
    print(f"[Config] Host    : {IP_HOST}:{PORT}")
    print(f"[Config] Episodes: {EPISODES} | Max Steps: {MAX_STEPS}\n")

    # ── สร้าง Real Environment ─────────────────────────────
    # Real_env_remote อ่าน state config จาก rl_params.yaml โดยตรง
    env = Real_env_remote(
        ip_host          = IP_HOST,
        port             = PORT,
        min_action       = MIN_ACTION,
        max_action       = MAX_ACTION,
        setpoint         = SETPOINT,
        delay_of_action  = DELAY_OF_ACTION,
        address_sensor   = ADDRESS_SENSOR,
        address_actuator = ADDRESS_ACTUATOR,
        config_path      = RL_CONFIG_PATH,
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
    returns, trajectories = test_agent_real(
        env, agent,
        episodes      = EPISODES,
        max_steps     = MAX_STEPS,
        deterministic = DETERMINISTIC,
    )

    print(f"\n[EVAL] Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

    if SAVE_PLOT:
        save_plots(trajectories, returns, PLOT_FOLDER)