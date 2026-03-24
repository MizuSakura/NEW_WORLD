# my_project/src/trainer/train_SAC_apex.py
"""
Ape-X Distributed SAC Trainer
==============================
Architecture:
    Actor workers  (CPU, multiprocessing ×N)
        - มี RCTankEnv + local n-step buffer ของตัวเอง
        - คำนวณ R_n locally (thread-safe)
        - ส่ง transition เข้า exp_queue
        - sync weights จาก weight_queue ทุก SYNC_EVERY steps

    Replay process  (CPU, ×1)
        - รับ transition จาก exp_queue
        - push เข้า PER buffer (1 process = thread-safe)
        - ส่ง batch ออกไป batch_queue ให้ learner

    Learner process  (GPU, ×1)
        - รับ batch จาก batch_queue
        - update actor + critic
        - ส่ง new weights → weight_queue
        - print STEP|... stdout ให้ server stream ได้

stdout protocol เหมือน train_SAC_agent.py ทุกตัวอักษร
→ server._stream_log ไม่ต้องแก้

Usage:
    python -m src.trainer.train_SAC_apex
"""

import os
import sys
import time
import yaml
import copy
import pickle
import numpy as np
from pathlib import Path
from collections import deque
import multiprocessing as mp
mp.set_start_method("spawn", force=True)   # Windows + CUDA ต้องใช้ spawn

import torch
import torch.nn.functional as F

import src.environment.register_envs      # noqa: F401
import gymnasium as gym

from src.agent.network import Actor, Critic
from src.agent.replaybuffer.n_step_per_replaybuffer import NStepPERReplayBuffer
from src.utils.logger_pyarrow import EpisodeLogger, MetricLogger
from src.environment.noise_manager import (
    NoiseManager, GaussianNoise, BoundedGaussianNoise,
    OUNoise, ScheduledNoise, NormalCurveScheduler,
)


# ======================================================================
# Constants (อ่านจาก yaml ตอน __main__ แล้วส่งเข้า process)
# ======================================================================
SENTINEL = None   # signal ให้ process หยุด


# ======================================================================
# Helpers ที่ใช้ร่วมกัน
# ======================================================================

def _resolve_path(name: str, default_dir: str, root: Path) -> Path:
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(".pt")
    if p.is_absolute():
        return p
    if len(p.parts) == 1:
        return root / default_dir / p
    return root / p


def _build_noise(noise_cfg: dict, seed: int) -> NoiseManager:
    ou    = noise_cfg.get("ou_noise",     {})
    gauss = noise_cfg.get("gaussian",     {})
    sens  = noise_cfg.get("sensor_noise", {})
    sched = noise_cfg.get("scheduler",    {})
    scheduler = NormalCurveScheduler(
        peak=sched.get("peak", 3000),
        std=sched.get("std",  1500),
        max_scale=sched.get("max_scale", 1.0),
    )
    return NoiseManager(
        action_noise=ScheduledNoise(
            OUNoise(mu=ou.get("mu", 0.), theta=ou.get("theta", 0.15),
                    sigma=ou.get("sigma", 0.10), dt=ou.get("dt", 0.1)),
            scheduler),
        process_noise=ScheduledNoise(
            GaussianNoise(sigma=gauss.get("sigma", 0.01)), scheduler),
        sensor_noise=BoundedGaussianNoise(
            sigma=sens.get("sigma", 0.02), clip=sens.get("clip", 0.05)),
        enabled=noise_cfg.get("enabled", True),
    )


def _make_env(env_name: str, noise_cfg: dict, rank: int):
    rng = np.random.default_rng(rank * 1000 + 42)
    R   = float(rng.uniform(1.0, 2.5))
    C   = float(rng.uniform(1.5, 3.0))
    env = gym.make(env_name, render_mode=None,
                   noise_manager=_build_noise(noise_cfg, seed=rank),
                   R=R, C=C, save_episode_image=False)
    env.reset(seed=rank)
    return env


def _build_actor_net(state_dim, action_dim, min_action, max_action,
                     layers, hidden) -> Actor:
    return Actor(
        state_dim, action_dim,
        np.array(min_action, dtype=np.float32),
        np.array(max_action, dtype=np.float32),
        simple_layers=layers,
        simple_hidden=hidden,
    )


# ======================================================================
# Local N-step buffer (per actor, ไม่ share กัน → thread-safe)
# ======================================================================

class LocalNStepBuffer:
    """
    คำนวณ n-step return ใน actor process เอง
    output: (s_0, a_0, R_n, s_n, done_n) ซึ่ง push-safe เข้า PER ได้เลย
    """
    def __init__(self, n_step: int, gamma: float):
        self.n   = n_step
        self.g   = gamma
        self.buf = deque(maxlen=n_step)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def ready(self) -> bool:
        return len(self.buf) == self.n

    def get(self):
        """คืน (s_0, a_0, R_n, s_n, done_n) หรือ None ถ้าไม่พร้อม"""
        if not self.ready():
            return None
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buf):
            R += (self.g ** i) * r
            if d:
                break
        s_0, a_0 = self.buf[0][0], self.buf[0][1]
        _, _, _, s_n, done_n = self.buf[-1]
        return s_0, a_0, R, s_n, float(done_n)

    def reset(self):
        self.buf.clear()


# ======================================================================
# Actor worker  (CPU subprocess)
# ======================================================================

def actor_worker(
    rank:        int,
    env_name:    str,
    noise_cfg:   dict,
    actor_cfg:   dict,
    state_dim:   int,
    action_dim:  int,
    min_action,
    max_action,
    n_step:      int,
    gamma:       float,
    sync_every:  int,
    total_eps:   int,
    max_steps:   int,
    exp_queue:   mp.Queue,
    weight_queue: mp.Queue,
    stop_event:  mp.Event,
    log_every:   int = 5,
):
    import src.environment.register_envs   # noqa

    env         = _make_env(env_name, noise_cfg, rank)
    local_buf   = LocalNStepBuffer(n_step, gamma)
    actor_net   = _build_actor_net(state_dim, action_dim,
                                   min_action, max_action,
                                   actor_cfg["layers"], actor_cfg["hidden"])
    actor_net.eval()

    obs, info       = env.reset()
    ep_reward       = 0.0
    ep_step         = 0
    global_step     = 0
    episode_count   = 0
    setpoint        = info.get("setpoint", 0.0)

    while not stop_event.is_set() and episode_count < total_eps:

        # ── sync weights from learner (CPU weights) ────────────────
        if global_step % sync_every == 0:
            try:
                while not weight_queue.empty():
                    w_bytes = weight_queue.get_nowait()
                    state_dict = pickle.loads(w_bytes)
                    actor_net.load_state_dict(state_dict)
            except Exception:
                pass

        # ── select action (CPU, no grad) ──────────────────────────
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            action, _ = actor_net.sample(t)
            action_np = action.cpu().numpy()[0]

        # ── env step ──────────────────────────────────────────────
        next_obs, reward, term, trunc, info = env.step(action_np)
        setpoint  = info.get("setpoint", setpoint)
        done      = term or trunc
        ep_reward += reward
        ep_step   += 1
        global_step += 1

        # ── local n-step buffer ───────────────────────────────────
        local_buf.push(obs, action_np, reward, next_obs, float(done))
        transition = local_buf.get()
        if transition is not None:
            exp_queue.put(transition)

        # ── periodic stdout (rank 0 only) ─────────────────────────
        if rank == 0 and ep_step % log_every == 0:
            print(
                f"STEP|{episode_count}|{ep_step}|"
                f"{float(next_obs[0]):.4f}|"
                f"{float(action_np[0]):.4f}|"
                f"{float(reward):.4f}|"
                f"{float(setpoint):.4f}",
                flush=True,
            )

        obs = next_obs

        if done:
            if rank == 0:
                print(
                    f"Episode {episode_count}/{total_eps} | "
                    f"Reward = {ep_reward:.2f} | "
                    f"Worker = {rank} | "
                    f"Steps = {ep_step} | "
                    f"Level = {float(obs[0]):.3f} | "
                    f"Setpoint = {float(setpoint):.3f}",
                    flush=True,
                )
            local_buf.reset()
            obs, info    = env.reset()
            setpoint     = info.get("setpoint", 0.0)
            ep_reward    = 0.0
            ep_step      = 0
            episode_count += 1

    env.close()
    exp_queue.put(SENTINEL)


# ======================================================================
# Replay process  (CPU subprocess)
# ======================================================================

def replay_process(
    state_dim:    int,
    action_dim:   int,
    capacity:     int,
    n_step:       int,
    gamma:        float,
    per_alpha:    float,
    per_beta:     float,
    batch_size:   int,
    num_actors:   int,
    exp_queue:    mp.Queue,
    batch_queue:  mp.Queue,
    prio_queue:   mp.Queue,
    stop_event:   mp.Event,
    min_buffer:   int = 2000,
):
    buf = NStepPERReplayBuffer(
        capacity=capacity,
        state_dim=state_dim,
        action_dim=action_dim,
        n_step=1,
        gamma=gamma,
        alpha=per_alpha,
        beta=per_beta,
    )

    done_actors  = 0
    total_pushed = 0

    while not stop_event.is_set():

        # ── รับ transition ────────────────────────────────────────
        try:
            item = exp_queue.get(timeout=0.05)
            if item is SENTINEL:
                done_actors += 1
                if done_actors >= num_actors:
                    break
                continue
            s0, a0, R_n, sn, dn = item
            buf.push(s0, a0, R_n, sn, dn)
            total_pushed += 1
        except Exception:
            pass

        # ── update priorities ─────────────────────────────────────
        try:
            while not prio_queue.empty():
                tree_idx, td_err = prio_queue.get_nowait()
                buf.update_priorities(tree_idx, td_err)
        except Exception:
            pass

        # ── sample batch ──────────────────────────────────────────
        if len(buf) >= min_buffer and not batch_queue.full():
            try:
                batch = buf.sample(batch_size)
                batch_queue.put(batch)
            except Exception:
                pass

    batch_queue.put(SENTINEL)


# ======================================================================
# Learner process  (GPU subprocess)
# ======================================================================

def learner_process(
    state_dim:    int,
    action_dim:   int,
    min_action,
    max_action,
    actor_cfg:    dict,
    critic_cfg:   dict,
    lr:           float,
    gamma:        float,
    tau:          float,
    alpha:        float,
    total_eps:    int,
    batch_queue:  mp.Queue,
    prio_queue:   mp.Queue,
    weight_queue: mp.Queue,
    stop_event:   mp.Event,
    checkpoint_path: str,
    final_model_path: str,
    logger_path:  str,
    logger_file:  str,
    broadcast_every: int = 10,
    save_every:   int = 100,
    device_str:   str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}", flush=True)

    min_a = np.array(min_action, dtype=np.float32)
    max_a = np.array(max_action, dtype=np.float32)

    # ── build networks ────────────────────────────────────────────
    actor = Actor(state_dim, action_dim, min_a, max_a,
                  simple_layers=actor_cfg["layers"],
                  simple_hidden=actor_cfg["hidden"]).to(device)

    critic = Critic(state_dim, action_dim,
                    simple_layers=critic_cfg["layers"],
                    simple_hidden=critic_cfg["hidden"],
                    use_encoder=critic_cfg.get("encoder", False)).to(device)

    target_critic = copy.deepcopy(critic).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    # ── resume checkpoint ─────────────────────────────────────────
    ckpt_path = Path(checkpoint_path)
    start_ep  = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        target_critic.load_state_dict(ckpt["target_critic"])
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        start_ep = ckpt.get("episode", 0)
        print(f"[Learner] Resumed from episode {start_ep}", flush=True)
    else:
        print("[Learner] No checkpoint. Starting fresh.", flush=True)

    metric_logger = MetricLogger(folder=logger_path, filename=logger_file,
                                 auto_increment=True)

    update_count = 0
    episode_est  = start_ep

    # broadcast initial weights
    _broadcast(actor, weight_queue)

    while not stop_event.is_set():

        # ── รับ batch ────────────────────────────────────────────
        try:
            batch = batch_queue.get(timeout=0.1)
        except Exception:
            continue

        if batch is SENTINEL:
            break

        # ── unpack + move to device ───────────────────────────────
        if len(batch) == 5:
            state, action, reward, next_state, done = batch
            tree_idx, is_weights = None, None
        else:
            state, action, reward, next_state, done, tree_idx, is_weights = batch

        def to_t(x):
            if torch.is_tensor(x):
                return x.float().to(device)
            return torch.as_tensor(
                np.array(x), dtype=torch.float32, device=device)

        state      = to_t(state)
        action     = to_t(action)
        reward     = to_t(reward)
        next_state = to_t(next_state)
        done       = to_t(done)
        if is_weights is not None:
            is_weights = to_t(is_weights)

        # ── target Q ─────────────────────────────────────────────
        with torch.no_grad():
            next_a, next_logp = actor.sample(next_state)
            q1_t, q2_t = target_critic(next_state, next_a)
            q_min = torch.min(q1_t, q2_t)
            target_q = reward + (1.0 - done) * (q_min - alpha * next_logp)
            target_q = torch.clamp(target_q, -1e6, 1e6)

        # ── critic loss ───────────────────────────────────────────
        q1, q2 = critic(state, action)
        td1 = q1 - target_q
        td2 = q2 - target_q
        td  = 0.5 * (td1.abs() + td2.abs())

        if is_weights is not None:
            critic_loss = (is_weights * (td1.pow(2) + td2.pow(2))).mean()
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # ── priority update ───────────────────────────────────────
        if tree_idx is not None:
            safe_td = torch.clamp(td, 0.0, 1e6).detach().squeeze(-1)
            prio_queue.put((tree_idx, safe_td.cpu().numpy()))

        # ── actor loss ────────────────────────────────────────────
        a_pi, log_pi = actor.sample(state)
        q1_pi, q2_pi = critic(state, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        if torch.isnan(q_pi).any() or torch.isnan(log_pi).any():
            continue

        actor_loss = (alpha * log_pi - q_pi).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # ── soft update target ────────────────────────────────────
        for p, tp in zip(critic.parameters(), target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # ── logging ───────────────────────────────────────────────
        metric_logger.log("loss_actor",  actor_loss.item())
        metric_logger.log("loss_critic", critic_loss.item())
        metric_logger.log("q1_mean",     q1.mean().item())
        metric_logger.log("entropy",     -log_pi.mean().item())
        metric_logger.log("alpha",       alpha)

        update_count += 1

        # ── broadcast weights → actors ────────────────────────────
        if update_count % broadcast_every == 0:
            _broadcast(actor, weight_queue)
            episode_est += 1

        # ── checkpoint ───────────────────────────────────────────
        if update_count % (save_every * 10) == 0:
            _save_checkpoint(actor, critic, target_critic,
                             actor_opt, critic_opt,
                             episode_est, ckpt_path)
            metric_logger.save()
            metric_logger.clear()

    # ── final save ────────────────────────────────────────────────
    _save_checkpoint(actor, critic, target_critic,
                     actor_opt, critic_opt, episode_est, ckpt_path)
    _save_model(actor, critic, target_critic, final_model_path)
    metric_logger.save()
    print(f"[Learner] Finished. Total updates: {update_count}", flush=True)


# ======================================================================
# Helpers for learner
# ======================================================================

def _broadcast(actor: Actor, weight_queue: mp.Queue):
    """
    Serialize actor weights (CPU) แล้วส่งเข้า weight_queue
    บันทึก device ก่อน → ย้ายไป CPU → serialize → ย้ายกลับ device เดิม
    """
    try:
        # บันทึก device ปัจจุบัน
        current_device = next(actor.parameters()).device

        # ย้ายไป CPU เพื่อ serialize
        actor.cpu()
        w = pickle.dumps(actor.state_dict())

        # ย้ายกลับ device เดิม (CUDA หรือ CPU)
        actor.to(current_device)

        # ล้าง queue เก่าก่อน
        while not weight_queue.empty():
            try:
                weight_queue.get_nowait()
            except Exception:
                break

        # ส่งให้ทุก worker
        for _ in range(weight_queue._maxsize or 8):
            try:
                weight_queue.put_nowait(w)
            except Exception:
                break

    except Exception as e:
        print(f"[Learner] broadcast error: {e}", flush=True)


def _save_checkpoint(actor, critic, target_critic,
                     actor_opt, critic_opt, episode, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episode":       episode,
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "actor_opt":     actor_opt.state_dict(),
        "critic_opt":    critic_opt.state_dict(),
        "hyperparams": {
            "state_dim":  actor.state_dim,
            "action_dim": actor.action_dim,
            "min_action": actor.min_action.cpu().tolist(),
            "max_action": actor.max_action.cpu().tolist(),
            "simple_layers_actor":  actor.simple_layers,
            "simple_hidden_actor":  actor.simple_hidden,
            "advanced_hidden_size_actor": actor.advanced_hidden_sizes,
            "simple_layers_critic":  critic.q1_net[0].in_features,
            "simple_hidden_critic":  256,
            "advanced_hidden_sizes_critic": None,
            "critic_encoder": False,
            "gamma": 0.99, "tau": 0.005, "alpha": 0.05,
        },
    }, path)
    print(f"[Learner] Checkpoint saved → {path} (ep {episode})", flush=True)


def _save_model(actor, critic, target_critic, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "hyperparams": {
            "state_dim":  actor.state_dim,
            "action_dim": actor.action_dim,
            "min_action": actor.min_action.cpu().tolist(),
            "max_action": actor.max_action.cpu().tolist(),
            "simple_layers_actor":  actor.simple_layers,
            "simple_hidden_actor":  actor.simple_hidden,
            "advanced_hidden_size_actor": actor.advanced_hidden_sizes,
            "simple_layers_critic":  256,
            "simple_hidden_critic":  256,
            "advanced_hidden_sizes_critic": None,
            "critic_encoder": False,
            "gamma": 0.99, "tau": 0.005, "alpha": 0.05,
        },
    }, path)
    print(f"[Learner] Model saved → {path}", flush=True)


# ======================================================================
# Probe env for dims
# ======================================================================

def _probe_env(env_name, noise_cfg):
    env = _make_env(env_name, noise_cfg, rank=0)
    obs, _ = env.reset()
    sd = obs.shape[0]
    ad = env.action_space.shape[0]
    mn = env.action_space.low.copy()
    mx = env.action_space.high.copy()
    env.close()
    return sd, ad, mn, mx


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":

    # ── load config ───────────────────────────────────────────────
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
    actor_cfg  = sac_cfg.get("actor",   {"layers": 2, "hidden": 256})
    critic_cfg = sac_cfg.get("critic",  {"layers": 2, "hidden": 256, "encoder": False})

    # ── hyperparams ───────────────────────────────────────────────
    ENV_NAME        = train_cfg.get("env_name",             "RCTankEnv-v0")
    EPISODES        = train_cfg.get("episodes",             10000)
    MAX_STEPS       = train_cfg.get("max_steps",            200)
    BATCH_SIZE      = train_cfg.get("batch_size",           512)
    NUM_ACTORS      = train_cfg.get("num_envs",             6)
    SYNC_EVERY      = train_cfg.get("apex_sync_every",      20)
    BROADCAST_EVERY = train_cfg.get("apex_broadcast_every", 10)
    CAPACITY        = sac_cfg.get("replay_capacity",        200000)
    N_STEP          = sac_cfg.get("n_step",                 3)
    GAMMA           = sac_cfg.get("gamma",                  0.99)
    PER_ALPHA       = sac_cfg.get("per_alpha",              0.6)
    PER_BETA        = sac_cfg.get("per_beta",               0.4)
    LR              = sac_cfg.get("learning_rate",          3e-4)
    TAU             = sac_cfg.get("tau",                    0.005)
    ALPHA           = sac_cfg.get("alpha",                  0.05)
    MIN_BUFFER      = max(BATCH_SIZE * 4, 2000)

    CKPT_PATH  = str(_resolve_path(
        train_cfg.get("checkpoint_path",  "Autosave_apex"),
        "models/checkpoint", PROJECT_ROOT))
    FINAL_PATH = str(_resolve_path(
        train_cfg.get("final_model_path", "Test_history_apex"),
        "models", PROJECT_ROOT))

    LOGGER_PATH = logger_cfg.get("agent_folder",
                      str(PROJECT_ROOT / "logs" / "agent" / "RC_Tank"))
    LOGGER_FILE = logger_cfg.get("agent_filename", "apex_")

    # ── probe env ─────────────────────────────────────────────────
    STATE_DIM, ACTION_DIM, MIN_ACTION, MAX_ACTION = _probe_env(
        ENV_NAME, noise_cfg)

    print("\n" + "="*58, flush=True)
    print("[Config] Ape-X Distributed SAC", flush=True)
    print("="*58, flush=True)
    print(f"  Env          : {ENV_NAME}",           flush=True)
    print(f"  Actors       : {NUM_ACTORS}  (CPU)",  flush=True)
    print(f"  State dim    : {STATE_DIM}",           flush=True)
    print(f"  Action dim   : {ACTION_DIM}",          flush=True)
    print(f"  Episodes     : {EPISODES}",            flush=True)
    print(f"  Batch size   : {BATCH_SIZE}",          flush=True)
    print(f"  Buffer cap   : {CAPACITY}",            flush=True)
    print(f"  N-step       : {N_STEP}",              flush=True)
    print(f"  Sync every   : {SYNC_EVERY} steps",   flush=True)
    print(f"  Checkpoint   : {CKPT_PATH}",           flush=True)
    print("="*58 + "\n",                             flush=True)

    # ── queues & events ───────────────────────────────────────────
    exp_queue    = mp.Queue(maxsize=5000)
    batch_queue  = mp.Queue(maxsize=8)
    prio_queue   = mp.Queue(maxsize=500)
    weight_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    stop_event   = mp.Event()

    # ── spawn processes ───────────────────────────────────────────
    processes = []

    # Replay process
    p_replay = mp.Process(
        target=replay_process,
        name="replay",
        kwargs=dict(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            capacity=CAPACITY, n_step=N_STEP, gamma=GAMMA,
            per_alpha=PER_ALPHA, per_beta=PER_BETA,
            batch_size=BATCH_SIZE, num_actors=NUM_ACTORS,
            exp_queue=exp_queue, batch_queue=batch_queue,
            prio_queue=prio_queue, stop_event=stop_event,
            min_buffer=MIN_BUFFER,
        ),
        daemon=True,
    )
    p_replay.start()
    processes.append(p_replay)

    # Learner process
    p_learner = mp.Process(
        target=learner_process,
        name="learner",
        kwargs=dict(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            min_action=MIN_ACTION.tolist(), max_action=MAX_ACTION.tolist(),
            actor_cfg=actor_cfg, critic_cfg=critic_cfg,
            lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
            total_eps=EPISODES,
            batch_queue=batch_queue, prio_queue=prio_queue,
            weight_queue=weight_queue, stop_event=stop_event,
            checkpoint_path=CKPT_PATH, final_model_path=FINAL_PATH,
            logger_path=LOGGER_PATH, logger_file=LOGGER_FILE,
            broadcast_every=BROADCAST_EVERY,
            save_every=100,
            device_str="cuda",
        ),
        daemon=True,
    )
    p_learner.start()
    processes.append(p_learner)

    # Actor workers (spawn หลัง learner เพื่อให้ initial weights พร้อม)
    time.sleep(1.0)
    for rank in range(NUM_ACTORS):
        p_actor = mp.Process(
            target=actor_worker,
            name=f"actor_{rank}",
            kwargs=dict(
                rank=rank, env_name=ENV_NAME, noise_cfg=noise_cfg,
                actor_cfg=actor_cfg,
                state_dim=STATE_DIM, action_dim=ACTION_DIM,
                min_action=MIN_ACTION.tolist(),
                max_action=MAX_ACTION.tolist(),
                n_step=N_STEP, gamma=GAMMA,
                sync_every=SYNC_EVERY,
                total_eps=EPISODES // NUM_ACTORS,
                max_steps=MAX_STEPS,
                exp_queue=exp_queue, weight_queue=weight_queue,
                stop_event=stop_event,
            ),
            daemon=True,
        )
        p_actor.start()
        processes.append(p_actor)

    print(f"[Main] {NUM_ACTORS} actor workers + 1 replay + 1 learner started",
          flush=True)

    # ── wait for learner to finish ────────────────────────────────
    try:
        p_learner.join()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — stopping all processes", flush=True)
    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        print("[Main] All processes stopped. Training finished.", flush=True)