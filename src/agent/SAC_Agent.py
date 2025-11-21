# my_project/src/agent/SAC_Agent.py

import torch
import torch.nn.functional as F
from pathlib import Path

from src.agent.network import Logger, Actor, Critic
from src.agent.replaybuffer import Vanilla_ReplayBuffer
from  src.data.logger_pyarrow import ArrowLogger

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    ------------------------------------------------------------
    ออกแบบให้ยืดหยุ่นสูงสำหรับงานวิจัย/ Simulation / Control system

    - รองรับ Actor แบบ simple/advanced
    - รองรับ Critic แบบ simple/advanced + encoder=True/False
    - Twin Critic + Target Critic
    - Fixed alpha
    - Replay Buffer มาตรฐาน
    - ระบบ Auto-save + Resume training
    """

    def __init__(self,
                 state_dim, action_dim,
                 min_action, max_action,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 replay_capacity=100000,
                 device='cuda',
                 logger_status=False,

                 # -------- Actor Options ----------
                 simple_layers_actor=2,
                 simple_hidden_actor=256,
                 advanced_hidden_size_actor=None,

                 # -------- Critic Options ---------
                 simple_layers_critic=2,
                 simple_hidden_critic=256,
                 advanced_hidden_sizes_critic=None,
                 critic_encoder=False,
                 logger_path=r"D:\Project_end\New_world\my_project\logs\agent"
                 ):

        # ------------------------------------------------------------
        # Device
        # ------------------------------------------------------------
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("[SACAgent] Using device:", self.device)

        # =====================================================================
        # Construct Actor
        # =====================================================================
        self.actor = Actor(
            state_dim, action_dim,
            min_action, max_action,
            simple_layers=simple_layers_actor,
            simple_hidden=simple_hidden_actor,
            advanced_hidden_sizes=advanced_hidden_size_actor
        ).to(self.device)

        # =====================================================================
        # Construct Critic (Main & Target) — Fully matching Architectures
        # =====================================================================
        self.critic = Critic(
            state_dim, action_dim,
            simple_layers=simple_layers_critic,
            simple_hidden=simple_hidden_critic,
            advanced_hidden_sizes=advanced_hidden_sizes_critic,
            use_encoder=critic_encoder
        ).to(self.device)

        self.target_critic = Critic(
            state_dim, action_dim,
            simple_layers=simple_layers_critic,
            simple_hidden=simple_hidden_critic,
            advanced_hidden_sizes=advanced_hidden_sizes_critic,
            use_encoder=critic_encoder
        ).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = Vanilla_ReplayBuffer(
            state_dim, action_dim,
            capacity=replay_capacity,
            device=self.device
        )

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Store critic config for save/load
        self.simple_layers_critic = simple_layers_critic
        self.simple_hidden_critic = simple_hidden_critic
        self.advanced_hidden_sizes_critic = advanced_hidden_sizes_critic
        self.critic_encoder = critic_encoder

        # Actor config (for save/load)
        self.simple_layers_actor = simple_layers_actor
        self.simple_hidden_actor = simple_hidden_actor
        self.advanced_hidden_size_actor = advanced_hidden_size_actor

        # Logger
        self.logger_path = logger_path
        self.logger = ArrowLogger(metric_base=self.logger_path)
        self.logger_status = logger_status
        self.action_log = None

    # ======================================================================
    # Action Selection
    # ======================================================================
    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if deterministic:
            mean, _ = self.actor.forward(state)
            y = torch.tanh(mean)
            action = self.actor.min_action + (y + 1) * 0.5 * (self.actor.max_action - self.actor.min_action)
        else:
            action, _ = self.actor.sample(state)
        self.action_log = action
        return action.cpu().detach().numpy()[0]

    # ======================================================================
    # Training
    # ======================================================================
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = [
            x.to(self.device) for x in (state, action, reward, next_state, done)
        ]

        # ------------------------------------------------------------
        # Compute target
        # ------------------------------------------------------------
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)

            q1_t, q2_t = self.target_critic(next_state, next_action)
            q_min = torch.min(q1_t, q2_t)

            target_q = reward + (1 - done) * self.gamma * (q_min - self.alpha * next_logp)

        # ------------------------------------------------------------
        # Critic update
        # ------------------------------------------------------------
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ------------------------------------------------------------
        # Actor update
        # ------------------------------------------------------------
        a_pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ------------------------------------------------------------
        # Soft update target critic
        # ------------------------------------------------------------
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        if self.logger_status:
            self.logger.log_metric("loss_actor", actor_loss.item())
            self.logger.log_metric("loss_critic", critic_loss.item())
            self.logger.log_metric("q1_mean", q1.mean().item())
            self.logger.log_metric("q2_mean", q2.mean().item())
            self.logger.log_metric("entropy", -log_pi.mean().item())
            self.logger.log_metric("alpha", self.alpha)
            self.logger.log_metric("tau", self.tau)
            self.logger.log_metric("action", self.action_log)

    # ======================================================================
    # Checkpoint: Save everything (model + optimizers + episode)
    # ======================================================================
    def save_checkpoint(self, episode, path="checkpoints/sac_checkpoint.pt"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,

            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),

            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),

            "hyperparams": {
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,

                # Action bounds
                "min_action": self.actor.min_action.cpu().tolist(),
                "max_action": self.actor.max_action.cpu().tolist(),

                # Actor architecture
                "state_dim": self.actor.state_dim,
                "action_dim": self.actor.action_dim,
                "simple_layers_actor": self.simple_layers_actor,
                "simple_hidden_actor": self.simple_hidden_actor,
                "advanced_hidden_size_actor": self.advanced_hidden_size_actor,

                # Critic architecture
                "simple_layers_critic": self.simple_layers_critic,
                "simple_hidden_critic": self.simple_hidden_critic,
                "advanced_hidden_sizes_critic": self.advanced_hidden_sizes_critic,
                "critic_encoder": self.critic_encoder,
            }
        }

        torch.save(checkpoint, path)
        print(f"[AutoSave] Saved checkpoint at episode {episode}")

    # ======================================================================
    # Load Checkpoint
    # ======================================================================
    def load_checkpoint(self, path="checkpoints/sac_checkpoint.pt"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = torch.load(path, map_location=self.device)
        hyper = data["hyperparams"]

        self.gamma = hyper["gamma"]
        self.tau = hyper["tau"]
        self.alpha = hyper["alpha"]

        state_dim = hyper["state_dim"]
        action_dim = hyper["action_dim"]

        min_action = torch.tensor(hyper["min_action"], dtype=torch.float32)
        max_action = torch.tensor(hyper["max_action"], dtype=torch.float32)

        # Actor config
        self.simple_layers_actor = hyper["simple_layers_actor"]
        self.simple_hidden_actor = hyper["simple_hidden_actor"]
        self.advanced_hidden_size_actor = hyper["advanced_hidden_size_actor"]

        # Critic config
        self.simple_layers_critic = hyper["simple_layers_critic"]
        self.simple_hidden_critic = hyper["simple_hidden_critic"]
        self.advanced_hidden_sizes_critic = hyper["advanced_hidden_sizes_critic"]
        self.critic_encoder = hyper["critic_encoder"]

        # ---- Rebuild Actor ----
        self.actor = Actor(
            state_dim, action_dim,
            min_action, max_action,
            simple_layers=self.simple_layers_actor,
            simple_hidden=self.simple_hidden_actor,
            advanced_hidden_sizes=self.advanced_hidden_size_actor
        ).to(self.device)
        self.actor.load_state_dict(data["actor"])

        # ---- Rebuild Critic ----
        self.critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.critic.load_state_dict(data["critic"])

        self.target_critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.target_critic.load_state_dict(data["target_critic"])

        # ---- Load Optimizers ----
        self.actor_opt.load_state_dict(data["actor_opt"])
        self.critic_opt.load_state_dict(data["critic_opt"])

        print(f"[Resume] Loaded checkpoint from episode {data['episode']}")
        return data["episode"]

    # ======================================================================
    # Save Model (weights only)
    # ======================================================================
    def save_model(self, path="sac_model.pt"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),

            "hyperparams": {
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,

                "min_action": self.actor.min_action.cpu().tolist(),
                "max_action": self.actor.max_action.cpu().tolist(),

                "state_dim": self.actor.state_dim,
                "action_dim": self.actor.action_dim,

                # actor
                "simple_layers_actor": self.simple_layers_actor,
                "simple_hidden_actor": self.simple_hidden_actor,
                "advanced_hidden_size_actor": self.advanced_hidden_size_actor,

                # critic
                "simple_layers_critic": self.simple_layers_critic,
                "simple_hidden_critic": self.simple_hidden_critic,
                "advanced_hidden_sizes_critic": self.advanced_hidden_sizes_critic,
                "critic_encoder": self.critic_encoder,
            }
        }

        torch.save(data, path)
        print(f"[SaveModel] Model saved → {path}")

    # ======================================================================
    # Load Model (weights only)
    # ======================================================================
    def load_model(self, path="sac_model.pt"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = torch.load(path, map_location=self.device)
        hyper = data["hyperparams"]

        # load hyper
        self.gamma = hyper["gamma"]
        self.tau = hyper["tau"]
        self.alpha = hyper["alpha"]

        state_dim = hyper["state_dim"]
        action_dim = hyper["action_dim"]

        min_action = torch.tensor(hyper["min_action"])
        max_action = torch.tensor(hyper["max_action"])

        # actor architecture
        self.simple_layers_actor = hyper["simple_layers_actor"]
        self.simple_hidden_actor = hyper["simple_hidden_actor"]
        self.advanced_hidden_size_actor = hyper["advanced_hidden_size_actor"]

        # critic architecture
        self.simple_layers_critic = hyper["simple_layers_critic"]
        self.simple_hidden_critic = hyper["simple_hidden_critic"]
        self.advanced_hidden_sizes_critic = hyper["advanced_hidden_sizes_critic"]
        self.critic_encoder = hyper["critic_encoder"]

        # recreate actor
        self.actor = Actor(
            state_dim, action_dim,
            min_action, max_action,
            simple_layers=self.simple_layers_actor,
            simple_hidden=self.simple_hidden_actor,
            advanced_hidden_sizes=self.advanced_hidden_size_actor
        ).to(self.device)
        self.actor.load_state_dict(data["actor"])

        # recreate critic
        self.critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.critic.load_state_dict(data["critic"])

        # recreate target critic
        self.target_critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.target_critic.load_state_dict(data["target_critic"])

        print(f"[LoadModel] Loaded from {path}")

    # ======================================================================
    # Logger utilities
    # ======================================================================
    def reset_logger(self):
        """
        Reset the entire logger. Useful when starting a new experiment or
        evaluation cycle.
        """
        self.logger = Logger()      # new fresh logger
        print("[Logger] Reset: created a new empty logger.")

    def clear_logger(self, key=None):
        """
        Clear specific metric or all metrics from the existing logger.
        
        Parameters
        ----------
        key : str or None
            - If None: clear all metrics
            - If str: clear only that metric
        """
        self.logger.clear(key)
        if key is None:
            print("[Logger] Cleared all metrics.")
        else:
            print(f"[Logger] Cleared metric '{key}'.")
