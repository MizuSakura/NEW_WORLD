#my_project\notebooks\ex07LSTM_SAC.py
"""
RSAC: Soft Actor-Critic with LSTM-based Actor and Critics
File: rsac_sac_lstm.py

Contents
- LSTMActor / LSTMCritic: recurrent networks to handle delays/partial-observability
- GaussianPolicyLSTM: actor outputs mean/log_std, sample actions with reparameterization
- SequenceReplayBuffer: stores episodes or sliding windows and samples sequences for LSTM
- SACAgentRSAC: training utilities, update steps for actor/critics/alpha, soft target update
- Checkpointing (save/load checkpoint), save/load model (weights), logger integration
- Example usage: how to initialise and train on a gym environment that returns low-dim observations

Notes:
- All comments and docstrings are in English per user's request to focus on English comments.
- Designed for clarity and extensibility rather than micro-optimised performance.
- This implementation assumes continuous action spaces (gym.spaces.Box).

Author: Generated for user request: SAC + LSTM (RSAC) with SACAgent-like logic
"""

import collections
from typing import Deque, Tuple, List
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path

# ----------------------------- Neural network modules -----------------------------

class LSTMEncoder(nn.Module):
    """
    Shared small LSTM encoder that maps a sequence of observations to a hidden vector.
    Used to keep actor/critic architectures concise and reusable.
    """
    def __init__(self, obs_dim: int, hidden_dim: int, lstm_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Project observation to a suitable size before feeding to LSTM
        self.fc_in = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        # Optional layer normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs_seq: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        obs_seq: (batch, seq_len, obs_dim)
        returns:
            output_seq: (batch, seq_len, hidden_dim)
            (h_n, c_n): final LSTM hidden states
        """
        # apply linear projection per timestep
        x = self.fc_in(obs_seq)
        x = F.relu(x)
        out_seq, (h_n, c_n) = self.lstm(x, hx)
        # normalize per timestep
        out_seq = self.layer_norm(out_seq)
        return out_seq, (h_n, c_n)


class LSTMActor(nn.Module):
    """
    Recurrent actor that reads a sequence and returns distribution parameters for the last step.
    The actor outputs mean and log_std for a Gaussian policy.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, lstm_layers: int = 1, action_scale: float = 1.0, action_bias: float = 0.0, simple_layers: int = 2, simple_hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.action_scale = action_scale
        self.action_bias = action_bias

        # store architecture config for save/load compatibility
        self.simple_layers = simple_layers
        self.simple_hidden = simple_hidden

        self.encoder = LSTMEncoder(obs_dim, hidden_dim, lstm_layers)
        # Map the last timestep's hidden vector to mean and log_std
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, simple_hidden),
            nn.ReLU(),
            nn.Linear(simple_hidden, action_dim)
        )
        self.logstd_head = nn.Sequential(
            nn.Linear(hidden_dim, simple_hidden),
            nn.ReLU(),
            nn.Linear(simple_hidden, action_dim)
        )
        # clamp logstd for numerical stability
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, obs_seq: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        obs_seq: (batch, seq_len, obs_dim)
        returns mean, log_std, (h_n, c_n)
        where mean/log_std correspond to the last timestep in the sequence
        """
        out_seq, (h_n, c_n) = self.encoder(obs_seq, hx)
        last = out_seq[:, -1, :]
        mean = self.mean_head(last)
        log_std = self.logstd_head(last)
        log_std = torch.tanh(log_std)  # keep in a reasonable range then scale
        # linear rescale to desired bounds
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std, (h_n, c_n)

    def sample(self, obs_seq: torch.Tensor, deterministic: bool = False):
        """
        Sample action using reparameterization trick.
        Returns action in environment scale and log_prob.
        """
        mean, log_std, _ = self.forward(obs_seq)
        std = log_std.exp()
        if deterministic:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            log_prob = None
            return action, log_prob

        # Reparameterization: sample N(0,1), then transform
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # rsample for gradient flow
        # apply tanh squashing to bound actions to (-1,1)
        action = torch.tanh(z)
        # compute log_prob with correction for tanh
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = action * self.action_scale + self.action_bias
        return action, log_prob


class LSTMCritic(nn.Module):
    """
    Recurrent critic that takes sequences and actions and estimates Q-values for the last timestep.
    We implement two critics (twin Q) to reduce overestimation bias.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, lstm_layers: int = 1, simple_layers: int = 2, simple_hidden: int = 256, use_encoder: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.use_encoder = use_encoder

        # architecture flags for save/load
        self.simple_layers = simple_layers
        self.simple_hidden = simple_hidden

        self.encoder = LSTMEncoder(obs_dim, hidden_dim, lstm_layers) if use_encoder else None
        # After getting last hidden, concatenate action and estimate Q
        in_dim = hidden_dim if use_encoder else obs_dim
        self.q_head = nn.Sequential(
            nn.Linear(in_dim + action_dim, simple_hidden),
            nn.ReLU(),
            nn.Linear(simple_hidden, 1)
        )

    def forward(self, obs_seq: torch.Tensor, action: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        obs_seq: (batch, seq_len, obs_dim)
        action: (batch, action_dim) corresponding to last timestep's action
        returns Q-value for the last timestep
        """
        if self.use_encoder:
            out_seq, (h_n, c_n) = self.encoder(obs_seq, hx)
            last = out_seq[:, -1, :]
        else:
            # if no encoder: use last observation directly
            last = obs_seq[:, -1, :]
        x = torch.cat([last, action], dim=-1)
        q = self.q_head(x)
        return q


# ----------------------------- Replay Buffer -----------------------------

class SequenceReplayBuffer:
    """
    Replay buffer that stores transitions and allows sampling of contiguous sequences.

    Storage format: we store individual transitions but sample sequences of fixed length
    which are then collated into tensors of shape (batch, seq_len, dim).

    Important for LSTM training: sampled sequences should preserve temporal order.
    """
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 1000000, seq_len: int = 8, device: str = 'cpu'):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device

        # ring buffers for scalar arrays
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition to the ring buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _valid_start_indices(self):
        """Return indices that can be used as start of a seq_len-long contiguous window."""
        # Only indices where we have seq_len contiguous elements without wrap-around
        if self.size < self.seq_len:
            return np.array([], dtype=np.int32)

        # make a range of valid starts considering current pointer and filled region
        max_index = self.size - self.seq_len
        # valid starts are 0..max_index-1 but must be offset to the ring buffer start
        # compute start offset (oldest data index)
        start_idx = (self.ptr - self.size) % self.capacity
        indices = (start_idx + np.arange(0, max_index + 1)) % self.capacity
        return indices

    def sample_batch(self, batch_size: int):
        """
        Sample a batch of sequences.
        Returns tensors: obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq
        obs_seq shape: (batch, seq_len, obs_dim)
        actions_seq shape: (batch, seq_len, action_dim)
        rewards_seq shape: (batch, seq_len, 1)  # reward aligned with each timestep
        next_obs_seq corresponds to next observations for each timestep
        """
        indices = self._valid_start_indices()
        if len(indices) == 0:
            raise ValueError('Not enough data to sample a sequence')

        chosen = np.random.choice(indices, size=batch_size, replace=True)

        obs_seq = np.zeros((batch_size, self.seq_len, self.obs.shape[1]), dtype=np.float32)
        actions_seq = np.zeros((batch_size, self.seq_len, self.actions.shape[1]), dtype=np.float32)
        rewards_seq = np.zeros((batch_size, self.seq_len, 1), dtype=np.float32)
        next_obs_seq = np.zeros((batch_size, self.seq_len, self.obs.shape[1]), dtype=np.float32)
        dones_seq = np.zeros((batch_size, self.seq_len, 1), dtype=np.float32)

        for i, start in enumerate(chosen):
            for t in range(self.seq_len):
                idx = (start + t) % self.capacity
                obs_seq[i, t] = self.obs[idx]
                actions_seq[i, t] = self.actions[idx]
                rewards_seq[i, t] = self.rewards[idx]
                next_obs_seq[i, t] = self.next_obs[idx]
                dones_seq[i, t] = self.dones[idx]

        # convert to torch tensors on device
        return (
            torch.tensor(obs_seq, device=self.device),
            torch.tensor(actions_seq, device=self.device),
            torch.tensor(rewards_seq, device=self.device),
            torch.tensor(next_obs_seq, device=self.device),
            torch.tensor(dones_seq, device=self.device)
        )


# ----------------------------- SAC Agent with LSTM -----------------------------

class SACAgentRSAC:
    """
    Soft Actor-Critic agent where Actor and Critics are LSTM-based.

    Key details:
    - The replay buffer samples sequences which the LSTM encoder consumes.
    - Critics evaluate Q for the last timestep of the sequence using the last hidden output.
    - The policy (actor) samples actions for the last timestep based on the sequence.
    - Target critics are maintained as moving averages for stability.
    - Save/Load checkpoint and model logic added to match project's SACAgent style.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space,  # gym.spaces.Box expected for scaling
        device: str = None,
        hidden_dim: int = 256,
        lstm_layers: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: float = None,
        lr: float = 3e-4,
        seq_len: int = 8,
        buffer_capacity: int = 200000,
        logger_status: bool = False,
        # simple architecture knobs for compatibility with original project
        simple_layers_actor: int = 2,
        simple_hidden_actor: int = 256,
        simple_layers_critic: int = 2,
        simple_hidden_critic: int = 256,
        critic_encoder: bool = True
    ):
        self.device = torch.device(device if device is not None and torch.cuda.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"[RSAC] Using device: {self.device}")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        # scale and bias to map tanh outputs to action_space
        action_low = float(action_space.low[0])
        action_high = float(action_space.high[0])
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        # store architecture config for save/load
        self.simple_layers_actor = simple_layers_actor
        self.simple_hidden_actor = simple_hidden_actor
        self.simple_layers_critic = simple_layers_critic
        self.simple_hidden_critic = simple_hidden_critic
        self.critic_encoder = critic_encoder

        # networks
        self.actor = LSTMActor(obs_dim, action_dim, hidden_dim, lstm_layers, action_scale, action_bias, simple_layers_actor, simple_hidden_actor).to(self.device)
        # Twin critics
        self.critic1 = LSTMCritic(obs_dim, action_dim, hidden_dim, lstm_layers, simple_layers=simple_layers_critic, simple_hidden=simple_hidden_critic, use_encoder=critic_encoder).to(self.device)
        self.critic2 = LSTMCritic(obs_dim, action_dim, hidden_dim, lstm_layers, simple_layers=simple_layers_critic, simple_hidden=simple_hidden_critic, use_encoder=critic_encoder).to(self.device)
        # Targets
        self.critic1_target = LSTMCritic(obs_dim, action_dim, hidden_dim, lstm_layers, simple_layers=simple_layers_critic, simple_hidden=simple_hidden_critic, use_encoder=critic_encoder).to(self.device)
        self.critic2_target = LSTMCritic(obs_dim, action_dim, hidden_dim, lstm_layers, simple_layers=simple_layers_critic, simple_hidden=simple_hidden_critic, use_encoder=critic_encoder).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = Adam(self.critic2.parameters(), lr=lr)

        # entropy temperature
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=lr)
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        # replay buffer
        self.replay = SequenceReplayBuffer(obs_dim, action_dim, capacity=buffer_capacity, seq_len=seq_len, device=str(self.device))

        # algorithm hyperparams
        self.gamma = gamma
        self.tau = tau

        # logger
        self.logger_status = logger_status
        try:
            from src.agent.network import Logger
            self.logger = Logger()
        except Exception:
            # fallback simple logger
            class _SimpleLogger:
                def __init__(self):
                    self.store = {}
                def log(self, k, v):
                    self.store[k] = v
            self.logger = _SimpleLogger()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ======================================================================
    # Action Selection
    # ======================================================================
    def select_action(self, obs_seq: np.ndarray, deterministic: bool = False):
        """
        Select an action given a numpy obs sequence of shape (seq_len, obs_dim) or (1, seq_len, obs_dim).
        Supports deterministic option matching original SACAgent behaviour.
        """
        self.actor.eval()
        # ensure shape (1, seq_len, obs_dim)
        if obs_seq.ndim == 2:
            obs_seq = obs_seq[np.newaxis, ...]
        obs_t = torch.tensor(obs_seq.astype(np.float32), device=self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(obs_t)
                y = torch.tanh(mean)
                action = y * self.actor.action_scale + self.actor.action_bias
                logp = None
            else:
                action, logp = self.actor.sample(obs_t, deterministic=False)
        self.actor.train()
        return action.cpu().numpy()[0]

    # ======================================================================
    # Training
    # ======================================================================
    def update(self, batch_size: int = 64):
        """
        Perform a single gradient update step using a batch of sequences sampled from buffer.
        We compute targets using target networks and use the last timestep for Bellman update.
        """
        if self.replay.size < max(batch_size, self.seq_len):
            return

        obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq = self.replay.sample_batch(batch_size)

        # last-step aligned values
        action_last = actions_seq[:, -1, :]
        reward_last = rewards_seq[:, -1, :]
        done_last = dones_seq[:, -1:]

        # ------------------ compute target Q ------------------
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs_seq)
            q1_next = self.critic1_target(next_obs_seq, next_action)
            q2_next = self.critic2_target(next_obs_seq, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward_last + (1 - done_last) * self.gamma * (q_next - self.alpha.detach() * next_logp)

        # ------------------ update critics ------------------
        q1 = self.critic1(obs_seq, action_last)
        q2 = self.critic2(obs_seq, action_last)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # ------------------ update actor ------------------
        action_new, logp_new = self.actor.sample(obs_seq)
        q1_new = self.critic1(obs_seq, action_new)
        q2_new = self.critic2(obs_seq, action_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * logp_new - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ------------------ update alpha ------------------
        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ------------------ soft update targets ------------------
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ------------------ logging ------------------
        if self.logger_status:
            try:
                self.logger.log('loss_actor', actor_loss.item())
                self.logger.log('loss_critic1', critic1_loss.item())
                self.logger.log('loss_critic2', critic2_loss.item())
                self.logger.log('alpha', self.alpha.item())
            except Exception:
                pass

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item()
        }

    # ======================================================================
    # Checkpoint: Save everything (model + optimizers + episode)
    # ======================================================================
    def save_checkpoint(self, episode, path="checkpoints/rsac_checkpoint.pt"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,

            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),

            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),

            "log_alpha": self.log_alpha.detach().cpu().numpy().tolist(),

            "hyperparams": {
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": float(self.alpha.detach().cpu().numpy().tolist()),

                # Action bounds
                "action_low": float(-self.actor.action_scale + self.actor.action_bias) if hasattr(self.actor, 'action_scale') else None,
                "action_high": float(self.actor.action_scale + self.actor.action_bias) if hasattr(self.actor, 'action_scale') else None,

                # Architecture
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "seq_len": self.seq_len,
                "simple_layers_actor": self.simple_layers_actor,
                "simple_hidden_actor": self.simple_hidden_actor,
                "simple_layers_critic": self.simple_layers_critic,
                "simple_hidden_critic": self.simple_hidden_critic,
                "critic_encoder": self.critic_encoder
            }
        }

        torch.save(checkpoint, path)
        print(f"[RSAC AutoSave] Saved checkpoint at episode {episode} -> {path}")

    # ======================================================================
    # Load Checkpoint
    # ======================================================================
    def load_checkpoint(self, path="checkpoints/rsac_checkpoint.pt"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = torch.load(path, map_location=self.device)
        hyper = data["hyperparams"]

        self.gamma = hyper["gamma"]
        self.tau = hyper["tau"]
        # alpha loaded below

        obs_dim = hyper["obs_dim"]
        action_dim = hyper["action_dim"]

        # restore architecture configs
        self.simple_layers_actor = hyper["simple_layers_actor"]
        self.simple_hidden_actor = hyper["simple_hidden_actor"]
        self.simple_layers_critic = hyper["simple_layers_critic"]
        self.simple_hidden_critic = hyper["simple_hidden_critic"]
        self.critic_encoder = hyper.get("critic_encoder", True)

        # ---- Rebuild networks ----
        self.actor = LSTMActor(obs_dim, action_dim, self.actor.hidden_dim, self.actor.lstm_layers, self.actor.action_scale, self.actor.action_bias, self.simple_layers_actor, self.simple_hidden_actor).to(self.device)
        self.actor.load_state_dict(data["actor"])

        self.critic1 = LSTMCritic(obs_dim, action_dim, self.critic1.hidden_dim, self.critic1.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic1.load_state_dict(data["critic1"])
        self.critic2 = LSTMCritic(obs_dim, action_dim, self.critic2.hidden_dim, self.critic2.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic2.load_state_dict(data["critic2"])

        self.critic1_target = LSTMCritic(obs_dim, action_dim, self.critic1.hidden_dim, self.critic1.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic1_target.load_state_dict(data["critic1_target"])
        self.critic2_target = LSTMCritic(obs_dim, action_dim, self.critic2.hidden_dim, self.critic2.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic2_target.load_state_dict(data["critic2_target"])

        # ---- Optimizers ----
        self.actor_opt = Adam(self.actor.parameters())
        self.critic1_opt = Adam(self.critic1.parameters())
        self.critic2_opt = Adam(self.critic2.parameters())

        self.actor_opt.load_state_dict(data["actor_opt"])
        self.critic1_opt.load_state_dict(data["critic1_opt"])
        self.critic2_opt.load_state_dict(data["critic2_opt"])

        # load alpha
        self.log_alpha = torch.tensor(data.get("log_alpha", np.log(0.2)), requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=3e-4)

        print(f"[RSAC Resume] Loaded checkpoint from {path} (episode {data.get('episode', '?')})")
        return data.get('episode', 0)

    # ======================================================================
    # Save Model (weights only)
    # ======================================================================
    def save_model(self, path="rsac_model.pt"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),

            "hyperparams": {
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": float(self.alpha.detach().cpu().numpy().tolist()),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "seq_len": self.seq_len,
                "simple_layers_actor": self.simple_layers_actor,
                "simple_hidden_actor": self.simple_hidden_actor,
                "simple_layers_critic": self.simple_layers_critic,
                "simple_hidden_critic": self.simple_hidden_critic,
                "critic_encoder": self.critic_encoder
            }
        }

        torch.save(data, path)
        print(f"[RSAC SaveModel] Model saved -> {path}")

    # ======================================================================
    # Load Model (weights only)
    # ======================================================================
    def load_model(self, path="rsac_model.pt"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = torch.load(path, map_location=self.device)
        hyper = data["hyperparams"]

        # load hyper
        self.gamma = hyper["gamma"]
        self.tau = hyper["tau"]
        # alpha set from file
        alpha_val = hyper.get("alpha", 0.2)
        self.log_alpha = torch.tensor(np.log(alpha_val), requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=3e-4)

        obs_dim = hyper["obs_dim"]
        action_dim = hyper["action_dim"]

        # recreate actor
        self.actor = LSTMActor(obs_dim, action_dim, self.actor.hidden_dim, self.actor.lstm_layers, self.actor.action_scale, self.actor.action_bias, self.simple_layers_actor, self.simple_hidden_actor).to(self.device)
        self.actor.load_state_dict(data["actor"])

        # recreate critics
        self.critic1 = LSTMCritic(obs_dim, action_dim, self.critic1.hidden_dim, self.critic1.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic1.load_state_dict(data["critic1"])
        self.critic2 = LSTMCritic(obs_dim, action_dim, self.critic2.hidden_dim, self.critic2.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic2.load_state_dict(data["critic2"])

        # recreate targets
        self.critic1_target = LSTMCritic(obs_dim, action_dim, self.critic1.hidden_dim, self.critic1.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic1_target.load_state_dict(data["critic1_target"])
        self.critic2_target = LSTMCritic(obs_dim, action_dim, self.critic2.hidden_dim, self.critic2.lstm_layers, simple_layers=self.simple_layers_critic, simple_hidden=self.simple_hidden_critic, use_encoder=self.critic_encoder).to(self.device)
        self.critic2_target.load_state_dict(data["critic2_target"])

        print(f"[RSAC LoadModel] Loaded model from {path}")


# ----------------------------- Example training stub -----------------------------

if __name__ == '__main__':
    """
    Example: how to use SACAgentRSAC with a Gym environment.
    This stub demonstrates:
      - building sequences for actor input
      - storing transitions into SequenceReplayBuffer
      - calling update() periodically

    Important: environments providing low-dimensional observations (not pixels) are expected.
    For pixel observations, a convolutional encoder should be added before the LSTM.

    """
    import gymnasium as gym
    from collections import deque

    env_name = 'MountainCarContinuous-v0'  # use a simple continuous control env as a demo
    env = gym.make(env_name, render_mode="human")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgentRSAC(obs_dim, action_dim, env.action_space, device='cuda', seq_len=8)

    num_episodes = 200
    max_steps = 300
    batch_size = 64
    checkpoint_path = Path(r"D:\Project_end\New_world\my_project\models\sac_checkpoint.pt")
    autosave_every = 10

    if checkpoint_path.exists():
        print("\n[Trainer] Found checkpoint. Loading...")
        start_episode = agent.load_checkpoint(checkpoint_path) + 1
        print(f"[Trainer] Resuming training from episode {start_episode}\n")
    else:
        print("\n[Trainer] No checkpoint found. Starting from episode 1\n")

    # a small rolling buffer that we will use to form sequences for online interaction
    rolling: Deque = deque(maxlen=agent.seq_len)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        rolling.clear()
        # prefill rolling buffer with initial observation repeated
        for _ in range(agent.seq_len):
            rolling.append(obs)

        ep_reward = 0.0
        for step in range(max_steps):
            obs_seq = np.array(rolling)  # shape (seq_len, obs_dim)
            action = agent.select_action(obs_seq, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # store transition: we store single transitions; replay buffer will assemble sequences
            agent.replay.add(obs, action, reward, next_obs, done)

            # step rolling window
            rolling.append(next_obs)
            obs = next_obs
            ep_reward += reward

            # update agent after each step (or every N steps)
            stats = agent.update(batch_size)

            if done:
                break
        if ep % autosave_every == 0:
            agent.save_checkpoint(ep, checkpoint_path)

        print(f'Episode {ep:03d}| reward: {ep_reward:.2f} | status train:{terminated}')

    print('Training finished')
