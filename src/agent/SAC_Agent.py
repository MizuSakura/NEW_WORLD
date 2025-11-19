#my_project\src\agent\SAC_Agent.py
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from src.agent.replaybuffer import Vanilla_ReplayBuffer

class Logger:
    """
    Lightweight metric logger for reinforcement learning experiments.

    This class stores metrics (losses, rewards, Q-values, etc.) in simple
    Python lists, making it easy to plot curves or compute statistics later.

    Typical use cases:
        - Track SAC actor/critic losses
        - Track episodic returns
        - Track entropy values, Q-targets, value estimates
        - Debug learning behavior over time

    ----------------------------------------------------------------------
    Usage Example
    ----------------------------------------------------------------------
    >>> logger = Logger()
    >>> logger.log("loss_actor", 0.52)
    >>> logger.log("loss_critic", 1.24)
    >>> logger.log("reward", -0.3)

    Append values during training:
    >>> for step in range(1000):
    ...     logger.log("loss_actor", np.random.randn())
    ...     logger.log("loss_critic", np.random.randn())

    Retrieve data:
    >>> actor_losses = logger.get("loss_actor")
    >>> print(len(actor_losses))

    Print summary of last N entries:
    >>> logger.summary(last_n=20)

    View all metric names:
    >>> print(logger.keys())
    ----------------------------------------------------------------------
    """

    def __init__(self):
        """Initialize an empty logger dictionary."""
        self.data = {}

    def log(self, key, value):
        """
        Record a scalar value under the given metric name.

        Parameters
        ----------
        key : str
            Name of the metric (e.g., "loss_actor").
        value : float
            Numeric value to append.
        """
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def get(self, key):
        """
        Retrieve all logged values associated with a metric.

        Parameters
        ----------
        key : str
            Metric name.

        Returns
        -------
        list
            List of recorded values (possibly empty).
        """
        return self.data.get(key, [])

    def keys(self):
        """
        Return a list of all metric names that have been logged.

        Returns
        -------
        list of str
        """
        return list(self.data.keys())

    def summary(self, last_n=10):
        """
        Print the mean of the last `last_n` values for each metric.

        Useful for quick debugging during training.

        Parameters
        ----------
        last_n : int
            Number of recent entries to average.
        """
        print("=== Logger Summary ===")
        for k, v in self.data.items():
            if len(v) > 0:
                print(f"{k}: mean(last {last_n}) = {np.mean(v[-last_n:]):.5f}")


class Actor(nn.Module):
    """
    Stochastic Actor Network for Soft Actor-Critic (SAC).

    Implements a Gaussian policy followed by a Tanh squashing function to
    produce bounded continuous actions.

    This version supports per-dimension action ranges such as:
        min_action = [-1, -2, -0.5]
        max_action = [ 1,  2,  1.0]

    ----------------------------------------------------------------------
    Usage Example
    ----------------------------------------------------------------------
    >>> actor = Actor(
    ...     state_dim=3,
    ...     action_dim=2,
    ...     min_action=[-1.0, -2.0],
    ...     max_action=[ 1.0,  2.0],
    ... )
    >>> state = torch.randn(4, 3)       # batch of 4 states
    >>> action, logp = actor.sample(state)
    >>> print(action.shape)             # torch.Size([4, 2])
    >>> print(logp.shape)               # torch.Size([4, 1])

    Deterministic action (for evaluation):
    >>> action_eval = actor.sample_deterministic(state)

    Move to GPU:
    >>> actor.cuda()
    >>> state = state.cuda()
    >>> action, logp = actor.sample(state)
    ----------------------------------------------------------------------

    Parameters
    ----------
    state_dim : int
        Dimensionality of state input.
    action_dim : int
        Number of action components.
    min_action : list or array-like
        Minimum action values per dimension.
    max_action : list or array-like
        Maximum action values per dimension.

    Notes
    -----
    SAC requires:
        - sampled action (using rsample → reparameterization trick)
        - log_prob of sampled action (after tanh correction)
    to compute the entropy-regularized policy loss.
    """

    def __init__(self, state_dim, action_dim, min_action, max_action):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Convert min/max to proper shape tensors and register them as buffers
        min_action = torch.tensor(min_action, dtype=torch.float32)
        max_action = torch.tensor(max_action, dtype=torch.float32)

        assert min_action.shape == (action_dim,)
        assert max_action.shape == (action_dim,)

        self.register_buffer("min_action", min_action)
        self.register_buffer("max_action", max_action)

        # Actor neural network (outputs Gaussian mean)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # Trainable log standard deviation (per dimension)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Soft bounds for log_std to maintain stability
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        """
        Compute Gaussian mean and standard deviation (before tanh).

        Parameters
        ----------
        state : torch.Tensor
            Shape: (batch_size, state_dim)

        Returns
        -------
        mean : torch.Tensor
            Gaussian mean.
        std : torch.Tensor
            Standard deviation (exp of clamped log_std).
        """
        mean = self.net(state)

        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        """
        Sample stochastic action using reparameterization trick.

        Steps
        -----
        1. Sample x_t ~ Normal(mean, std)
        2. y_t = tanh(x_t)
        3. Scale y_t from [-1, 1] → [min_action, max_action]
        4. Compute corrected log_prob for SAC:
            log_pi = log N(x_t) - log(1 - tanh(x_t)^2)

        Returns
        -------
        action : torch.Tensor
            Scaled continuous action.
        log_prob : torch.Tensor
            Log probability of sampled action (shape: [batch, 1]).
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Reparameterize: x_t = mean + std * epsilon
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)

        # Scale action to target range
        action = self.min_action + (y_t + 1) * 0.5 * (self.max_action - self.min_action)

        # Tanh correction for log prob
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def sample_deterministic(self, state):
        """
        Deterministic action for evaluation (no randomness).

        Equivalent to:
            action = tanh(mean(state))

        Parameters
        ----------
        state : torch.Tensor

        Returns
        -------
        action : torch.Tensor
            Deterministic scaled action.
        """
        mean, _ = self.forward(state)
        y_t = torch.tanh(mean)

        action = self.min_action + (y_t + 1) * 0.5 * (self.max_action - self.min_action)
        return action

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Twin Q-Network for Soft Actor-Critic (SAC).

    This module contains two independent Q-networks (Q1 and Q2) used to
    mitigate overestimation bias during critic learning (Double Q-learning).

    Each Q-network receives:
        - state:  shape [batch, state_dim]
        - action: shape [batch, action_dim]
    and outputs:
        - q1: shape [batch, 1]
        - q2: shape [batch, 1]

    ----------------------------------------------------------------------
    Usage Example
    ----------------------------------------------------------------------
    >>> critic = Critic(state_dim=3, action_dim=2)
    >>> state  = torch.randn(4, 3)     # batch of states
    >>> action = torch.randn(4, 2)     # batch of actions
    >>> q1, q2 = critic(state, action)
    >>> print(q1.shape, q2.shape)
    torch.Size([4, 1]) torch.Size([4, 1])

    Move to GPU:
    >>> critic.cuda()
    >>> state  = state.cuda()
    >>> action = action.cuda()
    >>> q1, q2 = critic(state, action)

    Used during SAC update:
    1) next_action, logp = actor.sample(next_state)
    2) q1_target, q2_target = critic_target(next_state, next_action)
    ----------------------------------------------------------------------

    Notes
    -----
    - SAC uses min(Q1, Q2) when computing targets to reduce overestimation.
    - Networks use Xavier initialization for stable training.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # ------------------------------
        # Q1 Network (independent)
        # ------------------------------
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # ------------------------------
        # Q2 Network (independent)
        # ------------------------------
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Custom initialization (stable for RL)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Xavier initialization for linear layers.

        Improves training stability for actor–critic systems,
        especially with ReLU activations.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state, action):
        """
        Compute Q1(s,a) and Q2(s,a).

        Parameters
        ----------
        state : torch.Tensor
            Shape: [batch, state_dim]
        action : torch.Tensor
            Shape: [batch, action_dim]

        Returns
        -------
        q1 : torch.Tensor  shape [batch, 1]
        q2 : torch.Tensor  shape [batch, 1]
        """
        # Concatenate along feature dimension
        sa = torch.cat([state, action], dim=1)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    ------------------------------------------------------------
    A clean and minimal SAC implementation suitable for research,
    simulation studies, and real-world control tasks.

    Key Features
    ------------
    - Stochastic Gaussian Actor (Tanh-squashed)
    - Twin Q-Critic network (Q1, Q2) to avoid overestimation
    - Target Critic for stable Bellman backups
    - Fixed entropy coefficient α (no automatic entropy tuning)
    - Works with a standard Replay Buffer
    - Optional logging for training curves

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector.
    action_dim : int
        Dimensionality of the action vector.
    min_action, max_action : float or array-like
        Action bounds after Tanh scaling.
    lr : float
        Learning rate for both actor and critic optimizers.
    gamma : float
        Discount factor used in the target Q-value computation.
    tau : float
        Soft update coefficient for the target critic.
    alpha : float
        Entropy regularization coefficient (higher = more exploration).
    replay_capacity : int
        Maximum size of the replay buffer.
    device : str
        'cpu' or 'cuda'.
    logger_status : bool
        Whether to enable metric logging during training.

    Example
    -------
    >>> agent = SACAgent(state_dim=3, action_dim=1,
    ...                  min_action=-1, max_action=1,
    ...                  logger_status=True)
    >>> state = env.reset()
    >>> action = agent.select_action(state)
    >>> agent.replay_buffer.push(state, action, reward, next_state, done)
    >>> agent.update(batch_size=64)
    """

    def __init__(self, state_dim, action_dim, min_action, max_action, 
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 replay_capacity=100000,
                 device='cuda',
                 logger_status=False):
        
        # Device selection with CUDA fallback
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # ------------------------------------------------------------
        # Initialize Actor, Critic, and Target Critic Networks
        # Target critic starts as a direct copy of the critic.
        # ------------------------------------------------------------
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # ------------------------------------------------------------
        # Optimizers for Actor and Critic
        # ------------------------------------------------------------
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ------------------------------------------------------------
        # Replay Buffer
        # ------------------------------------------------------------
        self.replay_buffer = Vanilla_ReplayBuffer(
            state_dim, action_dim,
            capacity=replay_capacity,
            device=self.device
        )

        # ------------------------------------------------------------
        # Hyperparameters
        # ------------------------------------------------------------
        self.gamma = gamma      # discount factor
        self.tau = tau          # target network soft update rate
        self.alpha = alpha      # entropy weight (encourages exploration)

        # Logger for training curves
        self.logger = Logger()
        self.logger_status = logger_status

    # ======================================================================
    # Action Selection
    # ======================================================================
    def select_action(self, state, deterministic=False):
        """
        Select an action given the environment state.

        Parameters
        ----------
        deterministic : bool
            - True:  use the mean action (good for evaluation)
            - False: sample stochastically (recommended for training)

        Returns
        -------
        action : np.ndarray
            Scaled action in the range [min_action, max_action].
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if deterministic:
            # Use the actor mean → tanh → scale back to action bounds.
            mean, _ = self.actor.forward(state)
            y_t = torch.tanh(mean)
            action = self.actor.min_action + (y_t + 1) * 0.5 * (self.actor.max_action - self.actor.min_action)
        else:
            # Sample using reparameterization trick.
            # Produces differentiable stochastic actions.
            action, _ = self.actor.sample(state)

        return action.cpu().detach().numpy()[0]

    # ======================================================================
    # Training Step
    # ======================================================================
    def update(self, batch_size=64):
        """
        Performs one SAC update step:
        - Sample batch from replay buffer
        - Update critic using Bellman backup
        - Update actor by maximizing expected Q minus entropy
        - Soft update target critic

        This function is called repeatedly during training.
        """
        if len(self.replay_buffer) < batch_size:
            return  # Not enough samples yet

        # ------------------------------------------------------------
        # Sample a random mini-batch from the Replay Buffer
        # ------------------------------------------------------------
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = [
            x.to(self.device) for x in (state, action, reward, next_state, done)
        ]

        # ------------------------------------------------------------
        # Compute the Target Q-value (Bootstrapped)
        # ------------------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_t, q2_t = self.target_critic(next_state, next_action)

            # Twin-critic trick: use min(Q1, Q2) to reduce positive bias
            min_q_t = torch.min(q1_t, q2_t)

            # SAC target: r + γ * (minQ - α logπ)
            target_q = reward + (1 - done) * self.gamma * (min_q_t - self.alpha * next_log_prob)

        # ------------------------------------------------------------
        # Critic Update (Minimize MSE between Q and target)
        # ------------------------------------------------------------
        q1, q2 = self.critic(state, action)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ------------------------------------------------------------
        # Actor Update (Maximize expected Q - α * entropy)
        # ------------------------------------------------------------
        a_pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor minimizes (α logπ - Q) which equals maximizing Q - α logπ
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ------------------------------------------------------------
        # Soft Update for Target Critic
        # θ_target ← τ θ + (1 - τ) θ_target
        # This stabilizes learning by slowly tracking the critic.
        # ------------------------------------------------------------
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # ------------------------------------------------------------
        # Logging (losses, entropy, Q values, etc.)
        # ------------------------------------------------------------
        if self.logger_status:
            self.logger.log("loss_actor", actor_loss.item())
            self.logger.log("loss_critic", critic_loss.item())
            self.logger.log("q1_mean", q1.mean().item())
            self.logger.log("q2_mean", q2.mean().item())
            self.logger.log("entropy", -log_pi.mean().item())
            self.logger.log("alpha", self.alpha)
            self.logger.log("tau", self.tau)
    
    # ======================================================================
    # Save / Load Model (Drop-Version Safe, pathlib + auto .pt)
    # ======================================================================
        # ======================================================================
    # Auto-Save / Auto-Load Checkpoints (with episode tracking)
    # ======================================================================
    def save_checkpoint(self, episode, path="checkpoints/sac_checkpoint.pt"):
        """
        Save full training state including:
        - networks (actor, critic, target critic)
        - optimizers
        - hyperparameters
        - episode number
        """
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
                "min_action": self.actor.min_action.cpu().tolist(),
                "max_action": self.actor.max_action.cpu().tolist(),
                "state_dim": self.actor.state_dim,
                "action_dim": self.actor.action_dim
            }
        }

        torch.save(checkpoint, path)
        print(f"[AutoSave] Checkpoint saved at episode {episode} → {path}")


    def load_checkpoint(self, path="checkpoints/sac_checkpoint.pt"):
        """
        Load full training state and return the episode number
        so training can be resumed seamlessly.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[AutoSave] No checkpoint found at: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # --- Load hyperparameters ---
        hyper = checkpoint["hyperparams"]
        self.gamma = hyper["gamma"]
        self.tau = hyper["tau"]
        self.alpha = hyper["alpha"]

        min_action = torch.tensor(hyper["min_action"], dtype=torch.float32)
        max_action = torch.tensor(hyper["max_action"], dtype=torch.float32)
        state_dim  = hyper["state_dim"]
        action_dim = hyper["action_dim"]

        # --- Recreate actor (architecture must match) ---
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(self.device)
        self.actor.load_state_dict(checkpoint["actor"])

        # --- Load critic & target critic ---
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])

        # --- Load optimizers ---
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])

        ep = checkpoint["episode"]
        print(f"[AutoSave] Checkpoint loaded from episode {ep}")

        return ep
    def save_model(self, path="sac_model.pt"):
        """
        Save only model weights + hyperparameters.
        Optimizers are excluded because this file is meant for evaluation.
        """
        path = Path(path)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
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
            },
        }

        torch.save(data, path)
        print(f"[SACAgent] Model saved to '{path}'")


    # ======================================================================
    # Load Final Model (For Evaluation)
    # ======================================================================
    def load_model(self, path="sac_model.pt"):
        """
        Load model weights only (for evaluation).
        Does not load optimizer states.
        """
        path = Path(path)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")

        if not path.exists():
            raise FileNotFoundError(f"[SACAgent] File not found: '{path}'")

        data = torch.load(path, map_location=self.device)

        # basic parameters
        hyper = data["hyperparams"]
        self.gamma = hyper["gamma"]
        self.tau = hyper["tau"]
        self.alpha = hyper["alpha"]

        # load weights
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.target_critic.load_state_dict(data["target_critic"])

        print(f"[SACAgent] Model loaded from '{path}'")
