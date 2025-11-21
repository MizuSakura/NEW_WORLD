#my_project\src\agent\network.py
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    """
    Actor Network for Soft Actor-Critic (SAC).
    -------------------------------------------------------
    Supports two configuration modes for network architecture:

    1) Simple Mode:
        - simple_layers: number of hidden layers
        - simple_hidden: number of neurons per layer (same for all)

        Example:
            simple_layers=3, simple_hidden=256
            -> [256, 256, 256]

    2) Advanced Mode:
        - advanced_hidden_sizes: list of integers defining each layer size

        Example:
            advanced_hidden_sizes = [128, 256, 128, 64]

    The network outputs:
        - mean (μ) of Gaussian policy
        - std (σ) computed from trainable log_std
    Followed by:
        - Tanh squashing
        - Action scaling to [min_action, max_action]
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        simple_layers=2,            # number of layers in simple mode
        simple_hidden=256,          # neurons per layer (simple mode)
        advanced_hidden_sizes=None, # list for advanced mode
        activation=nn.ReLU,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.simple_layers = simple_layers
        self.simple_hidden = simple_hidden
        self.advanced_hidden_sizes = advanced_hidden_sizes

        if self.simple_layers < 1:
            self.simple_layers = 1

        # -------------------------------------------------------------
        # Register action bounds as buffers (saved with model, no grad)
        # -------------------------------------------------------------
        min_action = torch.tensor(min_action, dtype=torch.float32)
        max_action = torch.tensor(max_action, dtype=torch.float32)

        assert min_action.shape == (action_dim,), "min_action size mismatch"
        assert max_action.shape == (action_dim,), "max_action size mismatch"

        self.register_buffer("min_action", min_action)
        self.register_buffer("max_action", max_action)

        # -------------------------------------------------------------
        # Choose network structure: Simple vs Advanced mode
        # -------------------------------------------------------------
        if advanced_hidden_sizes is not None:
            # Developer-defined architecture
            hidden_sizes = advanced_hidden_sizes
        else:
            # Auto-generate uniform architecture
            
            hidden_sizes = [simple_hidden] * self.simple_layers

        # -------------------------------------------------------------
        # Build MLP dynamically based on chosen architecture
        # -------------------------------------------------------------
        layers = []
        last_dim = state_dim

        for hs in hidden_sizes:
            layers.append(nn.Linear(last_dim, hs))
            layers.append(activation())  # activation after each linear layer
            last_dim = hs

        # Final output layer (Gaussian mean)
        layers.append(nn.Linear(last_dim, action_dim))

        # Sequential network
        self.net = nn.Sequential(*layers)

        # -------------------------------------------------------------
        # Trainable log standard deviation (per action dimension)
        # -------------------------------------------------------------
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Reason: too large → unstable; too small → no exploration
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    # -----------------------------------------------------------------
    # Forward pass: returns mean and std of Gaussian policy
    # -----------------------------------------------------------------
    def forward(self, state):
        """
        Compute mean and std (via clamped log_std) for the Gaussian policy.

        Inputs
        ------
        state : Tensor (batch_size, state_dim)

        Returns
        -------
        mean : Tensor
        std : Tensor
        """
        mean = self.net(state)

        # Clamp log_std to avoid exploding or vanishing std
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        return mean, std

    # -----------------------------------------------------------------
    # Sample stochastic action (rsample for reparameterization trick)
    # -----------------------------------------------------------------
    def sample(self, state):
        """
        Sample a stochastic action using SAC's reparameterization trick.

        Procedure:
            x ~ Normal(mean, std)
            y = tanh(x)
            action = scale_to_range(y)

            log_prob = log π(a|s) with tanh-correction

        Returns
        -------
        action : Tensor (scaled continuous action)
        log_prob : Tensor (entropy term for SAC loss)
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Reparameterized sample: x = μ + σ * ε
        x_t = dist.rsample()

        # Squash to [-1, 1]
        y_t = torch.tanh(x_t)

        # Scale from [-1,1] → [min_action, max_action]
        action = self.min_action + (y_t + 1) * 0.5 * (self.max_action - self.min_action)

        # Compute log probability with tanh correction
        log_prob = dist.log_prob(x_t)  # Gaussian log prob
        # Tanh correction term: log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    # -----------------------------------------------------------------
    # Deterministic action (evaluation mode)
    # -----------------------------------------------------------------
    def sample_deterministic(self, state):
        """
        Compute deterministic action (used during evaluation).

        action = tanh(mean(state))

        Returns
        -------
        action : Tensor
        """
        mean, _ = self.forward(state)
        y_t = torch.tanh(mean)

        action = self.min_action + (y_t + 1) * 0.5 * (self.max_action - self.min_action)
        return action


class Critic(nn.Module):
    """
    Twin Q-Network for Soft Actor-Critic (SAC)
    ---------------------------------------------------------------------------
    Supports two configurable features:

    1) Optional Encoder (use_encoder=True)
        - Shared between Q1 and Q2
        - Processes concatenated [state, action] into a compact feature
        - Helps when state-action input is large or nonlinear

    2) Simple / Advanced Architecture Modes
        - Simple mode:
              simple_layers = number of hidden layers
              simple_hidden = neurons per layer
              Example: simple_layers=3, simple_hidden=256
                       -> [256, 256, 256]
        - Advanced mode:
              advanced_hidden_sizes = list specifying exact hidden sizes
              Example: [128, 256, 128]

    Output:
        - Q1(s,a) : Tensor [batch, 1]
        - Q2(s,a) : Tensor [batch, 1]

    Notes:
        - Using two Q-networks reduces Q-value overestimation (Double Q-learning)
        - Xavier initialization improves training stability
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        use_encoder=False,         # enable/disable encoder module
        encoder_hidden=256,        # encoder output size
        simple_layers=2,
        simple_hidden=256,
        advanced_hidden_sizes=None,
        activation=nn.ReLU,
    ):
        super().__init__()

        self.use_encoder = use_encoder
        act = activation  # activation function constructor

        # ----------------------------------------------------------------------
        # Encoder Network (optional)
        # ----------------------------------------------------------------------
        # Input always starts as concatenation of [state, action]
        input_dim = state_dim + action_dim

        if use_encoder:
            # Shared encoder before feeding into Q1/Q2
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoder_hidden),
                act(),
            )
            encoder_output_dim = encoder_hidden
        else:
            self.encoder = None
            encoder_output_dim = input_dim

        # ----------------------------------------------------------------------
        # Determine hidden sizes for Q networks
        # ----------------------------------------------------------------------
        if advanced_hidden_sizes is not None:
            # Developer-defined architecture (explicit list)
            hidden_sizes = advanced_hidden_sizes
        else:
            # Auto-generated uniform architecture (simple mode)
            if simple_layers < 1:
                simple_layers = 1
            hidden_sizes = [simple_hidden] * simple_layers

        # ----------------------------------------------------------------------
        # Helper: build a Q-network dynamically
        # ----------------------------------------------------------------------
        def build_q_network():
            layers = []
            last_dim = encoder_output_dim  # Q network input dim

            # Add MLP hidden layers
            for hs in hidden_sizes:
                layers.append(nn.Linear(last_dim, hs))
                layers.append(act())
                last_dim = hs

            # Final layer outputs one Q-value
            layers.append(nn.Linear(last_dim, 1))

            return nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # Twin Q Networks (Q1 & Q2)
        # ----------------------------------------------------------------------
        self.q1_net = build_q_network()
        self.q2_net = build_q_network()

        # Initialize layers (improves learning stability)
        self.apply(self._init_weights)

    # --------------------------------------------------------------------------
    # Xavier uniform initialization for linear layers
    # --------------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------------
    # Forward pass for computing Q1(s,a) and Q2(s,a)
    # --------------------------------------------------------------------------
    def forward(self, state, action):
        """
        Compute both Q-values for SAC.

        Inputs
        ------
        state : Tensor [batch, state_dim]
        action : Tensor [batch, action_dim]

        Returns
        -------
        q1 : Tensor [batch, 1]
        q2 : Tensor [batch, 1]
        """

        # Concatenate state and action → original input
        sa = torch.cat([state, action], dim=1)

        # Optional shared encoder
        if self.use_encoder:
            h = self.encoder(sa)
        else:
            h = sa

        # Pass features through Q1 and Q2
        q1 = self.q1_net(h)
        q2 = self.q2_net(h)

        return q1, q2