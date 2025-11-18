import numpy as np
import torch

class Vanilla_ReplayBuffer:
    """
    A minimal and efficient Replay Buffer for off-policy Reinforcement Learning algorithms
    such as SAC, TD3, and DDPG.

    This implementation focuses on:
        - Fast pre-allocated NumPy storage
        - Zero conversion overhead until sampling (NumPy → PyTorch happens only on sample)
        - Efficient memory usage for large buffers
        - GPU-ready tensor outputs via the `device` argument

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.
    action_dim : int
        Dimension of the action vector.
    capacity : int, optional (default=100000)
        Maximum number of transitions stored in the buffer.
        When full, old transitions will be overwritten (circular buffer).
    device : str or torch.device, optional (default='cpu')
        Device on which returned tensors will be allocated.

    Attributes
    ----------
    ptr : int
        Write pointer indicating the next index to store a transition.
    size : int
        Current number of valid entries in the buffer.
    state, action, reward, next_state, done : np.ndarray
        Pre-allocated memory arrays for each element of the transition.
    """

    def __init__(self, state_dim, action_dim, capacity=100000, device='cpu'):
        self.capacity = capacity
        self.device = device

        self.ptr = 0      # index for next write operation
        self.size = 0     # number of elements currently stored

        # ----------------------------------------------------------------------
        # Pre-allocate memory for all transitions
        # Using fixed-size NumPy arrays ensures fast writes and avoids Python lists.
        # ----------------------------------------------------------------------
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s2, d):
        """
        Store a single transition (s, a, r, s2, done).

        This operation uses a circular buffer mechanism:
        - When the buffer is not full, data expands normally.
        - After reaching capacity, new data overwrites the oldest transition.

        Parameters
        ----------
        s : array-like
            Current state.
        a : array-like
            Action applied at state `s`.
        r : float
            Reward after performing action `a`.
        s2 : array-like
            Next state resulting from the action.
        d : float or bool
            Done flag indicating episode termination (1 = done, 0 = not done).
        """
        i = self.ptr

        # Write transition
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = d

        # Move pointer forward (circular indexing)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.

        Notes
        -----
        - Sampling is uniform (no prioritization).
        - Returned objects are PyTorch tensors located on the specified device.
        - Conversion to tensors occurs only here, minimizing overhead during storage.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        state : torch.Tensor
        action : torch.Tensor
        reward : torch.Tensor
        next_state : torch.Tensor
        done : torch.Tensor
        """
        # Randomly select indices from the available range [0, size)
        idx = np.random.randint(0, self.size, size=batch_size)

        # Convert selected transitions to PyTorch tensors
        state      = torch.tensor(self.state[idx], dtype=torch.float32, device=self.device)
        action     = torch.tensor(self.action[idx], dtype=torch.float32, device=self.device)
        reward     = torch.tensor(self.reward[idx], dtype=torch.float32, device=self.device)
        next_state = torch.tensor(self.next_state[idx], dtype=torch.float32, device=self.device)
        done       = torch.tensor(self.done[idx], dtype=torch.float32, device=self.device)

        return state, action, reward, next_state, done

    def __len__(self):
        """Return the current number of transitions stored in the buffer."""
        return self.size
