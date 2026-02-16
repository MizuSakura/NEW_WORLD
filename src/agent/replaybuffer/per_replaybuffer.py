#my_project\src\agent\replaybuffer\per_replaybuffer.py
import numpy as np
from src.agent.replaybuffer.sumtree import SumTree


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    ------------------------------------
    - SumTree for O(log N) sampling
    - Importance Sampling correction
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        eps: float = 1e-6,
        device="cpu",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.device = device

        # SumTree
        self.tree = SumTree(capacity)

        # storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.max_priority = 1.0

    # --------------------------------------------------
    def __len__(self):
        return self.tree.size

    # --------------------------------------------------
    def push(self, state, action, reward, next_state, done):
        """
        Store transition with max priority (optimistic initialization)
        """
        idx = self.tree.data_pointer

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        priority = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(priority)

    # --------------------------------------------------
    def sample(self, batch_size: int):
        """
        PER-safe sampling
        """
        total_p = self.tree.total_priority

        # ---------- CRITICAL SAFETY GUARD ----------
        if not np.isfinite(total_p) or total_p <= 0:
            raise RuntimeError(
                "[PrioritizedReplayBuffer] Invalid total priority detected"
            )

        batch_idx = np.zeros(batch_size, dtype=np.int32)
        tree_idx = np.zeros(batch_size, dtype=np.int32)
        IS_weights = np.zeros(batch_size, dtype=np.float32)

        segment = total_p / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        # ---- min probability guard ----
        leaf_priorities = self.tree.tree[
            -self.tree.capacity : self.tree.capacity - 1 + self.tree.size
        ]
        min_p = max(np.min(leaf_priorities), self.eps)
        min_prob = min_p / total_p

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)

            leaf_idx, priority, data_idx = self.tree.get_leaf(s)

            prob = max(priority / total_p, self.eps)
            IS_weights[i] = (prob / min_prob) ** (-self.beta)

            batch_idx[i] = data_idx
            tree_idx[i] = leaf_idx

        # ---- normalize IS weights safely ----
        max_w = np.max(IS_weights)
        if not np.isfinite(max_w) or max_w <= 0:
            IS_weights[:] = 1.0
        else:
            IS_weights /= max_w

        return (
            self.states[batch_idx],
            self.actions[batch_idx],
            self.rewards[batch_idx],
            self.next_states[batch_idx],
            self.dones[batch_idx],
            tree_idx,
            IS_weights.reshape(-1, 1),
        )

    # --------------------------------------------------
    def update_priorities(self, tree_indices, td_errors):
        """
        Update priorities using TD-error (safe)
        """
        td_errors = np.asarray(td_errors)
        td_errors = np.abs(td_errors)
        td_errors = np.clip(td_errors, self.eps, 1e6)
        td_errors[~np.isfinite(td_errors)] = self.eps

        for idx, td in zip(tree_indices, td_errors):
            priority = td ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
