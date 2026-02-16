# src/agent/replaybuffer/n_step.py

import numpy as np
import torch
from src.agent.replaybuffer.base import BaseReplayBuffer


class NStepReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        state_dim,
        action_dim,
        capacity=100000,
        n_step=3,
        gamma=0.99,
        device="cpu"
    ):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma

        # main replay buffer
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        # ---------- N-step ring buffer ----------
        self.ns_state = [None] * n_step
        self.ns_action = [None] * n_step
        self.ns_reward = np.zeros(n_step, dtype=np.float32)
        self.ns_next_state = [None] * n_step
        self.ns_done = np.zeros(n_step, dtype=np.bool_)

        self.ns_idx = 0
        self.ns_count = 0

    # --------------------------------------------------
    def _compute_n_step_return(self):
        R = 0.0
        for i in range(self.ns_count):
            R += (self.gamma ** i) * self.ns_reward[i]
            if self.ns_done[i]:
                break

        return R, self.ns_next_state[self.ns_count - 1], self.ns_done[self.ns_count - 1]

    # --------------------------------------------------
    def push(self, state, action, reward, next_state, done):
        # write to ring buffer
        self.ns_state[self.ns_idx] = state
        self.ns_action[self.ns_idx] = action
        self.ns_reward[self.ns_idx] = reward
        self.ns_next_state[self.ns_idx] = next_state
        self.ns_done[self.ns_idx] = done

        self.ns_idx = (self.ns_idx + 1) % self.n_step
        self.ns_count = min(self.ns_count + 1, self.n_step)

        # not enough steps yet
        if self.ns_count < self.n_step:
            return

        # build n-step transition (oldest element)
        idx0 = self.ns_idx  # oldest
        s0 = self.ns_state[idx0]
        a0 = self.ns_action[idx0]

        R_n, s_n, done_n = self._compute_n_step_return()

        self.state[self.ptr] = s0
        self.action[self.ptr] = a0
        self.reward[self.ptr] = R_n
        self.next_state[self.ptr] = s_n
        self.done[self.ptr] = done_n

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.ns_count = 0  # flush

    # --------------------------------------------------
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.state[idxs], device=self.device),
            torch.tensor(self.action[idxs], device=self.device),
            torch.tensor(self.reward[idxs], device=self.device),
            torch.tensor(self.next_state[idxs], device=self.device),
            torch.tensor(self.done[idxs], device=self.device),
        )

    def __len__(self):
        return self.size
