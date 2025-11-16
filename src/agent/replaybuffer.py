import numpy as np
import torch

class Vanailla_ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s2, d):
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = d

        # update pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        # Convert only at sampling time → ลด overhead
        return (
            torch.tensor(self.state[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.action[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.reward[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.done[idx], dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return self.size
