# src/agent/replaybuffer/manager.py


from src.agent.replaybuffer.vanilla_replaybuffer import VanillaReplayBuffer
from src.agent.replaybuffer.n_step_replaybuffer import NStepReplayBuffer
from src.agent.replaybuffer.per_replaybuffer import PrioritizedReplayBuffer
from src.agent.replaybuffer.n_step_per_replaybuffer import NStepPERReplayBuffer

class ReplayBufferManager:
    def __init__(
    self,
    buffer_type: str,
    state_dim: int,
    action_dim: int,
    capacity: int = 100000,
    device="cpu",
    **kwargs
    ):
        self.buffer_type = buffer_type.lower()


        if self.buffer_type == "vanilla":
            self.buffer = VanillaReplayBuffer(
            state_dim, action_dim, capacity, device
            )


        elif self.buffer_type == "nstep":
            self.buffer = NStepReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=capacity,
            n_step=kwargs.get("n_step", 3),
            gamma=kwargs.get("gamma", 0.99),
            device=device
            )


        elif self.buffer_type == "per":
            self.buffer = PrioritizedReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=capacity,
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4),
            device=device
            )


        elif self.buffer_type == "nstep_per":
            self.buffer = NStepPERReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=capacity,
            n_step=kwargs.get("n_step", 3),
            gamma=kwargs.get("gamma", 0.99),
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4),
            device=device
            )
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}")


    # --------------------------------------------------
    # Unified API
    # --------------------------------------------------
    def push(self, *args, **kwargs):
        return self.buffer.push(*args, **kwargs)


    def sample(self, batch_size):
        return self.buffer.sample(batch_size)


    def update_priorities(self, indices, td_errors):
        if hasattr(td_errors, "detach"):
            td_errors = (
                td_errors.detach()
                .cpu()
                .numpy()
            )

        self.buffer.update_priorities(indices, td_errors)


    def __len__(self):
        return len(self.buffer)