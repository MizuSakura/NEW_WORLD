# reward_types/control_v1.py

import numpy as np
from .base_reward import BaseReward


class ControlV1Reward(BaseReward):

    def compute(self):
        m = self.m

        if m.count < 2:
            return 0.0

        idx_now  = (m.ptr - 1) % m.buffer_size
        idx_prev = (m.ptr - 2) % m.buffer_size

        e_now  = m.error_buffer[idx_now]
        e_prev = m.error_buffer[idx_prev]
        a_now  = m.action_buffer[idx_now]
        a_prev = m.action_buffer[idx_prev]

        da = a_now - a_prev
        de = e_now - e_prev

        e_n  = e_now / (m.max_error + 1e-8)
        da_n = da    / (m.max_delta_action + 1e-8)
        de_n = de    / (m.max_delta_error + 1e-8)

        r_error  = -(e_n ** 2)
        r_smooth = -(da_n ** 2)
        r_osc    = -(de_n ** 2)
        r_lazy   = -0.05 * (abs(e_n) / (abs(da_n) + 1e-6))

        r_converge = 1.5 if abs(e_n) < 0.05 else 0.0

        reward = (
            1.0 * r_error
            + 0.8 * r_smooth
            + 0.3 * r_osc
            + r_lazy
            + r_converge
        )

        return 5.0 * np.tanh(reward / 5.0)