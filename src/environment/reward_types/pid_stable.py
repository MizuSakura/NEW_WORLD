# reward_types/pid_stable.py

import numpy as np
from .base_reward import BaseReward


class PIDStableReward(BaseReward):

    def compute(self):
        m = self.m

        if m.count < 2:
            return 0.0

        idx_now  = (m.ptr - 1) % m.buffer_size
        idx_prev = (m.ptr - 2) % m.buffer_size

        e_now  = m.error_buffer[idx_now]
        a_now  = m.action_buffer[idx_now]
        a_prev = m.action_buffer[idx_prev]

        da = a_now - a_prev

        e_n  = e_now / (m.max_error + 1e-8)
        da_n = da    / (m.max_delta_action + 1e-8)

        r_error  = -abs(e_n)
        r_smooth = -(da_n ** 2)

        zone_threshold = 0.1
        if abs(e_n) < zone_threshold:
            proximity = 1.0 - (abs(e_n) / zone_threshold)
            r_smooth *= (1.0 + 4.0 * proximity)

        r_bonus = 2.0 * np.exp(-8.0 * (e_n ** 2))

        reward = (
            1.0 * r_error
            + 1.0 * r_smooth
            + 2.0 * r_bonus
        )

        return 5.0 * np.tanh(reward / 5.0)