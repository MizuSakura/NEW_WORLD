# reward_types/hybrid.py

import numpy as np
from .base_reward import BaseReward


class HybridReward(BaseReward):

    def compute(self):
        m = self.m

        if m.count < 2:
            return 0.0

        e_now  = m.error_buffer[(m.ptr - 1) % m.buffer_size]
        a_now  = m.action_buffer[(m.ptr - 1) % m.buffer_size]
        a_prev = m.action_buffer[(m.ptr - 2) % m.buffer_size]

        da = a_now - a_prev
        de = m.delta_error()

        r_error     = -abs(e_now)
        r_smooth    = -abs(da)
        r_stability = -abs(de)

        k       = 3.0 / (m.max_error + 1e-8)
        r_bonus = np.exp(-k * abs(e_now))

        return (
            1.0 * r_error
            + 0.8 * r_smooth
            + 0.3 * r_stability
            + 0.5 * r_bonus
        )