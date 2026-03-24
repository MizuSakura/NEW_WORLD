# reward_types/continuous.py

import numpy as np
from .base_reward import BaseReward


class ContinuousReward(BaseReward):

    def compute(self):
        m = self.m

        if m.count < 2:
            return 0.0

        e_now  = m.error_buffer[(m.ptr - 1) % m.buffer_size]
        a_now  = m.action_buffer[(m.ptr - 1) % m.buffer_size]
        a_prev = m.action_buffer[(m.ptr - 2) % m.buffer_size]

        da = a_now - a_prev
        de = m.delta_error()

        if m.mode == "raw":
            r_error     = -abs(e_now)
            r_smooth    = -abs(da)
            r_stability = -abs(de)

        elif m.mode == "adaptive":
            m._update_adaptive_bounds(e_now, da, de)

            e_norm  = e_now / (m._ema_error + 1e-8)
            da_norm = da    / (m._ema_delta_action + 1e-8)
            de_norm = de    / (m._ema_delta_error + 1e-8)

            r_error     = -abs(e_norm)
            r_smooth    = -abs(da_norm)
            r_stability = -abs(de_norm)

        else:
            raise ValueError("mode must be 'raw' or 'adaptive'")

        return (
            m.w_error     * r_error
            + m.w_smooth    * r_smooth
            + m.w_stability * r_stability
        )