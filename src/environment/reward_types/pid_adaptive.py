import numpy as np
from .base_reward import BaseReward


class PIDAdaptiveReward(BaseReward):

    def compute(self) -> float:
        m = self.m

        if m.count < 2:
            return 0.0

        # ── Parameters ────────────────────────────────────────────
        wP_far   = 2.0   # เน้น error ตอน far
        wP_near  = 0.5

        wD_far   = 0.2
        wD_near  = 1.5   # เน้น smooth ตอน near

        wI       = 0.5
        wE       = 0.5   # delta error

        k_prox   = 6.0   # shape ของ transition

        threshold = 0.05

        # ── State ────────────────────────────────────────────────
        idx_now  = (m.ptr - 1) % m.buffer_size
        idx_prev = (m.ptr - 2) % m.buffer_size

        e_now = m.error_buffer[idx_now]
        e_prev = m.error_buffer[idx_prev]

        a_now  = m.action_buffer[idx_now]
        a_prev = m.action_buffer[idx_prev]

        da = a_now - a_prev
        de = e_now - e_prev

        # normalize
        e_n  = e_now / (m.max_error + 1e-8)
        da_n = da    / (m.max_delta_action + 1e-8)
        de_n = de    / (m.max_delta_error + 1e-8)

        # integral
        integral = m._update_integral(e_n, decay=0.95, clip=2.0)

        # ── Proximity (soft threshold) ───────────────────────────
        proximity = np.exp(-k_prox * (e_n ** 2))

        # ── Adaptive weights ─────────────────────────────────────
        wP = wP_far  * (1 - proximity) + wP_near * proximity
        wD = wD_far  * (1 - proximity) + wD_near * proximity

        # ── PID-style penalties ─────────────────────────────────
        rP = -(wP * (e_n ** 2))
        rD = -(wD * (da_n ** 2))
        rI = -(wI * (integral ** 2))
        rE = -(wE * (de_n ** 2))

        # ── Bonus (เมื่อเข้า threshold) ─────────────────────────
        bonus = 0.0
        if abs(e_n) < threshold:
            bonus = 1.0 * proximity  # smooth bonus

        reward = rP + rD + rI + rE + bonus

        return np.tanh(reward)