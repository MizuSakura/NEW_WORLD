import numpy as np
from .base_reward import BaseReward


class ZoneStableReward(BaseReward):
    """
    Reward: zone_stable
    -------------------
    Pure Negative Space (max = 0.0)

    Components:
        r_error     : linear penalty ตาม |error|
        r_integral  : accumulated error penalty (leaky integrator)
        r_smooth    : action roughness penalty
        r_tol_bonus : bonus เมื่ออยู่ใน tolerance zone
        r_stable    : bonus เมื่ออยู่ใน zone + action นิ่ง
    """

    def compute(self) -> float:
        m = self.m

        if m.count < 2:
            return 0.0

        # ── Parameters ────────────────────────────────────────────
        tol              = 0.02
        w_error          = 1.5
        w_integral       = 0.8
        integral_decay   = 0.95
        integral_clip    = 2.0
        w_smooth         = 1.0
        bonus_tol        = 0.3
        bonus_stable     = 0.2
        smooth_window    = 10
        stable_threshold = 0.10

        # ── Current normalized error ──────────────────────────────
        e_now = m.error_buffer[(m.ptr - 1) % m.buffer_size]
        e_n   = e_now / (m.max_error + 1e-8)

        # 1. Error penalty
        r_error = -w_error * abs(e_n)

        # 2. Integral penalty (leaky integrator)
        integral_norm = m._update_integral(e_n, decay=integral_decay, clip=integral_clip)
        r_integral    = -w_integral * integral_norm

        # 3. Smooth penalty (action roughness)
        roughness = m._action_roughness(smooth_window)
        r_smooth  = -w_smooth * roughness

        # 4. Zone bonuses
        r_tol_bonus = 0.0
        r_stable    = 0.0

        if abs(e_n) < tol:
            r_tol_bonus = bonus_tol

            if roughness < stable_threshold:
                stability_score = 1.0 - (roughness / stable_threshold)
                r_stable        = bonus_stable * stability_score

        reward = r_error + r_integral + r_smooth + r_tol_bonus + r_stable

        return min(reward, 0.0)