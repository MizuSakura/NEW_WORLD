# my_project/src/environment/reward_function_control.py

import numpy as np


class Reward_manager:
    """
    Reward Manager for Continuous Control
    --------------------------------------
    Modes:
        - "raw"       : No normalization
        - "adaptive"  : Physics-based normalization with adaptive bounds

    Reward Characteristics:
        - MAX = 0
        - Negative only
        - Stable for RL training
    """

    def __init__(
        self,
        buffer_size=5,
        mode="raw",  # "raw" or "adaptive"
        max_error=10.0,
        max_delta_action=1.0,
        max_delta_error=0.5,
    ):

        self.buffer_size = buffer_size
        self.mode = mode

        # --- Initial physical limits (can grow adaptively) ---
        self.max_error = max_error
        self.max_delta_action = max_delta_action
        self.max_delta_error = max_delta_error

        # FIFO buffers
        self.setpoint_buffer = np.zeros(buffer_size)
        self.state_buffer = np.zeros(buffer_size)
        self.error_buffer = np.zeros(buffer_size)
        self.action_buffer = np.zeros(buffer_size)

        # Reward weights
        self.w_error = 1.0
        self.w_smooth = 0.3
        self.w_stability = 0.3

        self.ptr = 0
        self.count = 0
        self.non_converge_counter = 0
        # -------------------------------
        # Reward registry (Strategy Map)
        # -------------------------------
        self.reward_registry = {
            "continuous": self.reward_continuous_control,
            "hybrid": self.reward_hybrid,
            "control_v1": self.reward_control_V1
        }


    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, init_setpoint, init_state, init_action):

        init_error = init_setpoint - init_state

        self.setpoint_buffer[:] = init_setpoint
        self.state_buffer[:] = init_state
        self.error_buffer[:] = init_error
        self.action_buffer[:] = init_action

        self.ptr = 0
        self.count = self.buffer_size
        self.non_converge_counter = 0

    # --------------------------------------------------
    # Update buffers
    # --------------------------------------------------
    def update(self, setpoint, state, action):

        error = setpoint - state

        self.setpoint_buffer[self.ptr] = setpoint
        self.state_buffer[self.ptr] = state
        self.error_buffer[self.ptr] = error
        self.action_buffer[self.ptr] = action

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def previous_action(self):
        if self.count < 2:
            return 0.0
        idx = (self.ptr - 2) % self.buffer_size
        return self.action_buffer[idx]

    def delta_error(self):
        if self.count < 2:
            return 0.0
        idx_now = (self.ptr - 1) % self.buffer_size
        idx_prev = (self.ptr - 2) % self.buffer_size
        return self.error_buffer[idx_now] - self.error_buffer[idx_prev]

    # --------------------------------------------------
    # Adaptive bound update
    # --------------------------------------------------
    def _update_adaptive_bounds(self, e_now, da, de):

        self.max_error = max(self.max_error, abs(e_now))
        self.max_delta_action = max(self.max_delta_action, abs(da))
        self.max_delta_error = max(self.max_delta_error, abs(de))


    def reward_type(self, reward_name: str):

        if reward_name not in self.reward_registry:
            raise ValueError(
                f"Reward '{reward_name}' not registered. "
                f"Available rewards: {list(self.reward_registry.keys())}"
            )

        return self.reward_registry[reward_name]()
    
    # --------------------------------------------------
    # Reward calculation
    # --------------------------------------------------
    def reward_continuous_control(self):

        if self.count < 2:
            return 0.0

        e_now = self.error_buffer[(self.ptr - 1) % self.buffer_size]
        a_now = self.action_buffer[(self.ptr - 1) % self.buffer_size]
        a_prev = self.action_buffer[(self.ptr - 2) % self.buffer_size]

        da = a_now - a_prev
        de = self.delta_error()

        # --------------------------------------------------
        # RAW MODE
        # --------------------------------------------------
        if self.mode == "raw":

            r_error = -abs(e_now)
            r_smooth = -abs(da)
            r_stability = -abs(de)

        # --------------------------------------------------
        # ADAPTIVE MODE
        # --------------------------------------------------
        elif self.mode == "adaptive":

            # Update bounds dynamically
            self._update_adaptive_bounds(e_now, da, de)

            # Normalize using updated bounds
            e_norm = e_now / (self.max_error + 1e-8)
            da_norm = da / (self.max_delta_action + 1e-8)
            de_norm = de / (self.max_delta_error + 1e-8)

            r_error = -abs(e_norm)
            r_smooth = -abs(da_norm)
            r_stability = -abs(de_norm)

        else:
            raise ValueError("Mode must be 'raw' or 'adaptive'")

        reward = (
            self.w_error * r_error
            + self.w_smooth * r_smooth
            + self.w_stability * r_stability
        )

        return reward

    def reward_hybrid(self):

        if self.count < 2:
            return 0.0

        e_now = self.error_buffer[(self.ptr - 1) % self.buffer_size]
        a_now = self.action_buffer[(self.ptr - 1) % self.buffer_size]
        a_prev = self.action_buffer[(self.ptr - 2) % self.buffer_size]

        da = a_now - a_prev
        de = self.delta_error()

        # ----- Penalty terms -----
        r_error = -abs(e_now)
        r_smooth = -abs(da)
        r_stability = -abs(de)

        # ----- Near-zero bonus -----
        k = 3.0  # shaping sharpness
        r_bonus = np.exp(-k * abs(e_now))

        # ----- Weights -----
        w1 = 1.0   # error
        w2 = 0.3   # smooth
        w3 = 0.3   # stability
        w4 = 0.5   # bonus

        reward = (
            w1 * r_error
            + w2 * r_smooth
            + w3 * r_stability
            + w4 * r_bonus
        )

        return reward
    
    def reward_control_V1(self):

        if self.count < 2:
            return 0.0

        idx_now = (self.ptr - 1) % self.buffer_size
        idx_prev = (self.ptr - 2) % self.buffer_size

        e_now = self.error_buffer[idx_now]
        e_prev = self.error_buffer[idx_prev]

        a_now = self.action_buffer[idx_now]
        a_prev = self.action_buffer[idx_prev]

        da = a_now - a_prev
        de = e_now - e_prev

        # --------------------------
        # Hyperparameters
        # --------------------------
        w_error = 1.0
        w_smooth = 0.1
        w_osc = 0.2
        w_lazy = 0.05
        w_converge = 1.0

        tolerance = 0.02
        eps = 1e-6

        # --------------------------
        # Core penalties
        # --------------------------
        r_error = - (e_now ** 2)
        r_smooth = - (da ** 2)
        r_osc = - (de ** 2)

        # --------------------------
        # Lazy penalty
        # error ใหญ่ แต่ Δa เล็ก
        # --------------------------
        r_lazy = - w_lazy * (abs(e_now) / (abs(da) + eps))

        # --------------------------
        # Convergence bonus
        # --------------------------
        if abs(e_now) < tolerance:
            r_converge = w_converge
        else:
            r_converge = 0.0

        reward = (
            w_error * r_error
            + w_smooth * r_smooth
            + w_osc * r_osc
            + r_lazy
            + r_converge
        )
        reward_scale = 5.0   # ปรับตามระบบคุณ

        reward = reward_scale * np.tanh(reward / reward_scale)

        return reward
