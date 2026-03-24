# my_project/src/environment/reward_function_control.py

import numpy as np
from .reward_types import REWARD_REGISTRY

class Reward_manager:
    """
    Reward Manager for Continuous Control
    --------------------------------------
    Modes:
        - "raw"      : No normalization (ใช้ค่าจริง)
        - "adaptive" : Adaptive bounds — แก้ bug drift แล้ว
                       ใช้ exponential moving average แทน max-only

    Reward Types:
        - "continuous"  : error + smooth + stability (raw/adaptive)
        - "hybrid"      : continuous + near-zero bonus
        - "control_v1"  : quadratic penalties + lazy + convergence
        - "pid_stable"  : *** NEW *** ออกแบบให้ action นิ่งแบบ PID
                          - normalize ด้วย fixed bounds (ไม่ drift)
                          - progressive smooth penalty ใกล้ setpoint
                          - convergence bonus แบบ exponential
    """

    def __init__(
        self,
        buffer_size=10,
        mode="raw",
        max_error=10.0,
        max_delta_action=5.0,
        max_delta_error=2.0,
    ):
        self.buffer_size = buffer_size
        self.mode = mode

        # --- Fixed physical limits (ใช้เป็น normalization reference) ---
        self.max_error        = max_error
        self.max_delta_action = max_delta_action
        self.max_delta_error  = max_delta_error

        # --- EMA bounds สำหรับ adaptive mode (แก้ bug drift จาก max-only) ---
        self._ema_error        = max_error
        self._ema_delta_action = max_delta_action
        self._ema_delta_error  = max_delta_error
        self._ema_alpha        = 0.05   # decay rate: ช้า=นิ่ง, เร็ว=ตามจริง

        # FIFO buffers
        self.setpoint_buffer = np.zeros(buffer_size)
        self.state_buffer    = np.zeros(buffer_size)
        self.error_buffer    = np.zeros(buffer_size)
        self.action_buffer   = np.zeros(buffer_size)

        self.ptr   = 0
        self.count = 0
        self.non_converge_counter = 0
        self._integral_error = 0.0

        # Reward weights สำหรับ continuous/hybrid
        self.w_error     = 1.0
        self.w_smooth    = 0.8   # เพิ่มจาก 0.3 → 0.8
        self.w_stability = 0.3

        # -------------------------------
        # Reward registry (Strategy Map)
        # -------------------------------
        self.reward_registry = REWARD_REGISTRY

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, init_setpoint, init_state, init_action):
        init_error = init_setpoint - init_state

        self.setpoint_buffer[:] = init_setpoint
        self.state_buffer[:]    = init_state
        self.error_buffer[:]    = init_error
        self.action_buffer[:]   = init_action

        self.ptr   = 0
        self.count = self.buffer_size
        self.non_converge_counter = 0
        self._integral_error = 0.0

        # Reset EMA bounds ด้วย — ไม่งั้น episode ใหม่ยังถือค่าเก่า
        self._ema_error        = self.max_error
        self._ema_delta_action = self.max_delta_action
        self._ema_delta_error  = self.max_delta_error

    # --------------------------------------------------
    # Update buffers
    # --------------------------------------------------
    def update(self, setpoint, state, action):
        error = setpoint - state

        self.setpoint_buffer[self.ptr] = setpoint
        self.state_buffer[self.ptr]    = state
        self.error_buffer[self.ptr]    = error
        self.action_buffer[self.ptr]   = action

        self.ptr   = (self.ptr + 1) % self.buffer_size
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
        idx_now  = (self.ptr - 1) % self.buffer_size
        idx_prev = (self.ptr - 2) % self.buffer_size
        return self.error_buffer[idx_now] - self.error_buffer[idx_prev]

    # --------------------------------------------------
    # Adaptive bound update — แก้ bug: ใช้ EMA แทน max-only
    # max-only ทำให้ bound โตแล้วไม่ลด → r_smooth กลายเป็น ~0
    # EMA ทำให้ bound ค่อยๆ decay ลงเมื่อ system นิ่งขึ้น
    # --------------------------------------------------
    def _update_adaptive_bounds(self, e_now, da, de):
        a = self._ema_alpha

        # EMA update: ถ้า abs ใหม่ใหญ่กว่า → ขึ้นเร็ว, เล็กกว่า → ลงช้า
        target_e  = max(abs(e_now), self.max_error  * 0.1)
        target_da = max(abs(da),    self.max_delta_action * 0.05)
        target_de = max(abs(de),    self.max_delta_error  * 0.05)

        self._ema_error        = (1-a) * self._ema_error        + a * target_e
        self._ema_delta_action = (1-a) * self._ema_delta_action + a * target_da
        self._ema_delta_error  = (1-a) * self._ema_delta_error  + a * target_de

    def _update_integral(self, e_n, decay=0.95, clip=10):
        """
        Leaky integrator (anti-windup)
        """
        self._integral_error = decay * self._integral_error + e_n
        self._integral_error = np.clip(self._integral_error, -clip, clip)
        return abs(self._integral_error)
    
    def _action_roughness(self, window=10):
        """
        Measure how 'rough' action is over a window
        (mean absolute delta action)
        """
        if self.count < 2:
            return 0.0

        size = min(window, self.count)

        indices = [(self.ptr - i - 1) % self.buffer_size for i in range(size)]

        actions = [self.action_buffer[i] for i in reversed(indices)]

        diffs = np.diff(actions)

        if len(diffs) == 0:
            return 0.0

        return np.mean(np.abs(diffs)) / (self.max_delta_action + 1e-8)

    # --------------------------------------------------
    # Reward type dispatcher
    # --------------------------------------------------
    def reward_type(self, reward_name: str):

        if reward_name not in self.reward_registry:
            raise ValueError(
                f"Reward '{reward_name}' not registered. "
                f"Available: {list(self.reward_registry.keys())}"
            )

        reward_class = self.reward_registry[reward_name]
        reward_obj   = reward_class(self)

        return reward_obj.compute()

