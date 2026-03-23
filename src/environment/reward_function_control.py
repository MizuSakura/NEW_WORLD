# my_project/src/environment/reward_function_control.py

import numpy as np


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

        # Reward weights สำหรับ continuous/hybrid
        self.w_error     = 1.0
        self.w_smooth    = 0.8   # เพิ่มจาก 0.3 → 0.8
        self.w_stability = 0.3

        # -------------------------------
        # Reward registry (Strategy Map)
        # -------------------------------
        self.reward_registry = {
            "continuous" : self.reward_continuous_control,
            "hybrid"     : self.reward_hybrid,
            "control_v1" : self.reward_control_V1,
            "pid_stable" : self.reward_pid_stable,      # NEW
        }

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

    # --------------------------------------------------
    # Reward type dispatcher
    # --------------------------------------------------
    def reward_type(self, reward_name: str):
        if reward_name not in self.reward_registry:
            raise ValueError(
                f"Reward '{reward_name}' not registered. "
                f"Available: {list(self.reward_registry.keys())}"
            )
        return self.reward_registry[reward_name]()

    # --------------------------------------------------
    # 1. reward_continuous_control
    # --------------------------------------------------
    def reward_continuous_control(self):
        if self.count < 2:
            return 0.0

        e_now  = self.error_buffer[(self.ptr - 1) % self.buffer_size]
        a_now  = self.action_buffer[(self.ptr - 1) % self.buffer_size]
        a_prev = self.action_buffer[(self.ptr - 2) % self.buffer_size]

        da = a_now - a_prev
        de = self.delta_error()

        if self.mode == "raw":
            r_error     = -abs(e_now)
            r_smooth    = -abs(da)
            r_stability = -abs(de)

        elif self.mode == "adaptive":
            self._update_adaptive_bounds(e_now, da, de)

            e_norm  = e_now / (self._ema_error        + 1e-8)
            da_norm = da    / (self._ema_delta_action  + 1e-8)
            de_norm = de    / (self._ema_delta_error   + 1e-8)

            r_error     = -abs(e_norm)
            r_smooth    = -abs(da_norm)
            r_stability = -abs(de_norm)

        else:
            raise ValueError("mode must be 'raw' or 'adaptive'")

        return (
            self.w_error     * r_error
            + self.w_smooth    * r_smooth
            + self.w_stability * r_stability
        )

    # --------------------------------------------------
    # 2. reward_hybrid
    # --------------------------------------------------
    def reward_hybrid(self):
        if self.count < 2:
            return 0.0

        e_now  = self.error_buffer[(self.ptr - 1) % self.buffer_size]
        a_now  = self.action_buffer[(self.ptr - 1) % self.buffer_size]
        a_prev = self.action_buffer[(self.ptr - 2) % self.buffer_size]

        da = a_now - a_prev
        de = self.delta_error()

        r_error     = -abs(e_now)
        r_smooth    = -abs(da)
        r_stability = -abs(de)

        k       = 3.0 / (self.max_error + 1e-8)   # normalize k ด้วย max_error
        r_bonus = np.exp(-k * abs(e_now))

        return (
            1.0 * r_error
            + 0.8 * r_smooth        # เพิ่มจาก 0.3
            + 0.3 * r_stability
            + 0.5 * r_bonus
        )

    # --------------------------------------------------
    # 3. reward_control_V1
    # --------------------------------------------------
    def reward_control_V1(self):
        if self.count < 2:
            return 0.0

        idx_now  = (self.ptr - 1) % self.buffer_size
        idx_prev = (self.ptr - 2) % self.buffer_size

        e_now  = self.error_buffer[idx_now]
        e_prev = self.error_buffer[idx_prev]
        a_now  = self.action_buffer[idx_now]
        a_prev = self.action_buffer[idx_prev]

        da = a_now - a_prev
        de = e_now - e_prev

        # normalize ด้วย max_error แทนค่า raw (แก้ scale issue)
        e_n  = e_now / (self.max_error        + 1e-8)
        da_n = da    / (self.max_delta_action  + 1e-8)
        de_n = de    / (self.max_delta_error   + 1e-8)

        w_error    = 1.0
        w_smooth   = 0.8    # เพิ่มจาก 0.1
        w_osc      = 0.3    # เพิ่มจาก 0.2
        w_lazy     = 0.05
        w_converge = 1.5    # เพิ่มจาก 1.0

        tolerance = 0.05    # 5% ของ normalized range
        eps       = 1e-6

        r_error  = -(e_n  ** 2)
        r_smooth = -(da_n ** 2)
        r_osc    = -(de_n ** 2)
        r_lazy   = -w_lazy * (abs(e_n) / (abs(da_n) + eps))

        r_converge = w_converge if abs(e_n) < tolerance else 0.0

        reward = (
            w_error    * r_error
            + w_smooth * r_smooth
            + w_osc    * r_osc
            + r_lazy
            + r_converge
        )

        reward_scale = 5.0
        return reward_scale * np.tanh(reward / reward_scale)

    # --------------------------------------------------
    # 4. reward_pid_stable  *** NEW ***
    #
    # ออกแบบให้ agent ทำตัวเหมือน PID:
    #   - ลด error → เข้าหา setpoint
    #   - smooth action → ไม่แกว่ง
    #   - progressive penalty: ยิ่งใกล้ setpoint ยิ่งต้องนิ่ง
    #   - exponential bonus: อยู่ใน dead zone ได้รับ reward หนาแน่น
    # --------------------------------------------------
    def reward_pid_stable(self):
        if self.count < 2:
            return 0.0

        idx_now  = (self.ptr - 1) % self.buffer_size
        idx_prev = (self.ptr - 2) % self.buffer_size

        e_now  = self.error_buffer[idx_now]
        a_now  = self.action_buffer[idx_now]
        a_prev = self.action_buffer[idx_prev]
        da     = a_now - a_prev

        # --- Normalize ด้วย fixed bounds (ไม่ drift) ---
        e_n  = e_now / (self.max_error        + 1e-8)
        da_n = da    / (self.max_delta_action  + 1e-8)

        # --- 1. Error penalty (quadratic → linear ใกล้ zero) ---
        r_error = -abs(e_n)

        # --- 2. Smooth penalty (quadratic = โทษหนักเมื่อแกว่งมาก) ---
        r_smooth = -(da_n ** 2)

        # --- 3. Progressive smooth: ยิ่งใกล้ setpoint ยิ่งต้องนิ่ง ---
        # ถ้า |e_n| < 20% แต่ยัง da ใหญ่ = โทษเพิ่ม 5×
        # เลียนแบบ PID derivative term ที่ dampen oscillation ใกล้ setpoint
        zone_threshold = 0.1   # 20% ของ max_error
        if abs(e_n) < zone_threshold:
            proximity = 1.0 - (abs(e_n) / zone_threshold)   # 0→1 ยิ่งใกล้ยิ่งมาก
            r_smooth = r_smooth * (1.0 + 4.0 * proximity)   # สูงสุด ×5

        # --- 4. Convergence bonus (exponential density ใน dead zone) ---
        # เหมือน integral term ของ PID ที่ pull ค่าเข้า setpoint
        r_bonus = 2.0 * np.exp(-8.0 * (e_n ** 2))

        # --- weights ---
        reward = (
            1.0 * r_error
            + 1.0 * r_smooth    # weight สูง = นิ่งสำคัญพอๆ กับ error
            + 2.0 * r_bonus
        )

        # tanh เพื่อ bound [-5, 0] (bonus ทำให้อาจบวกได้เล็กน้อย)
        reward_scale = 5.0
        return reward_scale * np.tanh(reward / reward_scale)