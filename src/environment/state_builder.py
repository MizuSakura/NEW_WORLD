# src/environment/state_builder.py
"""
StateBuilder — Canonical State Builder
---------------------------------------
สร้าง state vector เหมือนกันทุก environment
ผู้ใช้กำหนด feature ผ่าน rl_params.yaml section state
"""

import numpy as np
from collections import deque
from pathlib import Path


class StateBuilder:
    """
    สร้าง canonical state จาก config

    config format:
        level_history:    3    # จำนวน history
        action_history:   3
        error_history:    0    # 0 = ไม่ใช้
        setpoint_history: 3
        integral:         true
        derivative:       true
        level_max:        10.0

    state order:
        [level_history..., action_history...,
         error_history..., setpoint_history...,
         integral (optional), derivative (optional)]
    """

    def __init__(self, cfg: dict):
        self.level_hist_len    = int(cfg.get("level_history",    3))
        self.action_hist_len   = int(cfg.get("action_history",   3))
        self.error_hist_len    = int(cfg.get("error_history",    0))
        self.setpoint_hist_len = int(cfg.get("setpoint_history", 3))
        self.use_integral      = bool(cfg.get("integral",        True))
        self.use_derivative    = bool(cfg.get("derivative",      True))
        self.level_max         = float(cfg.get("level_max",      10.0))

        # Buffers
        self._level_buf    = deque(maxlen=max(self.level_hist_len,    1))
        self._action_buf   = deque(maxlen=max(self.action_hist_len,   1))
        self._error_buf    = deque(maxlen=max(self.error_hist_len,    1))
        self._setpoint_buf = deque(maxlen=max(self.setpoint_hist_len, 1))

        # PID state
        self._integral   = 0.0
        self._prev_error = 0.0
        self._dt         = 0.1

        # คำนวณ state_dim
        self.state_dim = self._compute_state_dim()

    def _compute_state_dim(self) -> int:
        dim = 0
        dim += self.level_hist_len
        dim += self.action_hist_len
        if self.error_hist_len > 0:
            dim += self.error_hist_len
        if self.setpoint_hist_len > 0:
            dim += self.setpoint_hist_len
        if self.use_integral:
            dim += 1
        if self.use_derivative:
            dim += 1
        return dim

    def reset(self, level: float, action: float, setpoint: float, dt: float = 0.1):
        """เรียกตอน environment reset"""
        self._dt         = dt
        self._integral   = 0.0
        self._prev_error = 0.0

        self._level_buf.clear()
        self._action_buf.clear()
        self._error_buf.clear()
        self._setpoint_buf.clear()
        self.get_state()

        for _ in range(max(self.level_hist_len, 1)):
            self._level_buf.append(level)
        for _ in range(max(self.action_hist_len, 1)):
            self._action_buf.append(action)
        for _ in range(max(self.error_hist_len, 1)):
            self._error_buf.append(0.0)
        for _ in range(max(self.setpoint_hist_len, 1)):
            self._setpoint_buf.append(setpoint)

    def get_state(self) -> np.ndarray:
        """คืน state ปัจจุบันโดยไม่อัปเดต buffer"""
        return self._build()


    def update(self, level: float, action: float, setpoint: float) -> np.ndarray:
        """
        อัปเดต buffers แล้วคืน state vector
        เรียกทุก step
        """
        error = setpoint - level
        error = np.clip(error, -self.level_max, self.level_max)

        # Integral (anti-windup)
        self._integral += error * self._dt
        self._integral  = np.clip(self._integral, -self.level_max, self.level_max)

        # Derivative
        derivative = (error - self._prev_error) / self._dt
        derivative  = np.clip(derivative, -self.level_max, self.level_max)
        self._prev_error = error

        # Update buffers
        self._level_buf.append(level)
        self._action_buf.append(action)
        self._error_buf.append(error)
        self._setpoint_buf.append(setpoint)

        return self._build()

    def _build(self) -> np.ndarray:
        parts = []

        if self.level_hist_len > 0:
            parts.extend(list(self._level_buf)[-self.level_hist_len:])

        if self.action_hist_len > 0:
            parts.extend(list(self._action_buf)[-self.action_hist_len:])

        if self.error_hist_len > 0:
            parts.extend(list(self._error_buf)[-self.error_hist_len:])

        if self.setpoint_hist_len > 0:
            parts.extend(list(self._setpoint_buf)[-self.setpoint_hist_len:])

        if self.use_integral:
            parts.append(self._integral)

        if self.use_derivative:
            parts.append(self._prev_error)   # derivative ล่าสุด

        return np.array(parts, dtype=np.float32)

    def obs_bounds(self, action_max: float):
        """คืน low/high สำหรับ observation_space"""
        low, high = [], []

        if self.level_hist_len > 0:
            low  += [0.0]          * self.level_hist_len
            high += [self.level_max] * self.level_hist_len

        if self.action_hist_len > 0:
            low  += [0.0]        * self.action_hist_len
            high += [action_max] * self.action_hist_len

        if self.error_hist_len > 0:
            low  += [-self.level_max] * self.error_hist_len
            high += [self.level_max]  * self.error_hist_len

        if self.setpoint_hist_len > 0:
            low  += [0.0]          * self.setpoint_hist_len
            high += [self.level_max] * self.setpoint_hist_len

        if self.use_integral:
            low  += [-self.level_max]
            high += [self.level_max]

        if self.use_derivative:
            low  += [-self.level_max]
            high += [self.level_max]

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "StateBuilder":
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            rl_cfg = yaml.safe_load(f)
        return cls(rl_cfg.get("state", {}))

    def __repr__(self):
        return (
            f"StateBuilder(dim={self.state_dim}, "
            f"level×{self.level_hist_len}, "
            f"action×{self.action_hist_len}, "
            f"error×{self.error_hist_len}, "
            f"setpoint×{self.setpoint_hist_len}, "
            f"integral={self.use_integral}, "
            f"derivative={self.use_derivative})"
        )
    