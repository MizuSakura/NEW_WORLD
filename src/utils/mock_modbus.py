# src/utils/mock_modbus.py
"""
MockModbusTCP
-------------
จำลอง Modbus TCP โดยใช้ RC Tank physics ภายใน
ใช้แทน ModbusTCP จริงเพื่อทดสอบ evaluation_agent_real
โดยไม่ต้องต่อ hardware

วิธีใช้ 2 แบบ:

  1. Monkey-patch (แนะนำ) — ไม่ต้องแก้ไข Apply_real_env.py:
        import src.utils.comucation_modbusTCP as modbus_module
        from src.utils.mock_modbus import MockModbusTCP
        modbus_module.ModbusTCP = MockModbusTCP

  2. ส่งตรงผ่าน eval script ที่รองรับ mock_mode=True

Physics (เหมือน RC_Tank_env.py voltage mode):
    current      = (action - level) / R
    delta_level  = (current / C) * dt
    level       += delta_level  (clip 0~level_max)
"""

import numpy as np
import time


class MockModbusTCP:
    """
    Drop-in replacement สำหรับ ModbusTCP
    มี API เหมือนกันทุก method ที่ Real_env_remote ใช้

    Parameters
    ----------
    host       : str   (ไม่ได้ใช้ — รับไว้เพื่อ signature compatible)
    port       : int   (ไม่ได้ใช้)
    R          : float  ค่า resistance ของ RC Tank
    C          : float  ค่า capacitance ของ RC Tank
    dt         : float  simulation timestep (วินาที)
    level_max  : float  ระดับน้ำสูงสุด
    min_raw    : int    raw value ต่ำสุดของ Remote IO
    max_raw    : int    raw value สูงสุดของ Remote IO
    action_max : float  action สูงสุด (สำหรับ scale raw → action)
    noise_std  : float  noise บน sensor (0.0 = ไม่มี noise)
    verbose    : bool   print debug ทุก read/write
    """

    def __init__(
        self,
        host: str   = "mock",
        port: int   = 502,
        R: float    = 1.5,
        C: float    = 2.0,
        dt: float   = 0.1,
        level_max: float  = 10.0,
        min_raw: int      = 0,
        max_raw: int      = 27647,
        action_max: float = 10.0,
        noise_std: float  = 0.0,
        verbose: bool     = False,
    ):
        # physics params
        self.R          = R
        self.C          = C
        self.dt         = dt
        self.level_max  = level_max
        self.min_raw    = min_raw
        self.max_raw    = max_raw
        self.action_max = action_max
        self.noise_std  = noise_std
        self.verbose    = verbose

        # internal state
        self._level       = 0.0
        self._last_action = 0.0
        self._connected   = False

        # ใช้สำหรับ debug — เก็บ history
        self.level_history  = []
        self.action_history = []

    # ------------------------------------------------------------------
    # Connection (always succeeds)
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        self._connected = True
        print(f"[MockModbus] Connected (simulated) — RC Tank physics active")
        return True

    def disconnect(self) -> bool:
        self._connected = False
        print("[MockModbus] Disconnected")
        return True

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Physics step (เรียกทุกครั้งที่ analog_read)
    # ------------------------------------------------------------------
    def _step_physics(self) -> float:
        """
        อัปเดต level ด้วย RC Tank dynamics (voltage mode)
            current     = (action - level) / R
            delta_level = (current / C) * dt
        """
        current     = (self._last_action - self._level) / self.R
        delta_level = (current / self.C) * self.dt
        self._level  = float(np.clip(
            self._level + delta_level, 0.0, self.level_max
        ))

        # optional sensor noise
        if self.noise_std > 0:
            noisy = self._level + np.random.normal(0, self.noise_std)
            noisy = float(np.clip(noisy, 0.0, self.level_max))
        else:
            noisy = self._level

        self.level_history.append(self._level)
        return noisy

    # ------------------------------------------------------------------
    # Input Registers (Analog Read) — sensor
    # ------------------------------------------------------------------
    def analog_read(self, address, count=1, device_id=1):
        """
        จำลองการอ่านค่า sensor level
        อัปเดต physics 1 step แล้วแปลงกลับเป็น raw Remote IO value
        """
        noisy_level = self._step_physics()

        raw = int(np.interp(
            noisy_level,
            [0.0, self.level_max],
            [self.min_raw, self.max_raw]
        ))

        if self.verbose:
            print(f"[MockModbus] analog_read addr={address} "
                  f"level={noisy_level:.4f} raw={raw}")

        return raw if count == 1 else [raw]

    # ------------------------------------------------------------------
    # Holding Registers (Write) — actuator
    # ------------------------------------------------------------------
    def write_holding_register(self, address, value, device_id=1):
        """
        รับ raw value จาก actuator → แปลงเป็น action จริง → เก็บไว้
        physics จะใช้ค่านี้ในรอบ analog_read ถัดไป
        """
        action = float(np.interp(
            value,
            [self.min_raw, self.max_raw],
            [0.0, self.action_max]
        ))
        self._last_action = float(np.clip(action, 0.0, self.action_max))
        self.action_history.append(self._last_action)

        if self.verbose:
            print(f"[MockModbus] write_HR addr={address} "
                  f"raw={value} → action={self._last_action:.4f}")
        return True

    def read_holding_registers(self, address, count=1, device_id=1):
        """อ่าน holding register — คืน 0 เสมอ (mock)"""
        return [0] * count

    # ------------------------------------------------------------------
    # Coils (Digital) — mock ทั้งหมด
    # ------------------------------------------------------------------
    def digital_write(self, address, value, device_id=1):
        if self.verbose:
            print(f"[MockModbus] digital_write addr={address} value={value}")
        return True

    def read_status_output(self, address, device_id=1):
        return False

    def digital_input(self, address, count=1, device_id=1):
        return False if count == 1 else [False] * count

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def reset_state(self, level: float = 0.0, action: float = 0.0):
        """รีเซ็ต internal state — เรียกระหว่าง episode ถ้าต้องการ"""
        self._level       = float(np.clip(level, 0.0, self.level_max))
        self._last_action = float(np.clip(action, 0.0, self.action_max))
        self.level_history.clear()
        self.action_history.clear()

    def __repr__(self):
        return (
            f"MockModbusTCP("
            f"level={self._level:.3f}, "
            f"action={self._last_action:.3f}, "
            f"R={self.R}, C={self.C}, dt={self.dt})"
        )