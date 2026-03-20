# D:\Project_end\New_world\my_project\src\environment\Apply_real_env.py
# Checkpoint 11 — ใช้ StateBuilder แทน hardcode deque

from pathlib import Path

import numpy as np
import time

from src.utils.comucation_modbusTCP import ModbusTCP
from src.environment.reward_function_control import Reward_manager
from src.environment.state_builder import StateBuilder


class Real_env_remote:
    """
    Real Remote Environment with ModbusTCP
    StateBuilder version — state structure identical to RCTankEnv_gym

    State format ถูกกำหนดจาก rl_params.yaml section 'state' ผ่าน StateBuilder
    ดังนั้น state_dim จะ match กับ gym env โดยอัตโนมัติ

    ตัวอย่าง (default yaml):
        [level×3, action×3, setpoint×3, integral, derivative]  → dim=11
    """

    def __init__(
        self,
        ip_host: str = "192.168.1.100",
        port: int = 502,
        min_action: float = 0.0,
        max_action: float = 10.0,
        setpoint: float = 5.0,
        delay_of_action: float = 0.2,
        address_sensor: int = None,
        address_actuator: int = None,
        config_path: Path = None,
    ):
        # --------------------------------------------------
        # StateBuilder — อ่านจาก rl_params.yaml
        # --------------------------------------------------
        if config_path is None:
            config_path = Path("src/API/config/rl_params.yaml")

        self.state_builder = StateBuilder.from_yaml(config_path)
        print(f"[Real_env_remote] {self.state_builder}")

        # --------------------------------------------------
        # Communication
        # --------------------------------------------------
        self.ip_host = ip_host
        self.modbus = ModbusTCP(host=self.ip_host, port=port)
        self.modbus.connect()

        # --------------------------------------------------
        # Reward Manager (SAME as sim)
        # --------------------------------------------------
        self.reward_manager = Reward_manager(buffer_size=5)

        # --------------------------------------------------
        # Remote IO scaling
        # --------------------------------------------------
        self.max_value_remote_IO = 27647
        self.min_value_remote_IO = 0
        self.address_sensor = address_sensor
        self.address_actuator = address_actuator

        # --------------------------------------------------
        # Action space
        # --------------------------------------------------
        self.min_action = min_action
        self.max_action = max_action
        self.action_dim = 1

        # --------------------------------------------------
        # Internal
        # --------------------------------------------------
        self.setpoint = setpoint
        self.delay = delay_of_action

    # -------------------------------------------------------
    # Properties — ให้ train_SAC_real_agent เรียกได้เหมือนเดิม
    # -------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return self.state_builder.state_dim

    # ======================================================
    # IO FUNCTIONS
    # ======================================================
    def read_sensor(self, address: int = None) -> float:
        if address is None:
            raise ValueError("Sensor register address is missing")

        raw_value = self.modbus.analog_read(address=address)
        value = np.interp(
            raw_value,
            [self.min_value_remote_IO, self.max_value_remote_IO],
            [self.min_action, self.max_action],
        )
        return float(value)

    def write_actuator(self, address: int = None, action: float = None):
        if address is None:
            raise ValueError("Actuator register address is missing")

        raw = int(
            np.interp(
                action,
                [self.min_action, self.max_action],
                [self.min_value_remote_IO, self.max_value_remote_IO],
            )
        )
        self.modbus.write_holding_register(address=address, value=raw)
        return action

    # ======================================================
    # RESET
    # ======================================================
    def reset(self):
        # อ่าน level จริงจาก sensor
        level = self.read_sensor(self.address_sensor)

        # สุ่ม setpoint (same philosophy as sim)
        self.setpoint = float(np.random.uniform(self.min_action, self.max_action))

        # reset StateBuilder → init ด้วย level จริง, action=0, setpoint ที่สุ่มได้
        init_action = np.clip(level, self.min_action, self.max_action)
        self.state_builder.reset(
            level=level,
            action=init_action,
            setpoint=self.setpoint,
            dt=self.delay,
        )

        # reset reward manager
        self.reward_manager.reset(
            init_setpoint=self.setpoint,
            init_state=level,
            init_action=init_action,
        )

        # state จาก StateBuilder (ไม่ต้อง update — reset คืนค่า state เริ่มต้น)
        state = self.state_builder.get_state()

        info = {"setpoint": self.setpoint, "level": level}
        return state, info

    # ======================================================
    # STEP
    # ======================================================
    def step(self, action):
        action = float(np.clip(action, self.min_action, self.max_action))

        # ส่ง action ไปยัง actuator จริง
        self.write_actuator(self.address_actuator, action)

        # รอ physical / communication delay
        time.sleep(self.delay)

        # อ่าน level จาก sensor จริง
        level = self.read_sensor(self.address_sensor)

        # update StateBuilder → คำนวณ history, integral, derivative ทั้งหมด
        state = self.state_builder.update(
            level=level,
            action=action,
            setpoint=self.setpoint,
        )

        # reward (same semantics as sim)
        self.reward_manager.update(
            setpoint=self.setpoint,
            state=level,
            action=action,
        )
        reward = self.reward_manager.reward_continuous_control()

        # termination (soft & safe for real system)
        error = abs(self.setpoint - level)
        done = error < 0.1

        info = {
            "error": error,
            "raw_level": level,
            "setpoint": self.setpoint,
        }

        return state, reward, done, info