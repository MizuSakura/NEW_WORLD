from src.utils.comucation_modbusTCP import ModbusTCP
from src.environment.reward_function_control import Reward_manager

import numpy as np
import time
from collections import deque


class Real_env_remote:
    """
    Real Remote Environment with ModbusTCP
    Canonical-state version for Sim-to-Real RL

    State format (IDENTICAL to RCTankEnv):
    [ level,
      prev_action_1, ..., prev_action_10,
      setpoint ]
    """

    def __init__(
        self,
        ip_host="192.168.1.100",
        port=502,
        min_action=0.0,
        max_action=10.0,
        setpoint=5.0,
        delay_of_action=0.2,
        address_sensor=None,
        address_actuator=None,
        action_history_len=10,
    ):
        # ==================================================
        # Communication
        # ==================================================
        self.ip_host = ip_host
        self.modbus = ModbusTCP(host=self.ip_host, port=port)
        self.modbus.connect()

        # ==================================================
        # Reward Manager (SAME as sim)
        # ==================================================
        self.reward_manager = Reward_manager(buffer_size=5)

        # ==================================================
        # Remote IO scaling
        # ==================================================
        self.max_value_remote_IO = 27647
        self.min_value_remote_IO = 0
        self.address_sensor = address_sensor
        self.address_actuator = address_actuator

        # ==================================================
        # Action space (IDENTICAL semantics to sim)
        # ==================================================
        self.min_action = min_action
        self.max_action = max_action

        # ==================================================
        # Canonical State
        # ==================================================
        self.action_history_len = action_history_len
        self.prev_actions = deque(maxlen=self.action_history_len)

        self.state_dim = 1 + self.action_history_len + 1
        # level + action_history + setpoint

        # ==================================================
        # Internal
        # ==================================================
        self.setpoint = setpoint
        self.delay = delay_of_action

    # ==================================================
    # IO FUNCTIONS
    # ==================================================
    def read_sensor(self, address=None):
        if address is None:
            raise ValueError("Sensor register address is missing")

        raw_value = self.modbus.analog_read(address=address)
        value = np.interp(
            raw_value,
            [self.min_value_remote_IO, self.max_value_remote_IO],
            [self.min_action, self.max_action],
        )
        return float(value)

    def write_actuator(self, address=None, action=None):
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

    # ==================================================
    # RESET
    # ==================================================
    def reset(self):
        # observed state from real sensor
        level = self.read_sensor(self.address_sensor)

        # randomize setpoint (same philosophy as sim)
        self.setpoint = np.random.uniform(self.min_action, self.max_action)

        # initialize action history
        self.prev_actions.clear()
        init_action = np.clip(level, self.min_action, self.max_action)

        for _ in range(self.action_history_len):
            self.prev_actions.append(init_action)

        # reset reward manager (TRUE observed state)
        self.reward_manager.reset(
            init_setpoint=self.setpoint,
            init_state=level,
            init_action=init_action,
        )

        state = np.array(
            [level] + list(self.prev_actions) + [self.setpoint],
            dtype=np.float32,
        )

        info = {"setpoint": self.setpoint}
        return state, info

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action):
        action = float(np.clip(action, self.min_action, self.max_action))

        # write action to actuator
        self.write_actuator(self.address_actuator, action)

        # physical / communication delay
        time.sleep(self.delay)

        # read observed level from sensor
        level = self.read_sensor(self.address_sensor)

        # update action history (KEY for canonical state)
        self.prev_actions.append(action)

        # build canonical state
        state = np.array(
            [level] + list(self.prev_actions) + [self.setpoint],
            dtype=np.float32,
        )

        # reward (same semantics as sim)
        error = abs(self.setpoint - level)
        self.reward_manager.update(
            setpoint=self.setpoint,
            state=level,
            action=action,
        )
        reward = self.reward_manager.reward_continuous_control()

        # termination (soft & safe for real system)
        done = error < 0.1

        info = {
            "error": error,
            "raw_level": level,
        }

        return state, reward, done, info
