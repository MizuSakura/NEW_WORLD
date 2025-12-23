from src.utils.comucation_modbusTCP import ModbusTCP
from src.environment.reward_function_control import Reward_manager
import numpy as np
import time


class Real_env_remote:
    """
    Real Remote Environment with ModbusTCP
    Designed for real-world continuous control with RL.
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
    ):
        # ===== Communication =====
        self.ip_host = ip_host
        self.modbus = ModbusTCP(host=self.ip_host, port=port)
        self.modbus.connect()

        # ===== Reward Manager =====
        self.reward_manager = Reward_manager(buffer_size=5)

        # ===== Remote IO Range =====
        self.max_value_remote_IO = 27647
        self.min_value_remote_IO = 0
        self.address_sensor = address_sensor
        self.address_actuator = address_actuator

        # ===== Action Space =====
        self.action_dim = 1
        self.min_action = min_action
        self.max_action = max_action

        # ===== Observation Space =====
        self.state_dim = 3  # [level, action, setpoint]

        # ===== Internal =====
        self.setpoint = setpoint
        self.delay = delay_of_action
        self.prev_action = None

    def read_sensor(self,address=None):
    
        if address is None:
            raise ValueError(f"The register address is missing")
        raw_value = self.modbus.analog_read(address = address)
        value = np.interp(raw_value, [self.min_value_remote_IO, self.max_value_remote_IO], [self.min_action, self.max_action])

        return float(value)
    
    def write_actuator(self,address = None,action=None):
        if address is None:
            raise ValueError(f"The register address is missing")
        raw = int(np.interp(action,[self.min_action, self.max_action],[self.min_value_remote_IO, self.max_value_remote_IO]))

        self.modbus.write_holding_register(address= address,value= raw)

        return action
    
    def reset(self):
        level = self.read_sensor(self.address_sensor)
        self.setpoint = np.random.uniform(0,10)

        # assume actuator holds current state initially
        if self.prev_action is None:
            self.prev_action = np.clip(level, self.min_action, self.max_action)

        self.reward_manager.reset(init_setpoint=self.setpoint,init_state=level,init_action=self.prev_action,)

        state = np.array([level, self.prev_action, self.setpoint],dtype=np.float32,)
        info = {"setpoint": self.setpoint}

        return state,info
    
    def step(self, action):
        action = float(np.clip(action, self.min_action, self.max_action))

        # write to actuator
        self.write_actuator(self.address_actuator, action)

        # actuator delay
        time.sleep(self.delay)

        # read sensor
        level = self.read_sensor(self.address_sensor)

        # build state
        state = np.array(
            [level, action, self.setpoint],
            dtype=np.float32,
        )

        # reward
        error = abs(self.setpoint - level)
        self.reward_manager.update(self.setpoint, level, action)
        reward = self.reward_manager.reward_continuous_control()

        # termination condition (virtual)
        done = error < 0.1

        info = {
            "error": error,
            "raw_level": level,
        }

        self.prev_action = action

        return state, reward, done, info