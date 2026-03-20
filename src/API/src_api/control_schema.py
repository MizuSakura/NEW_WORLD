"""
Control Parameters Schema
-------------------------
Validate control_params.yaml
"""

from pydantic import BaseModel, Field
from pathlib import Path
import yaml


# -------------------------------------------------
# Manual — Modbus
# -------------------------------------------------

class ModbusConfig(BaseModel):
    host: str = "192.168.1.100"
    port: int = 502
    device_id: int = 1

class OutputConfig(BaseModel):
    register_address: int = 1025
    coil_address: str = "0x4000"
    value_min: int = 0
    value_max: int = 4095

class PWMConfig(BaseModel):
    frequency: float = 0.1
    duty_min: float = 0.0
    duty_max: float = 1.0

class ManualConfig(BaseModel):
    modbus: ModbusConfig = ModbusConfig()
    output: OutputConfig = OutputConfig()
    pwm: PWMConfig = PWMConfig()


# -------------------------------------------------
# PID
# -------------------------------------------------

class PIDConfig(BaseModel):
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    setpoint: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    sample_time: float = 0.1


# -------------------------------------------------
# Emergency
# -------------------------------------------------

class EmergencyConfig(BaseModel):
    coil_range_start: str = "0x4000"
    coil_range_end: str = "0x40FF"
    register_range_start: int = 0
    register_range_end: int = 200
    reset_value: int = 0


# -------------------------------------------------
# Root
# -------------------------------------------------

class ControlConfig(BaseModel):
    manual: ManualConfig = ManualConfig()
    pid: PIDConfig = PIDConfig()
    emergency: EmergencyConfig = EmergencyConfig()


# -------------------------------------------------
# Main (Test)
# -------------------------------------------------

if __name__ == "__main__":
    config_path = Path(
        r"D:\Project_end\New_world\my_project\src\API\config\control_params.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = ControlConfig(**data)

    print("ControlConfig validation success")
    print("Modbus host    :", config.manual.modbus.host)
    print("Register addr  :", config.manual.output.register_address)
    print("PID Kp         :", config.pid.kp)
    print("Emergency end  :", config.emergency.coil_range_end)