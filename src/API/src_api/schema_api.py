"""
Schema API
----------

Define configuration schema for the network system.
Validate configuration loaded from network.yaml.
"""

from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import yaml


# -------------------------------------------------
# Device
# -------------------------------------------------

class DeviceConfig(BaseModel):
    id: str = Field(..., description="Device unique ID")
    location: str = Field(..., description="Device location")


# -------------------------------------------------
# MQTT
# -------------------------------------------------

class MQTTTopics(BaseModel):
    control: str
    state: str
    telemetry: str
    emergency: str
    hardware: str = "project/rl/nvidia01/hardware"
    mode: str = "project/rl/nvidia01/mode"
    action:    str = "project/rl/nvidia01/action"      
    heartbeat: str = "project/rl/nvidia01/heartbeat"   


class MQTTConfig(BaseModel):
    broker: str
    port: int = 1883
    keepalive: int = 60
    client_id: str
    username: Optional[str] = None
    password: Optional[str] = None
    qos: int = Field(1, ge=0, le=2)
    topics: MQTTTopics
    emergency_qos: int = 2


# -------------------------------------------------
# HTTP
# -------------------------------------------------

class HTTPConfig(BaseModel):
    enable: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


# -------------------------------------------------
# System
# -------------------------------------------------

class SystemConfig(BaseModel):
    mode: str = Field(..., description="RL / Manual")
    allow_manual_override: bool = True
    heartbeat_interval: int = 2
    emergency_stop: bool = False


# -------------------------------------------------
# Server
# -------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    upload_model: bool = True
    upload_log: bool = True


# -------------------------------------------------
# Training
# -------------------------------------------------

class TrainingConfig(BaseModel):
    mode: str = "offline"       # offline / online / transfer / hybrid
    algorithm: str = "SAC"
    episodes: int = 10000
    max_steps: int = 200
    batch_size: int = 1080
    auto_save_every: int = 1
    checkpoint_path: str = "models/checkpoint/Autosave.pt"
    final_model_path: str = "models/Test_history.pt"


# -------------------------------------------------
# SAC
# -------------------------------------------------

class ActorConfig(BaseModel):
    layers: int = 2
    hidden: int = 256

class CriticConfig(BaseModel):
    layers: int = 2
    hidden: int = 256
    encoder: bool = False

class SACConfig(BaseModel):
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.4
    actor: ActorConfig = ActorConfig()
    critic: CriticConfig = CriticConfig()


# -------------------------------------------------
# Noise
# -------------------------------------------------

class OUNoiseConfig(BaseModel):
    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.25
    dt: float = 0.1

class GaussianConfig(BaseModel):
    sigma: float = 0.02

class SchedulerConfig(BaseModel):
    peak: int = 3000
    std: float = 1500.0
    max_scale: float = 1.0

class NoiseConfig(BaseModel):
    enabled: bool = True
    ou_noise: OUNoiseConfig = OUNoiseConfig()
    gaussian: GaussianConfig = GaussianConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


# -------------------------------------------------
# Hardware
# -------------------------------------------------

class HardwareConfig(BaseModel):
    modbus_host: str = "192.168.1.20"
    modbus_port: int = 502


# -------------------------------------------------
# Root Config
# -------------------------------------------------

class NetworkConfig(BaseModel):
    device: DeviceConfig
    mqtt: MQTTConfig
    http: HTTPConfig
    system: SystemConfig
    server: ServerConfig = ServerConfig()
    training: TrainingConfig = TrainingConfig()
    sac: SACConfig = SACConfig()
    noise: NoiseConfig = NoiseConfig()
    hardware: HardwareConfig = HardwareConfig()

    def topic(self, name: str) -> str:
        return getattr(self.mqtt.topics, name)

    def topic_full(self, name: str) -> str:
        return f"{self.device.id}/{self.topic(name)}"


# -------------------------------------------------
# Main (Test schema validation)
# -------------------------------------------------

if __name__ == "__main__":

    print("=== Test NetworkConfig Schema ===")

    config_path = Path(
        r"D:\Project_end\New_world\my_project\src\API\config\network.yaml"
    )

    if not config_path.exists():
        print("network.yaml not found")
        exit()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = NetworkConfig(**data)

    print("Config validation success\n")
    print("Device ID      :", config.device.id)
    print("MQTT Broker    :", config.mqtt.broker)
    print("Training Mode  :", config.training.mode)
    print("SAC lr         :", config.sac.learning_rate)
    print("SAC episodes   :", config.training.episodes)
    print("Noise enabled  :", config.noise.enabled)
    print("Hardware host  :", config.hardware.modbus_host)