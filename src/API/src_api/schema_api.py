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
# MQTT Topics
# -------------------------------------------------

class MQTTTopics(BaseModel):

    control: str
    state: str
    telemetry: str
    emergency: str


# -------------------------------------------------
# MQTT Config
# -------------------------------------------------

class MQTTConfig(BaseModel):

    broker: str
    port: int = 1883

    keepalive: int = 60

    client_id: str

    username: Optional[str] = None
    password: Optional[str] = None

    qos: int = Field(1, ge=0, le=2)

    topics: MQTTTopics


# -------------------------------------------------
# HTTP Config
# -------------------------------------------------

class HTTPConfig(BaseModel):

    enable: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


# -------------------------------------------------
# System Config
# -------------------------------------------------

class SystemConfig(BaseModel):

    mode: str = Field(..., description="System mode RL / PID / MANUAL")
    allow_manual_override: bool = True
    heartbeat_interval: int = 2


# -------------------------------------------------
# Root Config
# -------------------------------------------------

class NetworkConfig(BaseModel):

    device: DeviceConfig
    mqtt: MQTTConfig
    http: HTTPConfig
    system: SystemConfig

    def topic(self, name: str):
        return getattr(self.mqtt.topics, name)
    
    def topic_full(self, name: str):
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
        print("❌ network.yaml not found")
        exit()

    # load yaml
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # validate schema
    config = NetworkConfig(**data)

    print("✅ Config validation success\n")

    print("Device ID:", config.device.id)
    print("MQTT Broker:", config.mqtt.broker)
    print("MQTT Client ID:", config.mqtt.client_id)
    print("HTTP Port:", config.http.port)
    print("System Mode:", config.system.mode)