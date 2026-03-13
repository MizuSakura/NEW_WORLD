"""
Loader API
----------

Load and validate network configuration.

Responsibilities
----------------
- Read network.yaml
- Validate with NetworkConfig schema
- Provide singleton config object
"""

from pathlib import Path
import yaml

from schema_api import NetworkConfig


# -------------------------------------------------
# Config path
# -------------------------------------------------

CONFIG_PATH = Path(
    r"D:\Project_end\New_world\my_project\src\API\config\network.yaml"
)


# -------------------------------------------------
# Singleton storage
# -------------------------------------------------

_config_instance: NetworkConfig | None = None


# -------------------------------------------------
# Load config
# -------------------------------------------------

def load_config() -> NetworkConfig:
    """
    Load and validate network.yaml
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = NetworkConfig(**data)

    return config


# -------------------------------------------------
# Get config (singleton)
# -------------------------------------------------

def get_config() -> NetworkConfig:
    """
    Return shared config instance
    """

    global _config_instance

    if _config_instance is None:
        _config_instance = load_config()

    return _config_instance


# -------------------------------------------------
# Reload config
# -------------------------------------------------

def reload_config() -> NetworkConfig:
    """
    Reload config from file
    """

    global _config_instance

    _config_instance = load_config()

    return _config_instance


# -------------------------------------------------
# Main (debug / test)
# -------------------------------------------------

if __name__ == "__main__":

    print("=== Load Network Config ===")

    config = get_config()

    print("Device:", config.device.id)
    print("MQTT Broker:", config.mqtt.broker)
    print("MQTT Client:", config.mqtt.client_id)

    print("Control Topic:", config.topic("control"))
    print("Telemetry Topic:", config.topic("telemetry"))

    print("HTTP Port:", config.http.port)