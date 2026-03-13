#my_project\src\API\src_api\create_default_configs.py
"""
Create Default Network Config
-----------------------------

Generate default configuration file for network communication
including device identity, MQTT broker, HTTP server, and system control.

This script is intended to be used by Front-end tools that
inject configuration values before generating YAML.

Usage
-----

python create_default_configs.py
"""

import yaml
from pathlib import Path


# -------------------------------------------------
# YAML writer
# -------------------------------------------------

def write_yaml(data: dict, path: Path):
    """
    Write dictionary data to YAML file
    """

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True
        )


# -------------------------------------------------
# Create config file
# -------------------------------------------------

def create_network_config(config_data: dict, config_dir: Path):
    """
    Create network.yaml configuration file
    """

    config_dir.mkdir(parents=True, exist_ok=True)

    network_path = config_dir / "network.yaml"

    write_yaml(config_data, network_path)

    print("✅ Network config created successfully")
    print(f"📄 Path : {network_path}")


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    print("=== Create Default Network Config ===")

    # -------------------------------------------------
    # Default Config (Front-end can modify here)
    # -------------------------------------------------

    NETWORK_CONFIG = {

        "device": {
            "id": "jetson_nvidia01",
            "location": "lab_rc_system"
        },

        "mqtt": {
            "broker": "192.168.1.10",
            "port": 1883,
            "keepalive": 60,

            "client_id": "jetson_nvidia01",
            "username": "",
            "password": "",

            "qos": 1,

            "topics": {
                "control": "project/rl/nvidia01/control",
                "state": "project/rl/nvidia01/state",
                "telemetry": "project/rl/nvidia01/telemetry",
                "emergency": "project/rl/nvidia01/emergency"
            }
        },

        "http": {
            "enable": True,
            "host": "0.0.0.0",
            "port": 8000
        },

        "system": {
            "mode": "RL",
            "allow_manual_override": True,
            "heartbeat_interval": 2
        }

    }

    # -------------------------------------------------
    # Config directory
    # -------------------------------------------------

    CONFIG_DIR = Path(
        r"D:\Project_end\New_world\my_project\src\API\config"
    )

    # -------------------------------------------------
    # Create file
    # -------------------------------------------------

    create_network_config(NETWORK_CONFIG, CONFIG_DIR)