"""
Control Config Loader
---------------------
Load and validate control_params.yaml
"""

from pathlib import Path
import yaml
from control_schema import ControlConfig

CONFIG_PATH = Path(
    r"D:\Project_end\New_world\my_project\src\API\config\control_params.yaml"
)

_instance: ControlConfig | None = None


def load_control_config() -> ControlConfig:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ControlConfig(**data)


def get_control_config() -> ControlConfig:
    global _instance
    if _instance is None:
        _instance = load_control_config()
    return _instance


def reload_control_config() -> ControlConfig:
    global _instance
    _instance = load_control_config()
    return _instance


def save_control_config(data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True,
                  default_flow_style=False)
        
if __name__ == "__main__":

    print("Loading Control Config...")

    try:
        cfg = get_control_config()

        print("\nConfig loaded successfully\n")
        print(cfg)

        print("\nConfig as dict\n")
        print(cfg.model_dump())

        print("\nTesting reload...")
        reload_control_config()
        print("Reload OK")

    except Exception as e:
        print("Config ERROR:", e)