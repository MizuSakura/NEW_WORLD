"""
RL Config Loader
----------------
Load and validate rl_params.yaml
Singleton pattern เหมือน loader_api.py
"""

from pathlib import Path
import yaml
from rl_schema import RLConfig

CONFIG_PATH = Path(
    r"D:\Project_end\New_world\my_project\src\API\config\rl_params.yaml"
)

_instance: RLConfig | None = None


def load_rl_config() -> RLConfig:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RLConfig(**data)


def get_rl_config() -> RLConfig:
    global _instance
    if _instance is None:
        _instance = load_rl_config()
    return _instance


def reload_rl_config() -> RLConfig:
    global _instance
    _instance = load_rl_config()
    return _instance


def save_rl_config(data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True,
                  default_flow_style=False)

if __name__ == "__main__":

    print("Loading RL config...")

    try:
        cfg = get_rl_config()

        print("\nConfig loaded successfully\n")
        print(cfg)

        print("\nConfig as dict\n")
        print(cfg.model_dump())

        print("\nTesting reload...")
        cfg2 = reload_rl_config()
        print("Reload OK")

    except Exception as e:
        print("Config ERROR:", e)