# hardware/src/utils/hw_config_loader.py
import yaml
from pathlib import Path
from functools import lru_cache

HW_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "hardware.yaml"

@lru_cache(maxsize=1)
def get_hw_config() -> dict:
    with open(HW_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def reload_hw_config() -> dict:
    get_hw_config.cache_clear()
    return get_hw_config()