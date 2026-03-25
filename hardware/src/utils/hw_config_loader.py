#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/utils/hw_config_loader.py
"""
แก้ไขให้ compatible กับ Python 3.6.9, PyYAML 3.12

PyYAML 3.12: yaml.safe_load ใช้ได้ปกติ
functools.lru_cache ใช้ได้ตั้งแต่ Python 3.2
"""

from __future__ import print_function

import yaml
from pathlib import Path
from functools import lru_cache

HW_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "hardware.yaml"


@lru_cache(maxsize=1)
def get_hw_config():
    # type: () -> dict
    with open(str(HW_CONFIG_PATH), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reload_hw_config():
    # type: () -> dict
    get_hw_config.cache_clear()
    return get_hw_config()

if __name__ == "__main__":
    print("--- Testing Hardware Config Loader ---")
    cfg = get_hw_config()
    if cfg:
        print("Status: Success")
        print("Data: {}".format(cfg))
    else:
        print("Status: Failed or Empty")
