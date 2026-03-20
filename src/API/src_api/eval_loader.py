"""
Eval Config Loader
------------------
Load and validate eval_params.yaml
"""

from pathlib import Path
import yaml
from eval_schema import EvalConfig

CONFIG_PATH = Path(
    r"D:\Project_end\New_world\my_project\src\API\config\eval_params.yaml"
)

_instance: EvalConfig | None = None


def load_eval_config() -> EvalConfig:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvalConfig(**data)


def get_eval_config() -> EvalConfig:
    global _instance
    if _instance is None:
        _instance = load_eval_config()
    return _instance


def reload_eval_config() -> EvalConfig:
    global _instance
    _instance = load_eval_config()
    return _instance


def save_eval_config(data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True,
                  default_flow_style=False)