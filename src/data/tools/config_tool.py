#my_project\src\data\tools\config_tool.py
import yaml
from pathlib import Path

# ------------------------------------------------------------
# FOLDER + FILENAME UTILITIES
# ------------------------------------------------------------
def ensure_folder(path):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def normalize_filename(name: str):
    name = str(name)
    if not name.lower().endswith(".yaml"):
        name += ".yaml"
    return name


# ------------------------------------------------------------
# CONFIG CREATION
# ------------------------------------------------------------
def generate_config(folder_path, filename="config.yaml"):
    folder = ensure_folder(folder_path)
    fname = normalize_filename(filename)
    full_path = folder / fname

    default_config = {
        "agent":{
        "target_keys": [
            "loss_actor", "loss_critic",
            "q1_mean", "q2_mean",
            "entropy", "alpha",
            "tau", "action",
        ],

        "color_map": {
            "loss_actor": "#ff0000",
            "loss_critic": "#0000ff",
            "q1_mean": "#00aa00",
            "q2_mean": "#aa00aa",
            "entropy": "#ffa500",
            "alpha": "#8b4513",
            "tau": "#ff69b4",
            "action": "#00ffff",
        },

        "plot": {
            "rows": 4,
            "cols": 2,
            "figsize": [5.8, 4.0],

        }
        },
        
        "episode": {
        "target_keys": [
            "state", "setpoint",
            "reward", "done",
            "episode", "action",
        ],

        "grouped_keys": [
            ["state", "setpoint"],
            ["reward"],
            ["action"],
        ],

        "color_map": {
            "episode": "red",
            "setpoint": "blue",
            "state": "green",
            "reward": "purple",
            "done": "orange",
            "action": "red"
        },

        "plot": {
            "rows": 2,
            "cols": 2,
            "figsize": [8, 5],
        }
        
    }
    }

    with open(full_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, allow_unicode=True, sort_keys=False)

    print(f"✔ Config created at: {full_path}")
    return full_path


# ------------------------------------------------------------
# CONFIG LOADER
# ------------------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    generate_config(folder_path=r"D:\Project_end\New_world\my_project\config",filename="config_color_SAC")