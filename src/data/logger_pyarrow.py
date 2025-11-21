#my_project\src\data\logger_pyarrow.py
import pyarrow as pa
import pyarrow.parquet as pq
import time
from pathlib import Path
import re


def next_indexed_file(base_path: Path):
    """
    Given base_path = logs/metric.parquet
    return new path = logs/metric_1.parquet, metric_2.parquet, ...

    If no numbered file exists → start at _1
    """
    folder = base_path.parent
    stem = base_path.stem    # metric
    suffix = base_path.suffix  # .parquet

    folder.mkdir(parents=True, exist_ok=True)

    # Find existing files
    pattern = re.compile(rf"{stem}_(\d+){suffix}")
    max_index = 0

    for f in folder.iterdir():
        m = pattern.fullmatch(f.name)
        if m:
            idx = int(m.group(1))
            max_index = max(max_index, idx)

    # Create next filename
    next_file = folder / f"{stem}_{max_index + 1}{suffix}"
    return next_file


class ArrowLogger:
    """
    Simple logger with auto-incrementing Parquet filenames.
    """

    def __init__(self,
                 metric_base="logs/metric.parquet",
                 episode_base="logs/episode.parquet"):

        self.metric_base = Path(metric_base)
        self.episode_base = Path(episode_base)

        self.metric_buffer = {
            "timestamp": [],
            "key": [],
            "value": [],
        }

        self.episode_buffer = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "done": [],
        }

    # ------------------------------------------------------------
    # Metric logger
    # ------------------------------------------------------------
    def log_metric(self, key, value):
        self.metric_buffer["timestamp"].append(time.time())
        self.metric_buffer["key"].append(str(key))
        self.metric_buffer["value"].append(float(value))

    # ------------------------------------------------------------
    # Episode logger
    # ------------------------------------------------------------
    def log_episode(self, episode, step, state, action, reward, next_state, done):
        self.episode_buffer["episode"].append(episode)
        self.episode_buffer["step"].append(step)
        self.episode_buffer["state"].append(state)
        self.episode_buffer["action"].append(action)
        self.episode_buffer["reward"].append(float(reward))
        self.episode_buffer["next_state"].append(next_state)
        self.episode_buffer["done"].append(bool(done))

    # ------------------------------------------------------------
    # Save with auto-increment filenames
    # ------------------------------------------------------------
    def save(self):
        # ---------------- Metric ----------------
        if len(self.metric_buffer["timestamp"]) > 0:
            path = next_indexed_file(self.metric_base)
            table = pa.Table.from_pydict(self.metric_buffer)
            pq.write_table(table, path)
            print("[Logger] saved metric →", path)

        # ---------------- Episode ----------------
        if len(self.episode_buffer["episode"]) > 0:
            path = next_indexed_file(self.episode_base)
            table = pa.Table.from_pydict(self.episode_buffer)
            pq.write_table(table, path)
            print("[Logger] saved episode →", path)

    # ------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------
    def clear(self):
        for k in self.metric_buffer:
            self.metric_buffer[k].clear()
        for k in self.episode_buffer:
            self.episode_buffer[k].clear()
