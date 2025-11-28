import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import re
from datetime import datetime


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_filename(base_name: str = None, ext=".parquet"):
    """Return filename, use timestamp if not provided."""
    if not base_name:
        ts = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        return f"{ts}{ext}"

    if not base_name.endswith(ext):
        return base_name + ext
    return base_name


def next_indexed_name(folder: Path, base_name: str):
    """Return auto-increment filename."""
    ensure_folder(folder)

    stem = Path(base_name).stem
    suffix = ".parquet"

    patt = re.compile(rf"{stem}_(\d+){suffix}")
    max_idx = 0

    for file in folder.iterdir():
        m = patt.fullmatch(file.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    return f"{stem}_{max_idx + 1}{suffix}"


# ---------------------------------------------------------------------
# Base Plugin Class
# ---------------------------------------------------------------------
class BaseLoggerPlugin:
    """Every logger plugin must implement:
       - log(...)
       - save()
       - clear()
    """

    def log(self, *args, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


# ---------------------------------------------------------------------
# Metric Logger Plugin
# ---------------------------------------------------------------------
class MetricLogger(BaseLoggerPlugin):

    def __init__(self, folder="logs/metric/", filename=None, auto_increment=True):
        self.folder = Path(folder) if folder else None
        self.filename = filename
        self.auto_increment = auto_increment

        if self.folder:
            ensure_folder(self.folder)

        self.buffer = {
            "timestamp": [],
            "key": [],
            "value": [],
        }

    def log(self, key, value):
        self.buffer["timestamp"].append(time.time())
        self.buffer["key"].append(str(key))
        self.buffer["value"].append(float(value))

    def _make_path(self):
        name = build_filename(self.filename)
        if self.auto_increment:
            name = next_indexed_name(self.folder, name.replace(".parquet", ""))
        return self.folder / name

    def save(self):
        if not self.folder:
            return

        if len(self.buffer["timestamp"]) == 0:
            return

        path = self._make_path()
        table = pa.Table.from_pydict(self.buffer)
        pq.write_table(table, path)
        print("[MetricLogger] Saved →", path)

    def clear(self):
        for k in self.buffer:
            self.buffer[k].clear()


# ---------------------------------------------------------------------
# Episode Logger Plugin
# ---------------------------------------------------------------------
class EpisodeLogger(BaseLoggerPlugin):

    def __init__(self, folder="logs/episode/", filename=None, auto_increment=True):
        if folder is None:
            raise ValueError("Episode logger folder must be provided.")

        self.folder = Path(folder)
        self.filename = filename
        self.auto_increment = auto_increment

        ensure_folder(self.folder)

        self.buffer = {
            "episode": [],
            "setpoint": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "done": [],
        }

    def log(self, episode, setpoint, step, state, action, reward, next_state, done):
        self.buffer["episode"].append(episode)
        self.buffer["setpoint"].append(setpoint)
        self.buffer["step"].append(step)
        self.buffer["state"].append(state)
        self.buffer["action"].append(action)
        self.buffer["reward"].append(float(reward))
        self.buffer["next_state"].append(next_state)
        self.buffer["done"].append(bool(done))

    def _make_path(self):
        name = build_filename(self.filename)
        if self.auto_increment:
            name = next_indexed_name(self.folder, name.replace(".parquet", ""))
        return self.folder / name

    def save(self):
        if len(self.buffer["episode"]) == 0:
            return

        path = self._make_path()
        table = pa.Table.from_pydict(self.buffer)
        pq.write_table(table, path)
        print("[EpisodeLogger] Saved →", path)

    def clear(self):
        for k in self.buffer:
            self.buffer[k].clear()


# ---------------------------------------------------------------------
# Logger Manager → Combine plugins
# ---------------------------------------------------------------------
# class ArrowLoggerManager:
#     """
#     Combine multiple loggers (plugins) but keep them independent.
#     Example:
#         manager = ArrowLoggerManager(
#             metric=MetricLogger(...),
#             episode=EpisodeLogger(...)
#         )
#     """

#     def __init__(self, **plugins):
#         self.plugins = plugins  # e.g. {"metric": MetricLogger(), "episode": EpisodeLogger()}

#     def save(self):
#         for name, plugin in self.plugins.items():
#             plugin.save()

#     def clear(self):
#         for name, plugin in self.plugins.items():
#             plugin.clear()

#     def __getitem__(self, key):
#         return self.plugins[key]
