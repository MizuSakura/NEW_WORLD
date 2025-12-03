#my_project\src\data\analysis_data.py
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import math

from src.data.tools.config_tool import load_config, ensure_folder


class ParquetPlotter:
    """
    Unified Plotter:
    - Auto-detects file type: Agent / Episode
    - Plots single file or folder
    - Supports auto/manual layout for subplot
    """

    DEFAULT_COLOR_MAP = {
        "episode": "red",
        "setpoint": "blue",
        "state": "green",
        "reward": "purple",
        "done": "orange",
        "action": "red"
    }

    def __init__(self, agent_cfg=None, episode_cfg=None):
        self.agent_cfg = agent_cfg or {}
        self.episode_cfg = episode_cfg or {}

    @staticmethod
    def from_yaml(path):
        cfg = load_config(path)
        if "agent" not in cfg and "episode" not in cfg:
            raise KeyError("Config must contain 'agent' or 'episode' section.")
        return ParquetPlotter(agent_cfg=cfg.get("agent", {}), episode_cfg=cfg.get("episode", {}))

    def plot(self, path, save_path=None, layout_mode="auto", rows=None, cols=None):
        path = Path(path)
        if save_path:
            save_path = Path(save_path)
            ensure_folder(save_path.parent)
        if not path.exists():
            raise ValueError(f"Path not found: {path}")

        if path.is_file() and path.suffix == ".parquet":
            df = pq.read_table(path).to_pandas()
            mode = self._detect_format(df)
            print(f"[DETECT] File format detected: {mode}")
            if mode == "agent":
                self._plot_agent(df, save_path, layout_mode, rows, cols)
            elif mode == "episode":
                self._plot_episode(df, save_path, layout_mode, rows, cols)
            else:
                raise ValueError("Unknown parquet structure.")
        elif path.is_dir():
            sample_file = next(path.glob("*.parquet"), None)
            if sample_file is None:
                print("No parquet files found.")
                return
            df = pq.read_table(sample_file).to_pandas()
            mode = self._detect_format(df)
            print(f"[DETECT] Folder format detected: {mode}")
            if mode == "agent":
                self._plot_agent_folder(path, save_path, layout_mode, rows, cols)
            elif mode == "episode":
                self._plot_episode_folder(path, save_path, layout_mode, rows, cols)

    def _detect_format(self, df):
        if {"key", "value"}.issubset(df.columns):
            return "agent"
        if {"state", "action", "reward", "done", "episode"}.intersection(df.columns):
            return "episode"
        return "unknown"

    def _plot_agent(self, df, save_path, layout_mode="manual", rows=None, cols=None):
        target_keys = self.agent_cfg["target_keys"]
        color_map = self.agent_cfg.get("color_map", {})
        total_plots = len(target_keys)
        if layout_mode == "auto":
            cols = math.ceil(math.sqrt(total_plots))
            rows = math.ceil(total_plots / cols)
        elif layout_mode == "manual":
            rows = rows or 4
            cols = cols or 2
        figsize = tuple(self.agent_cfg.get("plot", {}).get("figsize", (10, 6)))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for ax, key in zip(axes, target_keys):
            if key not in df["key"].unique():
                ax.text(0.5, 0.5, f"{key} (not found)", ha="center")
                ax.set_title(key)
                ax.grid(True)
                continue
            subset = df[df["key"] == key].reset_index(drop=True)
            x = subset.index
            y = subset["value"]
            color = color_map.get(key, "black")
            ax.plot(x, y, color=color, linewidth=1.3, label=key)
            ax.set_title(key)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True)

        for i in range(len(target_keys), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def _plot_agent_folder(self, folder_path, save_path, layout_mode="manual", rows=None, cols=None):
        folder = Path(folder_path)
        files = sorted(folder.glob("*.parquet"))
        if not files:
            print("No parquet files found.")
            return
        df_list = [pq.read_table(f).to_pandas() for f in files if not pq.read_table(f).to_pandas().empty]
        df = pd.concat(df_list, ignore_index=True)
        self._plot_agent(df, save_path, layout_mode, rows, cols)

    def _plot_episode(self, df, save_path, layout_mode="auto", rows=None, cols=None):
        grouped_keys = self.episode_cfg.get("grouped_keys", [["state", "setpoint"], ["reward"], ["action"]])
        color_map = self.episode_cfg.get("color_map", self.DEFAULT_COLOR_MAP)
        plot_keys = []
        for gk in grouped_keys:
            for k in gk:
                if k in df.columns:
                    if k == "state" and isinstance(df[k].iloc[0], (list, np.ndarray)):
                        for dim in range(len(df[k].iloc[0])):
                            plot_keys.append((k, dim))
                    else:
                        plot_keys.append((k, None))
        total_plots = len(plot_keys)
        if layout_mode == "auto":
            cols = math.ceil(math.sqrt(total_plots))
            rows = math.ceil(total_plots / cols)
        elif layout_mode == "manual":
            rows = rows or 2
            cols = cols or 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
        axes = axes.flatten()
        x = df.index

        ep_text = ""
        if "episode" in df.columns:
            ep_text += f"Episode: {df['episode'].iloc[-1]}   "
        if "done" in df.columns:
            ep_text += f"Done: {df['done'].iloc[-1]}"
        fig.suptitle(ep_text.strip(), fontsize=13, fontweight="bold")

        for ax, (key, dim) in zip(axes, plot_keys):
            y = df[key]
            color = color_map.get(key, "black")
            if dim is not None:
                y_arr = np.vstack(y.to_numpy())[:, dim]
                ax.plot(x, y_arr, color=color, linewidth=1.4, label=f"{key}[{dim}]")
            else:
                if np.issubdtype(y.dtype, np.number):
                    ax.plot(x, y, color=color, linewidth=1.5, label=key)
                elif isinstance(y.iloc[0], (list, np.ndarray)):
                    y_arr = np.vstack(y.to_numpy())
                    for d in range(y_arr.shape[1]):
                        ax.plot(x, y_arr[:, d], color=color, alpha=0.7, linewidth=1.4, label=f"{key}[{d}]")
            ax.set_title(f"{key}" if dim is None else f"{key}[{dim}]")
            ax.grid(True)
            ax.legend(fontsize=8, loc="upper right")

        for i in range(len(plot_keys), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def _plot_episode_folder(self, folder_path, save_path, layout_mode="auto", rows=None, cols=None):
        folder = Path(folder_path)
        files = sorted(folder.glob("*.parquet"))
        if not files:
            print("No parquet files found.")
            return

        all_states, all_setpoints, all_rewards, all_actions = [], [], [], []
        for f in files:
            df = pq.read_table(f).to_pandas()
            if df.empty:
                continue
            if "state" in df.columns:
                last_state = df["state"].iloc[-1]
                if isinstance(last_state, (list, np.ndarray)):
                    all_states.append(last_state)
            if "setpoint" in df.columns:
                all_setpoints.append(df["setpoint"].iloc[-1])
            if "reward" in df.columns:
                all_rewards.append(df["reward"].max())
            if "action" in df.columns:
                if np.issubdtype(df["action"].dtype, np.number):
                    all_actions.extend(df["action"].to_numpy())
                else:
                    all_actions.extend(np.vstack(df["action"].to_numpy()))

        all_states = np.array(all_states)
        all_setpoints = np.array(all_setpoints)
        all_rewards = np.array(all_rewards)
        all_actions = np.array(all_actions)

        total_plot = all_states.shape[1] + 3 if all_states.size else 3
        if layout_mode == "auto":
            cols = math.ceil(math.sqrt(total_plot))
            rows = math.ceil(total_plot / cols)
        elif layout_mode == "manual":
            rows = rows or 2
            cols = cols or 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
        axes = axes.flatten()
        idx = 0

        for d in range(all_states.shape[1]):
            axes[idx].hist(all_states[:, d], bins=50, color=self.DEFAULT_COLOR_MAP.get("state", "green"), alpha=0.7)
            axes[idx].set_title(f"State[{d}] Distribution")
            axes[idx].grid(True)
            idx += 1

        if len(all_setpoints):
            axes[idx].hist(all_setpoints, bins=50, color=self.DEFAULT_COLOR_MAP.get("setpoint", "blue"), alpha=0.7)
            axes[idx].set_title("Setpoint Distribution")
            axes[idx].grid(True)
            idx += 1

        if len(all_rewards):
            axes[idx].hist(all_rewards, bins=50, color=self.DEFAULT_COLOR_MAP.get("reward", "purple"), alpha=0.7)
            axes[idx].set_title("Reward Distribution")
            axes[idx].grid(True)
            idx += 1

        if len(all_actions):
            if all_actions.ndim == 1:
                axes[idx].hist(all_actions, bins=50, color=self.DEFAULT_COLOR_MAP.get("action", "red"), alpha=0.7)
            else:
                for d in range(all_actions.shape[1]):
                    axes[idx].hist(all_actions[:, d], bins=50, color=self.DEFAULT_COLOR_MAP.get("action", "red"), alpha=0.7)
                    axes[idx].set_title(f"Action[{d}] Distribution")
                    idx += 1

        for i in range(idx, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    FILE_CONFIG = BASE_DIR / "config" / "config_color_SAC.yaml"
    FILE_OR_FOLDER_DATA = r"D:\Project_end\New_world\my_project\logs\agent\RC_Tank\optimized__1.parquet"
    LAOUT_MODE = "auto" # "manual" , "auto"
    ROWS =  2
    COLUMNS = 2

    MODE_SAVE_PICTURE = True
    if MODE_SAVE_PICTURE:
        FOLDER_SAVE_PICTRUE = BASE_DIR / "data" / "picture" / "agent_pic" / "agent_SAC"
    else:
        FOLDER_SAVE_PICTRUE = None

    plotter = ParquetPlotter.from_yaml(FILE_CONFIG)
    plotter.plot(FILE_OR_FOLDER_DATA, save_path=FOLDER_SAVE_PICTRUE, layout_mode=LAOUT_MODE, rows= ROWS, cols= COLUMNS)
