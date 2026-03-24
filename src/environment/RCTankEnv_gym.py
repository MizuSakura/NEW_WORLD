# D:\Project_end\New_world\my_project\src\environment\RCTankEnv_gym.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from collections import deque
import os
from pathlib import Path
import yaml

from src.environment.noise_manager import NoiseManager
from src.environment.reward_function_control import Reward_manager
from src.environment.state_builder import StateBuilder

class RCTankEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        R=1.5,
        C=2.0,
        dt=0.1,
        control_mode="voltage",
        setpoint_level=5.0,
        level_max=10.0,
        max_action_volt=10.0,
        max_action_current=5.0,
        render_mode=None,
        save_episode_image: bool = True,
        save_dir: str = r"D:\Project_end\New_world\my_project\logs\image_log",
        noise_manager: NoiseManager = None,
        use_pid_state: bool = True,
        reward_type: str = "pid_adiaptive",      # เปลี่ยน default → pid_stable
        smooth_alpha: float = 0.0,            # NEW: low-pass filter บน action
                                               #      0.0 = ปิด (train), >0 = เปิด (deploy)
    ):
        super().__init__()
        self.R    = R
        self.C    = C
        self.dt   = dt
        self.mode = control_mode
        self.setpoint      = setpoint_level
        self.level_max     = level_max
        self.max_volt      = max_action_volt
        self.max_current   = max_action_current
        self.render_mode   = render_mode

        self.noise       = noise_manager
        self.use_pid_state = use_pid_state

        # ===== Action smoothing (low-pass filter) =====
        # ปิดตอน train (smooth_alpha=0.0) เปิดตอน deploy (เช่น 0.7)
        self.smooth_alpha      = smooth_alpha
        self._prev_action_smooth = 0.0

        # ===== StateBuilder =====
        _rl_cfg_path = Path(__file__).resolve().parents[2] / "src" / "API" / "config" / "rl_params.yaml"
        if _rl_cfg_path.exists():
            self.state_builder = StateBuilder.from_yaml(_rl_cfg_path)
        else:
            self.state_builder = StateBuilder({})

        self.level_history_len    = self.state_builder.level_hist_len
        self.action_history_len   = self.state_builder.action_hist_len
        self.setpoint_history_len = self.state_builder.setpoint_hist_len
        self.prev_error           = 0.0
        self.integral_error       = 0.0

        # ===== Observation Space =====
        action_high = self.max_volt if self.mode == "voltage" else self.max_current
        obs_low, obs_high = self.state_builder.obs_bounds(action_max=action_high)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # ===== Action Space =====
        if self.mode == "voltage":
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([self.max_volt], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([self.max_current], dtype=np.float32),
                dtype=np.float32,
            )

        # ===== Initial System States =====
        self.level         = 0.0
        self.time          = 0.0
        self.done          = 0.0
        self.reward_monitor = 0

        # ===== GUI =====
        self.render_scale = 3
        self.screen = None
        self.clock  = None
        self.width  = 900
        self.height = 470
        self.mode_plot = "fixed"

        # ===== Graph Data =====
        self.level_history  = []
        self.action_history = []
        self.reward_history = []

        self.save_episode_image = save_episode_image
        self.save_dir = save_dir
        self._init_episode_counter_from_folder()

        # ===== Reward Manager =====
        # แก้ไข: mode="raw" + fixed bounds แทน adaptive ที่ drift
        # max_delta_action = 50% ของ action range (สมเหตุสมผลสำหรับ RC tank)
        action_range = self.max_volt if self.mode == "voltage" else self.max_current
        self.reward_manager = Reward_manager(
            buffer_size      = 10,                  # เพิ่มจาก 5 → 10
            mode             = "raw",               # เปลี่ยนจาก "adaptive" → "raw"
            max_error        = self.level_max,      # 10.0
            max_delta_action = action_range * 0.5,  # 5.0 สำหรับ voltage mode
            max_delta_error  = self.level_max * 0.3 # 3.0
        )
        self.reward_type = reward_type

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.save_episode_image and len(self.reward_history) > 0:
            self._save_render_image()

        self.level    = float(self.np_random.uniform(0, self.level_max))
        self.setpoint = float(self.np_random.uniform(0, self.level_max))
        self.time     = 0.0
        self.done     = 0.0
        self.reward_monitor = 0

        # Reset PID state
        self.prev_error     = 0.0
        self.integral_error = 0.0

        # Reset action smoother
        self._prev_action_smooth = 0.0

        if self.noise is not None:
            self.noise.reset()

        self.level_history  = [self.level]
        self.action_history = []
        self.reward_history = []

        self.state_builder.reset(
            level    = self.level,
            action   = 0.0,
            setpoint = self.setpoint,
            dt       = self.dt,
        )
        obs = self.state_builder.update(
            level    = self.level,
            action   = 0.0,
            setpoint = self.setpoint,
        )

        self.reward_manager.reset(
            init_setpoint = self.setpoint,
            init_state    = self.level,
            init_action   = 0.0,
        )

        info = {"setpoint": self.setpoint}
        return obs, info

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):
        if isinstance(action, np.ndarray):
            action_val = float(action.item())
        else:
            action_val = float(action)

        # ===== Noise schedule step =====
        if self.noise is not None:
            self.noise.step()

        # ===== Action Noise =====
        if self.noise is not None:
            action_val = self.noise.apply_action_noise(action_val)

        # ===== Action Smoothing (low-pass filter) =====
        # ปิดตอน train (smooth_alpha=0.0) เปิดตอน deploy
        # replay buffer เห็น smoothed action → agent เรียนรู้ smooth policy โดยตรง
        if self.smooth_alpha > 0.0:
            action_val = (
                self.smooth_alpha * self._prev_action_smooth
                + (1.0 - self.smooth_alpha) * action_val
            )
        self._prev_action_smooth = action_val

        # ===== System Dynamics =====
        if self.mode == "voltage":
            action_val = np.clip(action_val, 0, self.max_volt)
            current    = (action_val - self.level) / self.R
            d_level    = (current / self.C) * self.dt
        else:
            action_val = np.clip(action_val, 0, self.max_current)
            net_flow   = action_val - (self.level / self.R)
            d_level    = (net_flow / self.C) * self.dt

        # ===== Process Noise =====
        if self.noise is not None:
            d_level = self.noise.apply_process_noise(d_level)

        self.level  = np.clip(self.level + d_level, 0, self.level_max)
        self.time  += self.dt

        # ===== PID State Computation =====
        error = self.setpoint - self.level
        error = np.clip(error, -self.level_max, self.level_max)

        # Integral (anti-windup)
        self.integral_error += error * self.dt
        self.integral_error  = np.clip(
            self.integral_error,
            -self.level_max,
            self.level_max
        )

        # Derivative
        derivative = (error - self.prev_error) / self.dt
        derivative = np.clip(derivative, -self.level_max, self.level_max)
        self.prev_error = error

        self.action_history.append(action_val)
        self.level_history.append(self.level)

        # ===== Observation (Sensor Noise) =====
        observed_level = self.level
        if self.noise is not None:
            observed_level = self.noise.apply_sensor_noise(observed_level)

        obs = self.state_builder.update(
            level    = observed_level,
            action   = action_val,
            setpoint = self.setpoint,
        )

        # ===== Reward =====
        self.reward_manager.update(
            setpoint = self.setpoint,
            state    = self.level,       # TRUE state (no noise)
            action   = action_val,
        )
        reward = self.reward_manager.reward_type(self.reward_type)

        self.reward_history.append(reward)
        self.reward_monitor = reward

        # ===== Termination =====
        # แก้ไข: ใช้ relative tolerance (3% ของ level_max) แทน absolute 0.05
        # level_max=10 → tolerance=0.3  (เหมาะสมกว่า 0.05 ที่แทบเป็นไปไม่ได้)
        abs_error = abs(self.setpoint - self.level)
        tolerance = 0.03 * self.level_max   # 3% → 0.3

        if abs_error < tolerance:
            self.done += self.dt
        else:
            self.done = 0.0

        terminated = self.done > 3.0   # ลดจาก 5s → 3s เพื่อให้ episode จบเร็วขึ้น
        truncated  = False
        info       = {"setpoint": self.setpoint, "abs_error": abs_error}

        return obs, reward, terminated, truncated, info

    # =====================================================
    # RENDER  (Enhanced with Dynamic Current Value Marker)
    # =====================================================
    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self._hires_surface = pygame.Surface((self.width * self.render_scale, self.height * self.render_scale))
            self.clock       = pygame.time.Clock()
            self.font_small  = pygame.font.SysFont("Arial", 12)
            self.font_medium = pygame.font.SysFont("Arial", 15)
            self.font_large  = pygame.font.SysFont("Arial", 18, bold=True)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

        self.screen.fill((245, 245, 245))

        current_reward    = self.reward_history[-1] if self.reward_history else 0.0
        cumulative_reward = np.sum(self.reward_history) if self.reward_history else 0.0

        header_items = [
            f"Level: {self.level:.2f}/{self.level_max:.1f}",
            f"Setpoint: {self.setpoint:.2f}",
            f"Action: {self.action_history[-1] if self.action_history else 0:.2f}",
            f"Time: {self.time:.1f}s",
            f"Reward: {current_reward:.3f}",
            f"Cumulative: {cumulative_reward:.2f}"
        ]

        spacing = 135
        for i, txt in enumerate(header_items):
            surf = self.font_large.render(txt, True, (30, 30, 30))
            self.screen.blit(surf, (20 + i * spacing, 10))

        tank_x, tank_y = 50, 90
        tank_w, tank_h = 160, 340

        pygame.draw.rect(self.screen, (60, 60, 60), (tank_x, tank_y, tank_w, tank_h), 3)

        title = self.font_medium.render("Water Tank", True, (20, 20, 20))
        self.screen.blit(title, (tank_x, tank_y - 28))

        water_ratio = np.clip(self.level / self.level_max, 0, 1)
        water_h     = tank_h * water_ratio

        pygame.draw.rect(
            self.screen,
            (50, 130, 255),
            (tank_x + 3, tank_y + tank_h - water_h, tank_w - 6, water_h)
        )

        sp_ratio = np.clip(self.setpoint / self.level_max, 0, 1)
        sp_y     = tank_y + tank_h - (tank_h * sp_ratio)
        pygame.draw.line(self.screen, (0, 200, 0), (tank_x, sp_y), (tank_x + tank_w, sp_y), 3)

        graph_x, graph_y = 270, 90
        graph_w, graph_h = 560, 340
        gap     = 25
        panel_h = (graph_h - 2 * gap) // 3

        level_y  = graph_y
        action_y = graph_y + panel_h + gap
        reward_y = graph_y + 2 * (panel_h + gap)

        self.screen.blit(self.font_medium.render("Level History",  True, (20, 20, 20)), (graph_x, level_y  - 25))
        self.screen.blit(self.font_medium.render("Action History", True, (20, 20, 20)), (graph_x, action_y - 25))
        self.screen.blit(self.font_medium.render("Reward History", True, (20, 20, 20)), (graph_x, reward_y - 25))

        pygame.draw.rect(self.screen, (80, 80, 80), (graph_x, level_y,  graph_w, panel_h), 2)
        pygame.draw.rect(self.screen, (80, 80, 80), (graph_x, action_y, graph_w, panel_h), 2)
        pygame.draw.rect(self.screen, (80, 80, 80), (graph_x, reward_y, graph_w, panel_h), 2)

        max_points = min(len(self.level_history), graph_w)

        if max_points > 1:
            lv = self.level_history[-max_points:]
            xs = [graph_x + i for i in range(len(lv))]
            ys = [level_y + panel_h - (v / self.level_max) * panel_h for v in lv]

            margin      = 0.02 * self.setpoint
            upper       = self.setpoint + margin
            lower       = self.setpoint - margin
            band_top    = level_y + panel_h - (upper / self.level_max) * panel_h
            band_bottom = level_y + panel_h - (lower / self.level_max) * panel_h

            pygame.draw.rect(self.screen, (255, 220, 220), (graph_x, band_top, graph_w, band_bottom - band_top))

            self._draw_time_grid(graph_x, level_y, graph_w, panel_h, len(lv), show_label=False)
            self._draw_grid(graph_x, level_y, graph_w, panel_h, self.level_max)

            sp_y_graph = level_y + panel_h - (self.setpoint / self.level_max) * panel_h
            pygame.draw.line(self.screen, (0, 180, 0), (graph_x, sp_y_graph), (graph_x + graph_w, sp_y_graph), 2)

            pygame.draw.lines(self.screen, (0, 70, 200), False, list(zip(xs, ys)), 2)

            last_x, last_y   = xs[-1], ys[-1]
            current_level    = lv[-1]

            pygame.draw.circle(self.screen, (0, 70, 200), (int(last_x), int(last_y)), 5)
            pygame.draw.line(self.screen, (160, 160, 160), (graph_x, last_y), (graph_x + graph_w, last_y), 1)
            pygame.draw.line(self.screen, (160, 160, 160), (last_x, level_y), (last_x, level_y + panel_h), 1)

            label      = self.font_small.render(f"{current_level:.2f}", True, (0, 70, 200))
            error_text = self.font_small.render(f"Error: {self.setpoint - self.level:.3f}", True, (200, 50, 50))
            self.screen.blit(error_text, (graph_x + graph_w + 5, last_y + 10))
            self.screen.blit(label,      (graph_x + graph_w + 10, last_y - 7))

        if len(self.action_history) > 1:
            act        = self.action_history[-max_points:]
            action_max = self.max_volt if self.mode == "voltage" else self.max_current

            self._draw_time_grid(graph_x, action_y, graph_w, panel_h, len(act), show_label=False)
            self._draw_grid(graph_x, action_y, graph_w, panel_h, action_max)

            xs = [graph_x + i for i in range(len(act))]
            ys = [action_y + panel_h - (a / action_max) * panel_h for a in act]

            pygame.draw.lines(self.screen, (200, 40, 40), False, list(zip(xs, ys)), 2)

            last_x, last_y   = xs[-1], ys[-1]
            current_action   = act[-1]

            pygame.draw.circle(self.screen, (200, 40, 40), (int(last_x), int(last_y)), 5)
            pygame.draw.line(self.screen, (150, 150, 150), (graph_x, last_y), (graph_x + graph_w, last_y), 1)
            pygame.draw.line(self.screen, (150, 150, 150), (last_x, action_y), (last_x, action_y + panel_h), 1)

            label = self.font_small.render(f"{current_action:.2f}", True, (200, 40, 40))
            self.screen.blit(label, (graph_x + graph_w + 5, last_y - 7))

        if len(self.reward_history) > 1:
            rw    = self.reward_history[-max_points:]
            r_min = min(rw)
            r_max = max(rw)

            if abs(r_max - r_min) < 1e-6:
                r_max += 1e-6

            self._draw_time_grid(graph_x, reward_y, graph_w, panel_h, len(rw), show_label=True)
            self._draw_grid(graph_x, reward_y, graph_w, panel_h, r_max, r_min, zero_line=True)

            xs = [graph_x + i for i in range(len(rw))]
            ys = [
                reward_y + panel_h - ((r - r_min) / (r_max - r_min)) * panel_h
                for r in rw
            ]

            pygame.draw.lines(self.screen, (120, 0, 160), False, list(zip(xs, ys)), 2)

            last_x         = xs[-1]
            last_y         = ys[-1]
            current_reward = rw[-1]
            cumulative_reward = np.sum(self.reward_history)

            pygame.draw.circle(self.screen, (120, 0, 160), (int(last_x), int(last_y)), 5)
            pygame.draw.line(self.screen, (160, 160, 160), (last_x, reward_y), (last_x, reward_y + panel_h), 1)

            reward_label = self.font_small.render(f"R: {current_reward:.3f}", True, (120, 0, 160))
            self.screen.blit(reward_label, (graph_x + graph_w + 5, last_y - 15))

            cum_label = self.font_small.render(f"CUM: {cumulative_reward:.2f}", True, (255, 140, 0))
            self.screen.blit(cum_label, (graph_x + graph_w + 5, last_y + 2))

        pygame.display.flip()

        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])
        else:
            array = pygame.surfarray.array3d(self.screen)
            return np.transpose(array, (1, 0, 2))

    def _draw_grid(self, x, y, w, h, y_max, y_min=0, zero_line=False):
        for i in range(6):
            yy  = y + h - i * (h / 5)
            pygame.draw.line(self.screen, (220, 220, 220), (x, yy), (x + w, yy), 1)
            val = y_min + (i / 5) * (y_max - y_min)
            txt = self.font_small.render(f"{val:.2f}", True, (100, 100, 100))
            self.screen.blit(txt, (x - 45, yy - 7))

        if zero_line and y_min < 0 < y_max:
            zero_y = y + h - ((0 - y_min) / (y_max - y_min)) * h
            pygame.draw.line(self.screen, (150, 150, 150), (x, zero_y), (x + w, zero_y), 1)

    def _draw_time_grid(self, x, y, w, h, history_len, show_label=True, fixed_interval=2.0):
        if history_len < 2:
            return

        mode = self.mode_plot
        if mode == "fixed":
            interval = fixed_interval
        else:
            total_time = history_len * self.dt
            if total_time <= 10:
                interval = 1.0
            elif total_time <= 30:
                interval = 2.0
            elif total_time <= 60:
                interval = 5.0
            else:
                interval = 10.0

        steps_per_mark = max(1, int(interval / self.dt))

        for i in range(history_len):
            x_pos = x + i
            if i % steps_per_mark == 0:
                pygame.draw.line(self.screen, (205, 205, 205), (x_pos, y), (x_pos, y + h), 1)
                if show_label:
                    time_sec = i * self.dt
                    label    = self.font_small.render(f"{time_sec:.0f}s", True, (110, 110, 110))
                    text_rect = label.get_rect()
                    text_rect.center = (x_pos, y + h + 18)
                    self.screen.blit(label, text_rect)

    def _init_episode_counter_from_folder(self):
        """นับ episode จำนวนจากไฟล์ที่มีอยู่ใน save_dir"""
        self._episode_counter = 0
        if self.save_episode_image:
            save_path = Path(self.save_dir)
            if save_path.exists():
                existing = list(save_path.glob("episode_*.png"))
                self._episode_counter = len(existing)

    def _save_render_image(self):
        """บันทึก frame ปัจจุบันเป็น PNG"""
        if self.screen is None:
            return
        save_path = Path(self.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename  = save_path / f"episode_{self._episode_counter:05d}.png"
        pygame.image.save(self.screen, str(filename))
        self._episode_counter += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None