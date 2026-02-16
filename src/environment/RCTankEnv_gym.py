# D:\Project_end\New_world\my_project\src\environment\RCTankEnv_gym.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from collections import deque

from src.environment.noise_manager import NoiseManager
from src.environment.reward_function_control import Reward_manager

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
        noise_manager: NoiseManager = None,   
    ):
        super().__init__()
        self.R = R
        self.C = C
        self.dt = dt
        self.mode = control_mode
        self.setpoint = setpoint_level
        self.level_max = level_max
        self.max_volt = max_action_volt
        self.max_current = max_action_current
        self.render_mode = render_mode

        # ===== Noise =====
        self.noise = noise_manager

        # ===== Action history for STATE ONLY =====
        self.action_history_len = 10
        self.prev_actions = deque(maxlen=self.action_history_len)

        # ===== Observation Space =====
        action_high = self.max_volt if self.mode == "voltage" else self.max_current

        obs_low = np.array(
            [0.0] + [0.0] * self.action_history_len + [0.0],
            dtype=np.float32
        )
        obs_high = np.array(
            [self.level_max]
            + [action_high] * self.action_history_len
            + [self.level_max],
            dtype=np.float32
        )

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
        self.level = 0.0
        self.time = 0.0
        self.done = 0.0

        # ===== GUI =====
        self.screen = None
        self.clock = None
        self.width = 800
        self.height = 450

        # ===== Graph Data =====
        self.level_history = []
        self.action_history = []

         # -----------------------------
        # Reward Manager
        # -----------------------------
        self.reward_manager = Reward_manager(buffer_size=5)

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = float(self.np_random.uniform(0, self.level_max))
        self.setpoint = float(self.np_random.uniform(0, self.level_max))
        self.time = 0.0
        self.done = 0.0

        if self.noise is not None:
            self.noise.reset()

        self.prev_actions.clear()
        for _ in range(self.action_history_len):
            self.prev_actions.append(0.0)

        self.level_history = [self.level]
        self.action_history = []

        obs = np.array(
            [self.level] + list(self.prev_actions) + [self.setpoint],
            dtype=np.float32
        )
        init_action = 0.0
        self.reward_manager.reset(
            init_setpoint=self.setpoint,
            init_state=self.level,
            init_action=init_action,
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

        # -------- Action Noise --------
        if self.noise is not None:
            action_val = self.noise.apply_action_noise(action_val)

        # -------- System Dynamics --------
        if self.mode == "voltage":
            action_val = np.clip(action_val, 0, self.max_volt)
            current = (action_val - self.level) / self.R
            d_level = (current / self.C) * self.dt
        else:
            action_val = np.clip(action_val, 0, self.max_current)
            net_flow = action_val - (self.level / self.R)
            d_level = (net_flow / self.C) * self.dt

        # -------- Process Noise --------
        if self.noise is not None:
            d_level = self.noise.apply_process_noise(d_level)

        self.level = np.clip(self.level + d_level, 0, self.level_max)
        self.time += self.dt

        # -------- Store histories --------
        self.prev_actions.append(action_val)
        self.level_history.append(self.level)
        self.action_history.append(action_val)

        # -------- Observation (Sensor Noise) --------
        observed_level = self.level
        if self.noise is not None:
            observed_level = self.noise.apply_sensor_noise(observed_level)

        obs = np.array(
            [observed_level] + list(self.prev_actions) + [self.setpoint],
            dtype=np.float32
        )

        # -------- Reward --------
        error = abs(self.setpoint - self.level)
         # -------- Reward (TRUE STATE, managed) --------
        self.reward_manager.update(
            setpoint=self.setpoint,
            state=self.level,      # TRUE state (no noise)
            action=action_val,
        )

        reward = self.reward_manager.reward_continuous_control()
        # -------- Termination --------
        if error < 0.05:
            self.done += self.dt
        else:
            self.done = 0.0

        terminated = self.done > 5.0
        truncated = False
        info = {"setpoint": self.setpoint}

        return obs, reward, terminated, truncated, info
    # =====================================================
    # RENDER  (COPIED 1:1 FROM USER)
    # =====================================================
    def render(self):
        if self.render_mode is None:
            return

        # -------------------------------------------------
        # Init pygame
        # -------------------------------------------------
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.SysFont("Arial", 13)
            self.font_medium = pygame.font.SysFont("Arial", 15)
            self.font_large = pygame.font.SysFont("Arial", 18, bold=True)

        # Handle close
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

        # Background
        self.screen.fill((245, 245, 245))

        # -------------------------------------------------
        # Header
        # -------------------------------------------------
        header_items = [
            f"Level: {self.level:.2f}/{self.level_max:.1f}",
            f"Setpoint: {self.setpoint:.2f}",
            f"Action: {self.action_history[-1] if self.action_history else 0:.2f}",
            f"Time: {self.time:.1f}s",
        ]
        spacing = 180
        for i, txt in enumerate(header_items):
            surf = self.font_large.render(txt, True, (30, 30, 30))
            self.screen.blit(surf, (40 + i * spacing, 20))

        # -------------------------------------------------
        # Tank Panel (Left)
        # -------------------------------------------------
        tank_x, tank_y = 50, 90
        tank_w, tank_h = 140, 330

        pygame.draw.rect(self.screen, (60, 60, 60),
                         (tank_x, tank_y, tank_w, tank_h), width=3)

        water_ratio = np.clip(self.level / self.level_max, 0, 1)
        water_h = tank_h * water_ratio
        pygame.draw.rect(
            self.screen,
            (50, 130, 255),
            (tank_x + 3, tank_y + tank_h - water_h, tank_w - 6, water_h)
        )

        sp_ratio = np.clip(self.setpoint / self.level_max, 0, 1)
        sp_y = tank_y + tank_h - (tank_h * sp_ratio)
        pygame.draw.line(self.screen, (0, 200, 0),
                         (tank_x, sp_y), (tank_x + tank_w, sp_y), 3)

        self.screen.blit(
            self.font_medium.render("Water Tank", True, (20, 20, 20)),
            (tank_x, tank_y - 28)
        )

        # -------------------------------------------------
        # Graph Panel (Right)
        # -------------------------------------------------
        graph_x, graph_y = 250, 90
        graph_w, graph_h = 540, 330

        gap = 40
        top_h = (graph_h - gap) // 2
        bottom_h = top_h

        self.screen.blit(
            self.font_medium.render("Level History", True, (20, 20, 20)),
            (graph_x, graph_y - 25)
        )
        self.screen.blit(
            self.font_medium.render("Action History", True, (20, 20, 20)),
            (graph_x, graph_y + top_h + gap - 25)
        )

        pygame.draw.rect(self.screen, (80, 80, 80),
                         (graph_x, graph_y, graph_w, top_h), width=2)
        pygame.draw.rect(self.screen, (80, 80, 80),
                         (graph_x, graph_y + top_h + gap, graph_w, bottom_h), width=2)

        # -------------------------------------------------
        # Grid lines + Y labels
        # -------------------------------------------------
        grid_lines = 5
        action_max = self.max_volt if self.mode == "voltage" else self.max_current

        for i in range(grid_lines + 1):
            gy = graph_y + i * top_h / grid_lines
            pygame.draw.line(self.screen, (220, 220, 220),
                             (graph_x, gy), (graph_x + graph_w, gy))
            val = self.level_max * (1 - i / grid_lines)
            txt = self.font_small.render(f"{val:.1f}", True, (90, 90, 90))
            self.screen.blit(txt, (graph_x - 45, gy - 8))

        for i in range(grid_lines + 1):
            gy = graph_y + top_h + gap + i * bottom_h / grid_lines
            pygame.draw.line(self.screen, (220, 220, 220),
                             (graph_x, gy), (graph_x + graph_w, gy))
            val = action_max * (1 - i / grid_lines)
            txt = self.font_small.render(f"{val:.1f}", True, (90, 90, 90))
            self.screen.blit(txt, (graph_x - 45, gy - 8))

        # -------------------------------------------------
        # Plot Level Line
        # -------------------------------------------------
        max_points = min(len(self.level_history), graph_w)
        if max_points > 1:
            lv = self.level_history[-max_points:]
            xs = [graph_x + i for i in range(len(lv))]
            ys = [graph_y + top_h - (v / self.level_max) * top_h for v in lv]
            pygame.draw.lines(self.screen, (0, 70, 200),
                              False, list(zip(xs, ys)), 2)

            sp_line_y = graph_y + top_h - (self.setpoint / self.level_max) * top_h
            pygame.draw.line(self.screen, (0, 180, 0),
                             (graph_x, sp_line_y),
                             (graph_x + graph_w, sp_line_y), 1)

        # -------------------------------------------------
        # Plot Action Line
        # -------------------------------------------------
        if len(self.action_history) > 1:
            act = self.action_history[-max_points:]
            xs = [graph_x + i for i in range(len(act))]
            ys = [
                graph_y + top_h + gap + bottom_h - (a / action_max) * bottom_h
                for a in act
            ]
            pygame.draw.lines(self.screen, (200, 40, 40),
                              False, list(zip(xs, ys)), 2)

        pygame.display.flip()

        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])
        else:
            array = pygame.surfarray.array3d(self.screen)
            return np.transpose(array, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
