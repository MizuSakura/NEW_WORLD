#D:\Project_end\New_world\my_project\src\environment\noise_manager.py
import numpy as np

# =====================================================
# Base Noise Interface
# =====================================================
class NoiseModel:
    def reset(self):
        pass

    def sample(self):
        raise NotImplementedError


# =====================================================
# Basic Noise Implementations
# =====================================================
class GaussianNoise(NoiseModel):
    def __init__(self, sigma=0.01):
        self.base_sigma = sigma
        self.sigma = sigma

    def set_scale(self, scale):
        self.sigma = self.base_sigma * scale

    def sample(self):
        return np.random.normal(0.0, self.sigma)


class BoundedGaussianNoise(GaussianNoise):
    def __init__(self, sigma=0.01, clip=0.05):
        super().__init__(sigma)
        self.clip = clip

    def sample(self):
        return np.clip(
            np.random.normal(0.0, self.sigma),
            -self.clip,
            self.clip
        )


class OUNoise(NoiseModel):
    def __init__(self, mu=0.0, theta=0.15, sigma=0.02, dt=0.1):
        self.mu = mu
        self.theta = theta
        self.base_sigma = sigma
        self.sigma = sigma
        self.dt = dt
        self.x = 0.0

    def set_scale(self, scale):
        self.sigma = self.base_sigma * scale

    def reset(self):
        self.x = 0.0

    def sample(self):
        dx = (
            self.theta * (self.mu - self.x) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn()
        )
        self.x += dx
        return self.x


# =====================================================
# Schedulers
# =====================================================
class NormalCurveScheduler:
    """
    Bell-shaped (cave) noise schedule
    """

    def __init__(self, peak, std, max_scale=1.0):
        self.peak = peak
        self.std = std
        self.max_scale = max_scale

    def scale(self, t):
        return self.max_scale * np.exp(
            -0.5 * ((t - self.peak) / self.std) ** 2
        )


class TDErrorController:
    """
    EMA-based TD-error tracker
    TD high  -> noise low
    TD low   -> noise high
    """

    def __init__(
        self,
        ema_alpha=0.05,
        min_scale=0.1,
        max_scale=1.0,
        target_td=0.5,
    ):
        self.ema_alpha = ema_alpha
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_td = target_td
        self.td_ema = None

    def update(self, td_error):
        td = float(abs(td_error))
        if self.td_ema is None:
            self.td_ema = td
        else:
            self.td_ema = (
                self.ema_alpha * td
                + (1 - self.ema_alpha) * self.td_ema
            )

    def scale(self):
        ratio = self.target_td / (self.td_ema + 1e-6)
        return np.clip(ratio, self.min_scale, self.max_scale)


class TDDrivenScheduler:
    def __init__(self, td_controller: TDErrorController):
        self.td_controller = td_controller

    def scale(self, _=None):
        return self.td_controller.scale()


# =====================================================
# Scheduled Noise Wrapper
# =====================================================
class ScheduledNoise(NoiseModel):
    """
    Noise wrapper supporting step-based or episode-based schedule
    """

    def __init__(
        self,
        noise: NoiseModel,
        scheduler,
        schedule_mode="step",  # "step" | "episode"
    ):
        self.noise = noise
        self.scheduler = scheduler
        self.schedule_mode = schedule_mode

        self.step_count = 0
        self.episode_count = 0

    def reset(self):
        self.step_count = 0
        self.noise.reset()

    def on_step(self):
        if self.schedule_mode == "step":
            self._apply_scale(self.step_count)
            self.step_count += 1

    def on_episode(self):
        if self.schedule_mode == "episode":
            self._apply_scale(self.episode_count)
            self.episode_count += 1

    def _apply_scale(self, t):
        scale = self.scheduler.scale(t)
        if hasattr(self.noise, "set_scale"):
            self.noise.set_scale(scale)

    def sample(self):
        return self.noise.sample()


# =====================================================
# Noise Manager
# =====================================================
class NoiseManager:
    """
    Central controller for all noise sources
    """

    def __init__(
        self,
        action_noise=None,
        process_noise=None,
        sensor_noise=None,
        enabled=True,
        td_controller: TDErrorController = None,
    ):
        self.action_noise = action_noise
        self.process_noise = process_noise
        self.sensor_noise = sensor_noise
        self.enabled = enabled
        self.td_controller = td_controller

    def reset(self):
        if not self.enabled:
            return
        for n in self._all_noises():
            if n is not None:
                n.reset()

    def step(self):
        """Call every environment step"""
        for n in self._all_noises():
            if hasattr(n, "on_step"):
                n.on_step()

    def episode_step(self):
        """Call at beginning of each episode"""
        for n in self._all_noises():
            if hasattr(n, "on_episode"):
                n.on_episode()

    def update_td_error(self, td_error):
        if self.td_controller is not None:
            self.td_controller.update(td_error)

    # -----------------------------
    # Apply noise
    # -----------------------------
    def apply_action_noise(self, action):
        if self.enabled and self.action_noise:
            return action + self.action_noise.sample()
        return action

    def apply_process_noise(self, value):
        if self.enabled and self.process_noise:
            return value + self.process_noise.sample()
        return value

    def apply_sensor_noise(self, value):
        if self.enabled and self.sensor_noise:
            return value + self.sensor_noise.sample()
        return value

    def _all_noises(self):
        return (self.action_noise, self.process_noise, self.sensor_noise)
