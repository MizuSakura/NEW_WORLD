"""
RL Parameters Schema
--------------------
Validate rl_params.yaml
เพิ่ม num_envs ใน TrainingConfig สำหรับ multi-env mode
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from pathlib import Path
import yaml


# -------------------------------------------------
# Training
# -------------------------------------------------

class TrainingConfig(BaseModel):
    mode: str = "offline"
    algorithm: str = "SAC"
    episodes: int = 10000
    max_steps: int = 200
    batch_size: int = 1080
    auto_save_every: int = 1
    checkpoint_path: str = "models/checkpoint/Autosave.pt"
    final_model_path: str = "models/Test_history.pt"
    env_name: str = "RCTankEnv-v0"
    render_mode: str = "human"
    num_envs: int = 8       # ← ใหม่: จำนวน parallel env สำหรับ mode multienv


# -------------------------------------------------
# SAC
# -------------------------------------------------

class ActorConfig(BaseModel):
    layers: int = 2
    hidden: int = 256

class CriticConfig(BaseModel):
    layers: int = 2
    hidden: int = 256
    encoder: bool = False

class SACConfig(BaseModel):
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.05     # default ใหม่ 0.05 (เดิม 0.4)
    replay_capacity: int = 200000
    buffer_type: str = "nstep_per"
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta: float = 0.4
    actor: ActorConfig = ActorConfig()
    critic: CriticConfig = CriticConfig()


# -------------------------------------------------
# Noise
# -------------------------------------------------

class OUNoiseConfig(BaseModel):
    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.10     # default ใหม่ 0.10 (เดิม 0.25)
    dt: float = 0.1

class GaussianConfig(BaseModel):
    sigma: float = 0.01     # default ใหม่ 0.01 (เดิม 0.02)

class SensorNoiseConfig(BaseModel):
    sigma: float = 0.02
    clip: float = 0.05

class SchedulerConfig(BaseModel):
    peak: int = 3000
    std: float = 1500.0
    max_scale: float = 1.0

class NoiseConfig(BaseModel):
    enabled: bool = True
    ou_noise: OUNoiseConfig = OUNoiseConfig()
    gaussian: GaussianConfig = GaussianConfig()
    sensor_noise: SensorNoiseConfig = SensorNoiseConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


# -------------------------------------------------
# Logger
# -------------------------------------------------

class LoggerConfig(BaseModel):
    episode_folder: str = "logs/episode"
    episode_filename: str = "episode_"
    agent_folder: str = "logs/agent/RC_Tank"
    agent_filename: str = "optimized_"


# -------------------------------------------------
# State
# รองรับทั้ง int และ str (yaml บางครั้ง quote ตัวเลข)
# รองรับ "Ture" typo → True
# -------------------------------------------------

class StateConfig(BaseModel):
    level_history:    int   = 3
    action_history:   int   = 3
    error_history:    int   = 0
    setpoint_history: int   = 3
    integral:         bool  = True
    derivative:       bool  = True
    level_max:        float = 10.0

    @validator("level_history", "action_history", "error_history",
               "setpoint_history", pre=True)
    def coerce_int(cls, v):
        return int(v)

    @validator("level_max", pre=True)
    def coerce_float(cls, v):
        return float(v)

    @validator("integral", "derivative", pre=True)
    def coerce_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "ture", "1", "yes")
        return bool(v)

    @property
    def state_dim(self) -> int:
        dim = self.level_history + self.action_history
        if self.error_history > 0:
            dim += self.error_history
        if self.setpoint_history > 0:
            dim += self.setpoint_history
        if self.integral:
            dim += 1
        if self.derivative:
            dim += 1
        return dim


# -------------------------------------------------
# Root
# -------------------------------------------------

class RLConfig(BaseModel):
    training: TrainingConfig = TrainingConfig()
    sac:      SACConfig      = SACConfig()
    noise:    NoiseConfig    = NoiseConfig()
    logger:   LoggerConfig   = LoggerConfig()
    state:    StateConfig    = StateConfig()


# -------------------------------------------------
# Main (Test)
# -------------------------------------------------

if __name__ == "__main__":
    config_path = Path(
        r"D:\Project_end\New_world\my_project\src\API\config\rl_params.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = RLConfig(**data)

    print("RLConfig validation success")
    print("Algorithm     :", config.training.algorithm)
    print("Episodes      :", config.training.episodes)
    print("Num envs      :", config.training.num_envs)
    print("Learning rate :", config.sac.learning_rate)
    print("Alpha         :", config.sac.alpha)
    print("Buffer type   :", config.sac.buffer_type)
    print("Noise enabled :", config.noise.enabled)
    print("State dim     :", config.state.state_dim)