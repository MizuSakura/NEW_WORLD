"""
Eval Parameters Schema
----------------------
Validate eval_params.yaml
"""

from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import yaml


# -------------------------------------------------
# Gym Eval
# -------------------------------------------------

class GymEvalConfig(BaseModel):
    model_path:   str  = "models/checkpoint/Autosave.pt"
    env_name:     str  = "RCTankEnv-v0"
    render_mode:  str  = "human"
    episodes:     int  = 10
    max_steps:    int  = 500
    deterministic: bool = True
    save_plot:    bool = True
    plot_folder:  str  = "logs/eval/gym"


# -------------------------------------------------
# Real Eval
# -------------------------------------------------

class RealEvalEnvConfig(BaseModel):
    ip_host:         str   = "192.168.1.100"
    port:            int   = 502
    min_action:      float = 0.0
    max_action:      float = 10.0
    setpoint:        float = 5.0
    delay_of_action: float = 0.2
    address_sensor:  int   = 1
    address_actuator: int  = 1025

class RealEvalConfig(BaseModel):
    model_path:   str  = "models/checkpoint/sac_checkpoint_real.pt"
    episodes:     int  = 5
    max_steps:    int  = 500
    deterministic: bool = True
    save_plot:    bool = True
    plot_folder:  str  = "logs/eval/real"
    env: RealEvalEnvConfig = RealEvalEnvConfig()


# -------------------------------------------------
# LSTM Eval
# -------------------------------------------------

class LSTMEvalEnvConfig(BaseModel):
    sequence_size: int   = 30
    min_action:    float = 0.0
    max_action:    float = 10.0

class LSTMEvalConfig(BaseModel):
    model_path:      str  = "models/checkpoint/sac_on_lstm.pt"
    lstm_model_path: str  = "models/lstm_model_2_s.pth"
    scaler_zip:      str  = "config/AutoScaler_DATA_INPUT_DATA_OUTPUT_scalers.zip"
    episodes:        int  = 10
    max_steps:       int  = 500
    deterministic:   bool = True
    save_plot:       bool = True
    plot_folder:     str  = "logs/eval/lstm"
    env: LSTMEvalEnvConfig = LSTMEvalEnvConfig()

class MQTTCtrlEvalConfig(BaseModel):
    model_path:    str  = "models/checkpoint/Autosave.pt"
    deterministic: bool = True
# -------------------------------------------------
# Root
# -------------------------------------------------

class EvalConfig(BaseModel):
    gym:  GymEvalConfig  = GymEvalConfig()
    real: RealEvalConfig = RealEvalConfig()
    lstm: LSTMEvalConfig = LSTMEvalConfig()
    mqtt_ctrl: MQTTCtrlEvalConfig = MQTTCtrlEvalConfig()


# -------------------------------------------------
# Main (Test)
# -------------------------------------------------

if __name__ == "__main__":
    config_path = Path(
        r"D:\Project_end\New_world\my_project\src\API\config\eval_params.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = EvalConfig(**data)

    print("EvalConfig validation success")
    print("Gym model    :", config.gym.model_path)
    print("Real model   :", config.real.model_path)
    print("LSTM model   :", config.lstm.model_path)
    print("Real env IP  :", config.real.env.ip_host)
