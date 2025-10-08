# src/environment/data_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from src.utils.logger import Logger
from src.environment.RC_Tank_env import RC_Tank_Env
from src.environment.signal_generator import SignalGenerator

# -----------------------------
# Run simulation
# -----------------------------
def run_simulation(env, signal_generator, signal_type='pwm', signal_params=None,
                   log_dt=0.1, folder_save_csv=Path("./data/raw"),
                   folder_save_plot=Path("./data/picture"), file_prefix="sim"):
    """
    Run simulation for a given environment and signal generator.
    """
    folder_save_csv.mkdir(parents=True, exist_ok=True)
    folder_save_plot.mkdir(parents=True, exist_ok=True)

    # Reset environment
    env.reset(default=0.0)

    # Generate signal
    signal_params = signal_params or {}
    if signal_type == 'pwm':
        _, signal = signal_generator.pwm(**signal_params)
    elif signal_type == 'step':
        _, signal = signal_generator.step(**signal_params)
    elif signal_type == 'ramp':
        _, signal = signal_generator.ramp(**signal_params)
    elif signal_type == 'impulse':
        _, signal = signal_generator.impulse(**signal_params)
    elif signal_type == 'sinusoid':
        _, signal = signal_generator.sinusoid(**signal_params)
    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    logger = Logger() # หากไม่มีไฟล์ Logger ให้คอมเมนต์ส่วนนี้ไปก่อน
    dt = signal_generator.dt
    log_interval = max(1, int(log_dt / dt))
    DATA_OUTPUT, ACTION, TIME_LOG = [], [], []

    # Run simulation
    for idx, val in enumerate(signal):
        action = val * (env.max_action_volt if env.mode=='voltage' else env.max_action_current)
        out, done = env.step(action)
        if idx % log_interval == 0:
            ACTION.append(action)
            DATA_OUTPUT.append(out)
            TIME_LOG.append(idx*dt)

    # Log and save
    logger.add_data_log(["TIME", "DATA_INPUT", "DATA_OUTPUT"], [TIME_LOG, ACTION, DATA_OUTPUT])
    file_name_csv = f"{file_prefix}_{signal_type}.csv"
    logger.save_to_csv(file_name=file_name_csv, folder_name=folder_save_csv)

    # --- ส่วนที่แก้ไข ---
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(TIME_LOG, DATA_OUTPUT, label='DATA_OUTPUT')
    plt.plot(TIME_LOG, ACTION, label='DATA_INPUT', alpha=0.5)
    plt.title(f"Simulation - {signal_type} (duty={signal_params.get('duty', 'N/A'):.2f}, freq={signal_params.get('freq', 'N/A'):.2f})")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    file_name_plot = f"{file_prefix}_{signal_type}.png"
    plt.savefig(folder_save_plot / file_name_plot)
    
    plt.pause(2) 
    plt.close()
   
    
# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    TIME_SIM = 1000
    DT = 0.01
    DT_LOG = 0.1
    SETPOINT = 5.0
    MAX_VOLT = 24
    MAX_CURR = 3
    MODE = 'voltage'  # 'voltage' or 'current'
    env = RC_Tank_Env(control_mode=MODE, max_action_volt=MAX_VOLT,level_max=MAX_VOLT,
                      max_action_current=MAX_CURR, setpoint_level=SETPOINT, dt=DT)
    sg = SignalGenerator(t_end=TIME_SIM, dt=DT)

    folder_csv = Path("./data/raw")
    folder_plot = Path("./data/picture")

    # ทดลองรันหลาย duty cycle และหลาย freq ของ PWM
    duty_list = np.linspace(0.5, 0.5, 1)
    freq_list = np.linspace(0.01, 0.01, 1)  # ตัวอย่างช่วง freq 0.1 ถึง 2.0 Hz

    
    for duty in duty_list:
        for freq in freq_list:
            run_simulation(
                env, sg, signal_type='pwm',
                signal_params={'amplitude': 1, 'freq': freq, 'duty': duty},
                log_dt=DT_LOG, folder_save_csv=folder_csv,
                folder_save_plot=folder_plot,
                file_prefix=f"pwm_duty_{duty:.2f}_freq_{freq:.2f}"
            )
