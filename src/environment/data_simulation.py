# src/environment/data_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from src.utils.logger import Logger
from src.environment.RC_Tank_env import RC_Tank_Env
from src.environment.signal_generator import SignalGenerator

# ==============================================================
# SIMULATION CORE FUNCTION
# ==============================================================
def run_simulation(env, signal_generator, signal_type='pwm', signal_params=None,
                   log_dt=0.1, folder_save_csv=Path("./data/raw"),
                   folder_save_plot=Path("./data/picture"), file_prefix="sim"):
    """
    Run simulation for a given environment and signal generator.
    Logs TIME, DATA_INPUT, and DATA_OUTPUT, and saves results as both CSV and plot.
    """
    folder_save_csv.mkdir(parents=True, exist_ok=True)
    folder_save_plot.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # RESET ENVIRONMENT AND GENERATE SIGNAL
    # ==========================================================
    env.reset(default=0.0)
    signal_params = signal_params or {}

    if signal_type == 'pwm':
        _, signal = signal_generator.pwm(**signal_params)
    elif signal_type == 'step':
        _, signal = signal_generator.step(**signal_params)
    elif signal_type == 'ramp':
        _, signal = signal_generator.ramp(**signal_params)
    elif signal_type == 'impulse':
        _, signal = signal_generator.impulse(**signal_params)
    elif signal_type in ['sinusoid', 'sine']:
        _, signal = signal_generator.sinusoid(**signal_params)
    elif signal_type == 'triangle':
        _, signal = signal_generator.triangle(**signal_params)
    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    # ==========================================================
    # SIMULATION LOOP
    # ==========================================================
    logger = Logger()
    dt = signal_generator.dt
    log_interval = max(1, int(log_dt / dt))
    DATA_OUTPUT, ACTION, TIME_LOG = [], [], []

    for idx, val in enumerate(signal):
        action = val * (env.max_action_volt if env.mode == 'voltage' else env.max_action_current)
        out, done = env.step(action)
        if idx % log_interval == 0:
            ACTION.append(action)
            DATA_OUTPUT.append(out)
            TIME_LOG.append(idx * dt)

    # ==========================================================
    # SAVE DATA (CSV) + PLOT RESULTS
    # ==========================================================
    logger.add_data_log(["TIME", "DATA_INPUT", "DATA_OUTPUT"], [TIME_LOG, ACTION, DATA_OUTPUT])

    # Extract readable metadata (for both file name and plot title)
    meta_info = ", ".join([
        f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
        for k, v in signal_params.items()
    ])
    meta_suffix = "_".join([
        f"{k}_{v}" for k, v in signal_params.items() if isinstance(v, (int, float, str))
    ])

    # File names
    file_name_plot = f"{file_prefix}_{signal_type}_{meta_suffix}.png"
    file_name_csv = f"{file_prefix}_{signal_type}_{meta_suffix}.csv"

    # --- Plot simulation results ---
    plt.figure(figsize=(10, 4))
    plt.plot(TIME_LOG, DATA_OUTPUT, label='DATA_OUTPUT')
    plt.plot(TIME_LOG, ACTION, label='DATA_INPUT', alpha=0.5)
    plt.title(f"Simulation - {signal_type.upper()} ({meta_info})")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Save files
    plt.savefig(folder_save_plot / file_name_plot)
    logger.save_to_csv(file_name=file_name_csv, folder_name=folder_save_csv)

    plt.pause(2)
    plt.close()

# ==============================================================
# MAIN ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    # ------------------------
    # Simulation Configuration
    # ------------------------
    TIME_SIM = 100
    DT = 0.1
    DT_LOG = 0.01
    SETPOINT = 5.0
    MAX_VOLT = 24
    MAX_CURR = 3
    MODE = 'voltage'  # or 'current'

    # Initialize environment & signal generator
    env = RC_Tank_Env(control_mode=MODE, max_action_volt=MAX_VOLT, level_max=MAX_VOLT,
                      max_action_current=MAX_CURR, setpoint_level=SETPOINT, dt=DT)
    sg = SignalGenerator(t_end=TIME_SIM, dt=DT)

    folder_csv = Path("./data/raw")
    folder_plot = Path("./data/picture")

    # ------------------------
    # User-selectable signal type
    # ------------------------
    #available_signals = ['pwm', 'sinusoid', 'triangle', 'step', 'ramp', 'impulse']
    available_signals = ['pwm', 'sinusoid', 'triangle', 'step', 'ramp', 'impulse']

    print("Available signal types:")
    for i, s in enumerate(available_signals, start=1):
        print(f"{i}. {s}")
    choice = int(input("\nSelect signal type [1-6]: "))
    signal_type = available_signals[choice - 1]
    print(f"\n👉 Running simulation with signal: {signal_type.upper()}")

    # ------------------------
    # Run according to signal type
    # ------------------------
    freq_list = [0.1]
    if signal_type == 'pwm':
        duty_list = [0.1,0.2,0.,0.4,0.5,0.6,0.7,0.8,0.9]
        
        for duty in duty_list:
            for freq in freq_list:
                run_simulation(
                    env, sg, signal_type='pwm',
                    signal_params={'amplitude': 1, 'freq': freq, 'duty': duty},
                    log_dt=DT_LOG, folder_save_csv=folder_csv,
                    folder_save_plot=folder_plot,
                    file_prefix=f"pwm_duty_{duty:.2f}_freq_{freq:.2f}"
                )

    elif signal_type in ['sinusoid', 'sine']:
        for freq in freq_list:
            run_simulation(
                env, sg, signal_type='sinusoid',
                signal_params={'amplitude': 1.0, 'frequency': freq, 'phase': 0.0},
                log_dt=DT_LOG, folder_save_csv=folder_csv,
                folder_save_plot=folder_plot,
                file_prefix=f"sine_freq_{freq:.2f}"
            )

    elif signal_type == 'triangle':
        for freq in freq_list:
            run_simulation(
                env, sg, signal_type='triangle',
                signal_params={'amplitude': 1.0, 'frequency': freq},
                log_dt=DT_LOG, folder_save_csv=folder_csv,
                folder_save_plot=folder_plot,
                file_prefix=f"triangle_freq_{freq:.2f}"
            )

    elif signal_type == 'step':
        run_simulation(env, sg, signal_type='step',
                       signal_params={'amplitude': 1.0, 'start_time': 5.0},
                       log_dt=DT_LOG, folder_save_csv=folder_csv,
                       folder_save_plot=folder_plot,
                       file_prefix="step")

    elif signal_type == 'ramp':
        run_simulation(env, sg, signal_type='ramp',
                       signal_params={'slope': 0.2},
                       log_dt=DT_LOG, folder_save_csv=folder_csv,
                       folder_save_plot=folder_plot,
                       file_prefix="ramp")

    elif signal_type == 'impulse':
        run_simulation(env, sg, signal_type='impulse',
                       signal_params={'amplitude': 1.0, 'time': 5.0},
                       log_dt=DT_LOG, folder_save_csv=folder_csv,
                       folder_save_plot=folder_plot,
                       file_prefix="impulse")

    print("\n✅ Simulation complete!")
