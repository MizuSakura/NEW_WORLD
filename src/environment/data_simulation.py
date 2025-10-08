# src/environment/data_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from src.utils.logger import Logger
from src.environment.RC_Tank_env import RC_Tank_Env
from src.environment.signal_generator import SignalGenerator
import argparse
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

     # --- หัวใจของการแก้ไข! ---
    # สร้าง Title แบบไดนามิก
    title = f"Simulation - {signal_type}"
    duty = signal_params.get('duty')
    freq = signal_params.get('freq')
    
    # เพิ่มรายละเอียดลงใน Title เฉพาะเมื่อมีค่านั้นอยู่จริง
    details = []
    if duty is not None:
        details.append(f"duty={duty:.2f}")
    if freq is not None:
        details.append(f"freq={freq:.2f}")
    
    if details:
        title += f" ({', '.join(details)})"
    

    # --- ส่วนที่แก้ไข ---
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(TIME_LOG, DATA_OUTPUT, label='DATA_OUTPUT')
    plt.plot(TIME_LOG, ACTION, label='DATA_INPUT', alpha=0.5)
    plt.title(title) # อันนี้ใช้ตัวแปร title แทน
    # plt.title(f"Simulation - {signal_type} (duty={signal_params.get('duty', 'N/A'):.2f}, freq={signal_params.get('freq', 'N/A'):.2f})")
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
    # 1. สร้าง "เครื่องมืออ่านคำสั่ง"
    parser = argparse.ArgumentParser(description="Run RC Tank Simulation")

    # 2. กำหนด "ช่องรับคำสั่ง" สำหรับ Environment
    parser.add_argument('--R', type=float, default=1.5, help='Resistance value')
    parser.add_argument('--C', type=float, default=2.0, help='Capacitance value')
    parser.add_argument('--dt', type=float, default=0.01, help='Simulation time step')
    parser.add_argument('--control_mode', type=str, default='voltage', choices=['voltage', 'current'], help='Control mode')
    parser.add_argument('--setpoint_level', type=float, default=5.0, help='Target water level')
    parser.add_argument('--level_max', type=float, default=10.0, help='Maximum water level')
    parser.add_argument('--max_action_volt', type=float, default=24.0, help='Max voltage action')
    parser.add_argument('--max_action_current', type=float, default=5.0, help='Max current action')

    # 3. กำหนด "ช่องรับคำสั่ง" สำหรับ Signal
    parser.add_argument('--time_sim', type=float, default=30.0, help='Total simulation time')
    parser.add_argument('--signal_type', type=str, default='pwm', choices=['pwm', 'step', 'ramp', 'impulse'], help='Type of signal')
    parser.add_argument('--amplitude', type=float, default=12.0, help='Signal amplitude')
    parser.add_argument('--duty', type=float, default=0.5, help='PWM duty cycle')
    parser.add_argument('--freq', type=float, default=1.0, help='PWM frequency')
    
    # 4. "อ่านคำสั่ง" ที่ส่งเข้ามา
    args = parser.parse_args()

    # 5. ใช้ "คำสั่งที่อ่านได้" ในการสร้าง Env และ Signal
    env = RC_Tank_Env(R=args.R, C=args.C, dt=args.dt, control_mode=args.control_mode, 
                      setpoint_level=args.setpoint_level, level_max=args.level_max,
                      max_action_volt=args.max_action_volt, max_action_current=args.max_action_current)
    
    sg = SignalGenerator(t_end=args.time_sim, dt=args.dt)

    # signal_params = {
    #     'amplitude': args.amplitude,
    #     'duty': args.duty,
    #     'freq': args.freq
    # }     

    signal_params = {'amplitude': args.amplitude} 
    if args.signal_type == 'pwm':
        signal_params['duty'] = args.duty
        signal_params['freq'] = args.freq

    folder_csv = Path("./data/raw")
    folder_plot = Path("./data/picture")
    
    # 6. รัน Simulation ด้วยค่าที่กำหนด!
    run_simulation(env, sg, signal_type=args.signal_type, signal_params=signal_params,
                   log_dt=0.1, folder_save_csv=folder_csv,
                   folder_save_plot=folder_plot, file_prefix=f"sim_{args.signal_type}")

    print(f"Simulation with {args.signal_type} completed successfully!")
    

# fix title to read N/A by
 # --- หัวใจของการแก้ไข! ---
    # # สร้าง Title แบบไดนามิก
    # title = f"Simulation - {signal_type}"
    # duty = signal_params.get('duty')
    # freq = signal_params.get('freq')
    
    # # เพิ่มรายละเอียดลงใน Title เฉพาะเมื่อมีค่านั้นอยู่จริง
    # details = []
    # if duty is not None:
    #     details.append(f"duty={duty:.2f}")
    # if freq is not None:
    #     details.append(f"freq={freq:.2f}")
    
    # if details:
    #     title += f" ({', '.join(details)})"
    
