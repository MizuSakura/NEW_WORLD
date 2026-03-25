#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/environment/signal_generator_hardware.py
from __future__ import print_function
import numpy as np
from scipy.signal import sawtooth

class SignalGenerator(object):

    def __init__(self, t_end=10.0, dt=0.01):
        self.dt = dt
        self.t = np.arange(0, t_end, dt)

        self._signal_map = {
            "pwm": self.pwm,
            "step": self.step,
            "ramp": self.ramp,
            "impulse": self.impulse,
            "sinusoid": self.sinusoid,
            "sine": self.sinusoid,
            "triangle": self.triangle,
        }

    def generate_from_config(self, config):
        """
        Generate signal from dictionary config.
        Compatible with Python 3.6 (no type hints for dict).
        """
        if "type" not in config:
            raise ValueError("Signal config missing 'type'")
        if "params" not in config:
            raise ValueError("Signal config missing 'params'")

        signal_type = config["type"]
        params = config["params"]

        if signal_type not in self._signal_map:
            # Change f-string to .format()
            raise ValueError("Unsupported signal type: {}".format(signal_type))

        return self._signal_map[signal_type](**params)

    # ======================================================================
    # PWM SIGNAL
    # ======================================================================
    def pwm(self, amplitude=1.0, duty_cycle=0.5, frequency=1.0,
            freq=None, duty=None):
        """Generate PWM signal."""
        frequency = freq if freq is not None else frequency
        duty_cycle = duty if duty is not None else duty_cycle
        
        # In Python 3.6/Old Numpy, explicit casting to float is safer
        condition = np.mod(self.t * frequency, 1) < duty_cycle
        signal = amplitude * condition.astype(np.float64)
        return self.t, signal

    def step(self, amplitude=1.0, start_time=1.0):
        signal = np.where(self.t >= start_time, amplitude, 0.0)
        return self.t, signal

    def ramp(self, slope=0.5):
        signal = slope * self.t
        return self.t, signal

    def impulse(self, amplitude=1.0, time=1.0):
        signal = np.zeros_like(self.t)
        # Find index closest to the target time
        idx = np.argmin(np.abs(self.t - time))
        signal[idx] = amplitude
        return self.t, signal

    def sinusoid(self, amplitude=1.0, frequency=1.0, phase=0.0, freq=None):
        frequency = freq if freq is not None else frequency
        signal = amplitude * np.sin(2 * np.pi * frequency * self.t + phase)
        return self.t, signal

    def triangle(self, amplitude=1.0, frequency=1.0, freq=None):
        frequency = freq if freq is not None else frequency
        signal = amplitude * sawtooth(2 * np.pi * frequency * self.t, width=0.5)
        return self.t, signal

if __name__ == "__main__":
    # 1. Initialize Generator
    # Test with 2.0 seconds duration, 0.1s time step
    gen = SignalGenerator(t_end=2.0, dt=0.1)
    print("--- Testing SignalGenerator (Python 3.6.9) ---")
    print("Time vector: {}".format(gen.t))

    # 2. Test PWM Generation (Manual Call)
    print("\n[Test 1] PWM Signal:")
    t, sig_pwm = gen.pwm(amplitude=5.0, frequency=1.0, duty_cycle=0.5)
    print("PWM Result: {}".format(sig_pwm))

    # 3. Test Step Generation (Manual Call)
    print("\n[Test 2] Step Signal (Start at 0.5s):")
    t, sig_step = gen.step(amplitude=1.0, start_time=0.5)
    print("Step Result: {}".format(sig_step))

    # 4. Test generate_from_config (The way your 'response' class uses it)
    print("\n[Test 3] Generate from Config (Sinusoid):")
    sample_config = {
        "type": "sinusoid",
        "params": {
            "amplitude": 10.0,
            "frequency": 2.0,
            "phase": 0.0
        }
    }
    
    try:
        t_cfg, sig_cfg = gen.generate_from_config(sample_config)
        print("Config Success! First 5 values: {}".format(sig_cfg[:5]))
    except Exception as e:
        print("Config Failed: {}".format(e))

    # 5. Test Error Handling
    print("\n[Test 4] Error Handling (Invalid Type):")
    try:
        gen.generate_from_config({"type": "unknown", "params": {}})
    except ValueError as e:
        print("Caught expected error: {}".format(e))

    print("\n--- All tests completed ---")
