import numpy as np
from scipy.signal import sawtooth

class SignalGenerator:
    """
    Generate various test input signals for simulation such as 
    PWM, step, ramp, impulse, sinusoidal, and triangle waveforms.
    """

    def __init__(self, t_end=10.0, dt=0.01):
        """Initialize the signal generator with a time vector."""
        self.dt = dt                      # ✅ add this line
        self.t = np.arange(0, t_end, dt)

    # ======================================================================
    # PWM SIGNAL
    # ======================================================================
    def pwm(self, amplitude=1.0, duty_cycle=0.5, frequency=1.0,
            freq=None, duty=None):
        """Generate PWM signal."""
        frequency = freq if freq is not None else frequency
        duty_cycle = duty if duty is not None else duty_cycle
        signal = amplitude * (np.mod(self.t * frequency, 1) < duty_cycle).astype(float)
        return self.t, signal

    def step(self, amplitude=1.0, start_time=1.0):
        signal = np.where(self.t >= start_time, amplitude, 0.0)
        return self.t, signal

    def ramp(self, slope=0.5):
        signal = slope * self.t
        return self.t, signal

    def impulse(self, amplitude=1.0, time=1.0):
        signal = np.zeros_like(self.t)
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
