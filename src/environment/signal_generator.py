import numpy as np

class SignalGenerator:
    def __init__(self, t_end=100, dt=0.1):
        self.dt = dt
        self.t = np.arange(0, t_end, dt)

    def pwm(self, amplitude=1, freq=1, duty=0.5):
        """
        Generate PWM signal
        """
        T = 1 / freq
        return amplitude * ((self.t % T) < duty * T)

    def step(self, amplitude=1, start_time=0):
        """
        Step signal: amplitude after start_time
        """
        return (self.t >= start_time) * amplitude

    def ramp(self, slope=1, start_time=0):
        """
        Linear ramp starting from start_time
        """
        return slope * np.maximum(0, self.t - start_time)

    def impulse(self, amplitude=1, at_time=0.0):
        """
        Impulse signal: single spike at specific time
        """
        signal = np.zeros_like(self.t)
        idx = np.argmin(np.abs(self.t - at_time))
        signal[idx] = amplitude
        return signal
