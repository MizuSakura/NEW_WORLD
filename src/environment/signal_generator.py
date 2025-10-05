# src/environment/signal_generator.py
import numpy as np

class SignalGenerator:
    """
    Generator สำหรับสร้างสัญญาณ input
    รองรับ:
      - PWM
      - Step
      - Ramp
      - Impulse
      - Sinusoid (optional เพิ่มได้)
    คืนค่า: time_array, signal
    """
    def __init__(self, t_end=100, dt=0.1):
        self.dt = dt
        self.t_end = t_end
        self.time_array = np.arange(0, t_end, dt)

    # -----------------------------
    # Step signal
    # -----------------------------
    def step(self, amplitude=1.0, start_time=0.0):
        signal = amplitude * (self.time_array >= start_time)
        return self.time_array, signal

    # -----------------------------
    # Ramp signal
    # -----------------------------
    def ramp(self, slope=1.0, start_time=0.0):
        signal = slope * np.maximum(0, self.time_array - start_time)
        return self.time_array, signal

    # -----------------------------
    # Impulse signal
    # -----------------------------
    def impulse(self, amplitude=1.0, at_time=0.0):
        signal = np.zeros_like(self.time_array)
        idx = np.argmin(np.abs(self.time_array - at_time))
        signal[idx] = amplitude
        return self.time_array, signal

    # -----------------------------
    # PWM signal
    # -----------------------------
    def pwm(self, amplitude=1.0, freq=1.0, duty=0.5):
        T = 1 / freq
        signal = amplitude * ((self.time_array % T) < duty * T)
        return self.time_array, signal

    # -----------------------------
    # Sinusoid (option)
    # -----------------------------
    def sinusoid(self, amplitude=1.0, freq=1.0, phase=0.0):
        signal = amplitude * np.sin(2 * np.pi * freq * self.time_array + phase)
        return self.time_array, signal
