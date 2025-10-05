import numpy as np
from src.environment.signal_generator import SignalGenerator

def test_step_signal():
    sg = SignalGenerator(t_end=1, dt=0.1)
    sig = sg.step(amplitude=2, start_time=0.5)
    assert sig.shape == sg.t.shape
    assert np.all(sig[sg.t < 0.5] == 0)
    assert np.all(sig[sg.t >= 0.5] == 2)

def test_impulse_signal():
    sg = SignalGenerator(t_end=1, dt=0.1)
    sig = sg.impulse(amplitude=1, at_time=0.5)
    idx = np.argmin(np.abs(sg.t - 0.5))
    # มี spike เพียงจุดเดียว
    assert sig[idx] == 1
    assert np.sum(sig) == 1
    assert np.count_nonzero(sig) == 1

def test_pwm_signal():
    sg = SignalGenerator(t_end=1, dt=0.01)
    sig = sg.pwm(amplitude=1, freq=5, duty=0.5)
    assert np.max(sig) == 1
    assert np.min(sig) == 0
    assert sig.shape == sg.t.shape
