import pytest
import numpy as np
from src.environment.RC_Tank_env import RC_Tank_Env


# ------------------------------------------------------------------
# 1️⃣  Test: reset() function
# ------------------------------------------------------------------
def test_reset_returns_valid_level_and_done():
    env = RC_Tank_Env(level_max=10.0)
    level, done = env.reset(default=3.5)
    assert isinstance(level, float)
    assert 0 <= level <= env.level_max
    assert done is False


def test_reset_random_initialization():
    env = RC_Tank_Env(level_max=10.0)
    level_1, _ = env.reset(default=None)
    level_2, _ = env.reset(default=None)
    assert 0 <= level_1 <= env.level_max
    assert 0 <= level_2 <= env.level_max
    # ความน่าจะเป็นสูงที่ค่าไม่ซ้ำกัน
    assert level_1 != level_2


# ------------------------------------------------------------------
# 2️⃣  Test: voltage control mode
# ------------------------------------------------------------------
def test_step_voltage_clipping():
    env = RC_Tank_Env(control_mode='voltage', max_action_volt=10)
    env.reset(default=0)
    v_out, done = env.step(action=20)  # เกิน max → ต้องถูก clip
    assert 0 <= v_out <= env.level_max
    assert done in [True, False]


def test_step_voltage_negative_input():
    env = RC_Tank_Env(control_mode='voltage', max_action_volt=10)
    env.reset(default=5.0)
    v_out, done = env.step(action=-10)  # น้อยกว่า 0 → ถูก clip เป็น 0
    assert 0 <= v_out <= env.level_max
    assert done in [True, False]


# ------------------------------------------------------------------
# 3️⃣  Test: current control mode
# ------------------------------------------------------------------
def test_step_current_clipping():
    env = RC_Tank_Env(control_mode='current', max_action_current=5)
    env.reset(default=0)
    v_out, done = env.step(action=10)  # เกิน max → ต้องถูก clip
    assert 0 <= v_out <= env.level_max
    assert done in [True, False]


def test_step_current_negative_input():
    env = RC_Tank_Env(control_mode='current', max_action_current=5)
    env.reset(default=2.0)
    v_out, done = env.step(action=-10)
    assert 0 <= v_out <= env.level_max


# ------------------------------------------------------------------
# 4️⃣  Test: setpoint detection
# ------------------------------------------------------------------
def test_setpoint_done_when_reached():
    env = RC_Tank_Env(setpoint_level=1.0)
    env.reset(default=1.0)
    _, done = env.step(action=0)
    assert done is True


def test_not_done_when_far_from_setpoint():
    env = RC_Tank_Env(setpoint_level=5.0)
    env.reset(default=0.0)
    _, done = env.step(action=0)
    assert done is False


# ------------------------------------------------------------------
# 5️⃣  Test: multi-step evolution
# ------------------------------------------------------------------
def test_level_evolves_over_time():
    env = RC_Tank_Env(control_mode='voltage')
    env.reset(default=0.0)
    prev_level = env.level
    for _ in range(10):
        level, done = env.step(action=5.0)
        assert 0 <= level <= env.level_max
        prev_level = level
    # ตรวจว่าระดับน้ำเปลี่ยนจริง
    assert level != 0.0
