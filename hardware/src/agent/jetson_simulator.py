# hardware/src/agent/jetson_simulator.py
"""
Jetson Simulator
----------------
จำลอง Jetson สำหรับทดสอบ Full Loop บน Laptop โดยไม่มี Jetson จริง

Modes:
    MQTT_CONTROL — รับ action จาก Laptop SAC → actuator
    RL           — SAC บน Jetson เอง (simulate) → actuator
    PID          — PID control → actuator
    MANUAL       — รับ action จาก Dashboard โดยตรง → actuator

ทุก mode:
    - publish telemetry → Dashboard (level, setpoint, action, error, reward)
    - publish heartbeat → Laptop ทุก 2s
    - Emergency Stop → actuator=0 แต่ MQTT ยังเชื่อมต่อ loop ยังรัน
    - MQTT disconnect → fallback PID standalone
    - MQTT reconnect → กลับ mode เดิม

Global Setpoint:
    - รับจาก Dashboard ผ่าน mode topic (payload มี setpoint)
    - lock ไม่ random อีก

วิธีรัน:
    python -m hardware.src.agent.jetson_simulator
    python -m hardware.src.agent.jetson_simulator --mode PID --setpoint 7.0
    python -m hardware.src.agent.jetson_simulator --broker localhost
"""

import argparse
import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# RC Tank Physics
# ======================================================================
class TankPhysics:
    def __init__(self, level_max=10.0, dt=0.1):
        self.level     = float(np.random.uniform(1, level_max * 0.4))
        self.level_max = level_max
        self.dt        = dt
        self.R         = float(np.random.uniform(1.0, 2.5))
        self.C         = float(np.random.uniform(1.5, 3.0))

    def step(self, action: float) -> float:
        inflow   = action / self.level_max
        outflow  = self.level / (self.R * self.C)
        dlevel   = (inflow - outflow) * self.dt
        self.level = float(np.clip(self.level + dlevel, 0.0, self.level_max))
        return self.level


# ======================================================================
# PID Controller
# ======================================================================
class PIDController:
    def __init__(self, kp=2.0, ki=0.1, kd=0.05,
                 setpoint=5.0, out_min=0.0, out_max=10.0, dt=0.1):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.sp = setpoint
        self.mn = out_min; self.mx = out_max; self.dt = dt
        self._integral = 0.0; self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0; self._prev_err = 0.0

    def set_setpoint(self, sp: float):
        self.sp = sp
        self.reset()

    def compute(self, measurement: float) -> float:
        err             = self.sp - measurement
        self._integral += err * self.dt
        self._integral  = float(np.clip(self._integral, self.mn, self.mx))
        deriv           = (err - self._prev_err) / self.dt
        self._prev_err  = err
        return float(np.clip(
            self.kp * err + self.ki * self._integral + self.kd * deriv,
            self.mn, self.mx
        ))


# ======================================================================
# Simple RL Simulator (standalone mode)
# ======================================================================
class SimpleRL:
    """จำลอง SAC ที่รันบน Jetson เอง"""
    def __init__(self, out_min=0.0, out_max=10.0):
        self.mn = out_min; self.mx = out_max

    def select_action(self, level: float, setpoint: float) -> float:
        error  = setpoint - level
        action = np.clip(5.0 + error * 1.5, self.mn, self.mx)
        action += float(np.random.normal(0, 0.03))
        return float(np.clip(action, self.mn, self.mx))


# ======================================================================
# Jetson Simulator
# ======================================================================
class JetsonSimulator:

    VALID_MODES = ("MQTT_CONTROL", "RL", "PID", "MANUAL")

    # Dashboard mode → simulator mode
    MODE_MAP = {
        "RL":           "MQTT_CONTROL",
        "MANUAL":       "MANUAL",
        "PID":          "PID",
        "MQTT_CONTROL": "MQTT_CONTROL",
        "STANDALONE":   "PID",
    }

    TOPICS = {
        "state":         "project/rl/nvidia01/state",
        "action":        "project/rl/nvidia01/action",
        "heartbeat":     "project/rl/nvidia01/heartbeat",
        "telemetry":     "project/rl/nvidia01/telemetry",
        "mode":          "project/rl/nvidia01/mode",
        "emergency":     "project/rl/nvidia01/emergency",
        "manual_action": "project/rl/nvidia01/manual_action",
    }

    def __init__(self, broker="localhost", port=1883,
                 initial_mode="MQTT_CONTROL",
                 setpoint=None, delay=0.2):

        self._broker = broker
        self._port   = port
        self._mode   = initial_mode
        self._delay  = delay

        # Physics + Controllers
        self._tank = TankPhysics(level_max=10.0, dt=delay)
        self._setpoint        = setpoint or float(np.random.uniform(3, 8))
        self._setpoint_locked = setpoint is not None  # lock ถ้า set จาก CLI

        self._pid = PIDController(setpoint=self._setpoint, dt=delay)
        self._rl  = SimpleRL()

        # State
        self._running         = False
        self._mqtt_connected  = False
        self._emergency       = False
        self._last_action     = 0.0
        self._last_action_ts  = 0.0
        self._manual_action   = 0.0
        self._action_timeout  = 5.0
        self._hb_interval     = 2.0
        self._last_hb_ts      = 0.0
        self._step            = 0

        # MQTT
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="jetson_simulator"
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ──────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        print(f"[Simulator] Connecting MQTT → {self._broker}:{self._port}")
        try:
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            print(f"[Simulator] MQTT connect failed: {e} → standalone")
        self._run_loop()

    def stop(self):
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()
        print("[Simulator] Stopped")

    # ──────────────────────────────────────────────────────────────
    def _run_loop(self):
        print(f"[Simulator] Running — mode: {self._mode} | SP: {self._setpoint:.2f}")
        while self._running:
            loop_start = time.time()
            self._step += 1
            level = self._tank.level

            # emergency → actuator=0 แต่ loop ยังรัน
            if self._emergency:
                self._tank.step(0.0)
                self._publish_telemetry(level, 0.0)
                time.sleep(self._delay)
                continue

            # คำนวณ action
            action, source = self._compute_action(level)

            # อัปเดต physics
            new_level = self._tank.step(action)
            self._last_action = action

            # Heartbeat
            now = time.time()
            if now - self._last_hb_ts >= self._hb_interval:
                self._publish_heartbeat()
                self._last_hb_ts = now

            # Telemetry
            self._publish_telemetry(new_level, action)

            # Print ทุก 10 steps
            if self._step % 10 == 0:
                print(
                    f"[{self._mode}|{source}] "
                    f"LV:{new_level:.3f} SP:{self._setpoint:.3f} "
                    f"ACT:{action:.3f} ERR:{self._setpoint - new_level:.3f} "
                    f"STEP:{self._step}",
                    flush=True
                )

            # Auto setpoint ทุก 300 steps ถ้าไม่ lock
            if self._step % 300 == 0 and not self._setpoint_locked:
                self._setpoint = float(np.random.uniform(2, 9))
                self._pid.set_setpoint(self._setpoint)
                print(f"[Simulator] Auto SP: {self._setpoint:.2f}", flush=True)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self._delay - elapsed))

    def _compute_action(self, level: float):
        if self._mode == "MQTT_CONTROL":
            # ส่ง state ไป Laptop
            self._publish_state(level)
            # ถ้า action timeout → fallback PID
            if self._last_action_ts == 0.0:
                return self._pid.compute(level), "FALLBACK"
            if time.time() - self._last_action_ts > self._action_timeout:
                return self._pid.compute(level), "FALLBACK"
            return self._last_action, "SAC"

        elif self._mode == "RL":
            return self._rl.select_action(level, self._setpoint), "RL_LOCAL"

        elif self._mode == "PID":
            return self._pid.compute(level), "PID"

        elif self._mode == "MANUAL":
            return self._manual_action, "MANUAL"

        return 0.0, "NONE"

    # ──────────────────────────────────────────────────────────────
    def _publish_state(self, level: float):
        self._publish(self.TOPICS["state"], {
            "level":    round(float(level), 4),
            "setpoint": round(float(self._setpoint), 4),
            "step":     self._step,
            "ts":       datetime.now().isoformat(),
        })

    def _publish_heartbeat(self):
        self._publish(self.TOPICS["heartbeat"], {
            "id":   "jetson_simulator",
            "mode": self._mode,
            "ts":   datetime.now().isoformat(),
        })

    def _publish_telemetry(self, level: float, action: float):
        error = self._setpoint - level
        self._publish(self.TOPICS["telemetry"], {
            "level":    round(float(level),   4),
            "setpoint": round(float(self._setpoint), 4),
            "action":   round(float(action),  4),
            "error":    round(float(error),   4),
            "reward":   round(-abs(float(error)), 4),
            "mode":     self._mode,
            "step":     self._step,
            "ts":       datetime.now().isoformat(),
        })

    def _publish(self, topic: str, payload: dict):
        if not self._mqtt_connected:
            return
        try:
            self._client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False),
                qos=1
            )
        except Exception as e:
            print(f"[Simulator] Publish error: {e}")

    # ──────────────────────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._mqtt_connected = True
            client.subscribe(self.TOPICS["action"],        1)
            client.subscribe(self.TOPICS["mode"],          1)
            client.subscribe(self.TOPICS["emergency"],     2)
            client.subscribe(self.TOPICS["manual_action"], 1)
            print(f"[Simulator] MQTT connected — mode: {self._mode}")
        else:
            print(f"[Simulator] MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None, reasoncode=None):
        self._mqtt_connected = False
        print(f"[Simulator] MQTT disconnected → PID fallback")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        topic = msg.topic
        if topic == self.TOPICS["action"]:
            self._handle_action(data)
        elif topic == self.TOPICS["mode"]:
            self._handle_mode(data)
        elif topic == self.TOPICS["emergency"]:
            self._handle_emergency(data)
        elif topic == self.TOPICS["manual_action"]:
            self._handle_manual_action(data)

    def _handle_action(self, data: dict):
        action = float(np.clip(float(data.get("action", self._last_action)), 0.0, 10.0))
        self._last_action    = action
        self._last_action_ts = time.time()

    def _handle_mode(self, data: dict):
        raw  = str(data.get("mode", self._mode)).upper()
        mode = self.MODE_MAP.get(raw, raw)
        if mode not in self.VALID_MODES:
            print(f"[Simulator] Unknown mode: {raw}")
            return
        if mode != self._mode:
            print(f"[Simulator] Mode: {self._mode} → {mode}")
            self._mode = mode
            if mode in ("PID", "MQTT_CONTROL"):
                self._pid.reset()

        # รับ setpoint จาก dashboard → lock
        if "setpoint" in data:
            sp = float(data["setpoint"])
            self._setpoint        = sp
            self._setpoint_locked = True
            self._pid.set_setpoint(sp)
            print(f"[Simulator] SP locked → {sp:.2f}")

    def _handle_emergency(self, data: dict):
        cmd = str(data.get("command", ""))
        if cmd == "EMERGENCY_STOP":
            self._emergency   = True
            self._last_action = 0.0
            print("[Simulator] *** EMERGENCY STOP — actuator=0 ***")
        elif cmd == "CLEAR":
            self._emergency = False
            self._pid.reset()
            print("[Simulator] Emergency cleared → resume")

    def _handle_manual_action(self, data: dict):
        self._manual_action = float(np.clip(float(data.get("action", 0.0)), 0.0, 10.0))


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Jetson Simulator")
    parser.add_argument("--broker",   default="localhost")
    parser.add_argument("--port",     type=int, default=1883)
    parser.add_argument("--mode",     default="MQTT_CONTROL",
                        choices=["MQTT_CONTROL", "RL", "PID", "MANUAL"])
    parser.add_argument("--setpoint", type=float, default=None,
                        help="Fixed setpoint (default: random, lock after Set SP)")
    parser.add_argument("--delay",    type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 55)
    print("[Simulator] Jetson Simulator")
    print(f"[Simulator] Broker   : {args.broker}:{args.port}")
    print(f"[Simulator] Mode     : {args.mode}")
    print(f"[Simulator] Setpoint : {args.setpoint or 'random (lock on Set SP)'}")
    print("=" * 55)

    sim = JetsonSimulator(
        broker       = args.broker,
        port         = args.port,
        initial_mode = args.mode,
        setpoint     = args.setpoint,
        delay        = args.delay,
    )
    try:
        sim.start()
    except KeyboardInterrupt:
        print("\n[Simulator] Interrupted")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()