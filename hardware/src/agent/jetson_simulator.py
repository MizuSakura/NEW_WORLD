#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/agent/jetson_simulator.py

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
        self._integral = 0.0
        self._prev_err = 0.0

    def set_setpoint(self, sp: float):
        self.sp = sp
        self.reset()

    def compute(self, measurement: float) -> float:
        err = self.sp - measurement
        self._integral += err * self.dt
        self._integral = float(np.clip(self._integral, self.mn, self.mx))
        deriv = (err - self._prev_err) / self.dt
        self._prev_err = err
        return float(np.clip(
            self.kp * err + self.ki * self._integral + self.kd * deriv,
            self.mn, self.mx
        ))


# ======================================================================
# Simple RL Simulator
# ======================================================================
class SimpleRL:
    def __init__(self, out_min=0.0, out_max=10.0):
        self.mn = out_min
        self.mx = out_max

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

    def __init__(self, broker, port, user, password,
                 initial_mode="MQTT_CONTROL",
                 setpoint=None, delay=0.2):

        self._broker = broker
        self._port   = port
        self._user   = user
        self._pass   = password
        self._mode   = initial_mode
        self._delay  = delay

        self._tank = TankPhysics(level_max=10.0, dt=delay)
        self._setpoint        = setpoint or float(np.random.uniform(3, 8))
        self._setpoint_locked = setpoint is not None

        self._pid = PIDController(setpoint=self._setpoint, dt=delay)
        self._rl  = SimpleRL()

        self._running        = False
        self._mqtt_connected = False
        self._emergency      = False

        self._last_action     = 0.0
        self._last_action_ts  = 0.0
        self._manual_action   = 0.0

        self._action_timeout = 5.0
        self._hb_interval    = 2.0
        self._last_hb_ts     = 0.0
        self._step           = 0

        # ✅ FIX: compatible with paho 1.6.1
        self._client = mqtt.Client(client_id="jetson_simulator")
        self._client.username_pw_set(self._user, self._pass)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        # auto reconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=5)

    # --------------------------------------------------------------
    def start(self):
        self._running = True
        print(f"[Simulator] Connecting MQTT → {self._broker}:{self._port}")

        try:
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            print(f"[Simulator] MQTT connect failed: {e}")

        self._run_loop()

    def stop(self):
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()
        print("[Simulator] Stopped")

    # --------------------------------------------------------------
    def _run_loop(self):
        print(f"[Simulator] Running — mode: {self._mode} | SP: {self._setpoint:.2f}")

        while self._running:
            start = time.time()
            self._step += 1
            level = self._tank.level

            if self._emergency:
                self._tank.step(0.0)
                self._publish_telemetry(level, 0.0)
                time.sleep(self._delay)
                continue

            action, source = self._compute_action(level)
            new_level = self._tank.step(action)
            self._last_action = action

            now = time.time()
            if now - self._last_hb_ts >= self._hb_interval:
                self._publish_heartbeat()
                self._last_hb_ts = now

            self._publish_telemetry(new_level, action)

            if self._step % 10 == 0:
                print(f"[{self._mode}|{source}] LV:{new_level:.3f} SP:{self._setpoint:.3f} ACT:{action:.3f}")

            time.sleep(max(0.0, self._delay - (time.time() - start)))

    # --------------------------------------------------------------
    def _compute_action(self, level):
        if self._mode == "MQTT_CONTROL":
            self._publish_state(level)

            if self._last_action_ts == 0.0:
                return self._pid.compute(level), "FALLBACK"

            if time.time() - self._last_action_ts > self._action_timeout:
                return self._pid.compute(level), "FALLBACK"

            return self._last_action, "SAC"

        elif self._mode == "RL":
            return self._rl.select_action(level, self._setpoint), "RL"

        elif self._mode == "PID":
            return self._pid.compute(level), "PID"

        elif self._mode == "MANUAL":
            return self._manual_action, "MANUAL"

        return 0.0, "NONE"

    # --------------------------------------------------------------
    def _publish(self, topic, payload):
        if not self._mqtt_connected:
            return
        self._client.publish(topic, json.dumps(payload), qos=1)

    def _publish_state(self, level):
        self._publish(self.TOPICS["state"], {
            "level": level,
            "setpoint": self._setpoint
        })

    def _publish_heartbeat(self):
        self._publish(self.TOPICS["heartbeat"], {"id": "jetson"})

    def _publish_telemetry(self, level, action):
        self._publish(self.TOPICS["telemetry"], {
            "level": level,
            "action": action,
            "mode": self._mode
        })

    # --------------------------------------------------------------
    # ✅ FIX: old API signature
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._mqtt_connected = True
            print("[Simulator] MQTT connected")

            client.subscribe(self.TOPICS["action"])
            client.subscribe(self.TOPICS["mode"])
            client.subscribe(self.TOPICS["emergency"])
            client.subscribe(self.TOPICS["manual_action"])
        else:
            print("[Simulator] connect fail", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._mqtt_connected = False
        print("[Simulator] MQTT disconnected")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except:
            return

        if msg.topic == self.TOPICS["action"]:
            self._last_action = float(data.get("action", 0))
            self._last_action_ts = time.time()


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="100.85.77.73")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--user", default="yessuskhonpui")
    parser.add_argument("--password", default="246810")
    parser.add_argument("--mode", default="MQTT_CONTROL")
    args = parser.parse_args()

    print(f"[Simulator] Broker: {args.broker}:{args.port}")

    sim = JetsonSimulator(
        broker=args.broker,
        port=args.port,
        user=args.user,
        password=args.password,
        initial_mode=args.mode,
    )

    try:
        sim.start()
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
