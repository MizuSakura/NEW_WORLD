# hardware/src/jetson_agent.py
"""
Jetson Agent
------------
ทำงานบน NVIDIA Jetson — ควบคุม RC Tank ผ่าน Modbus TCP

Mode:
    MQTT_CONTROL  — รับ action จาก Laptop ผ่าน MQTT
                    Laptop คำนวณ SAC → ส่งมา → Jetson เขียน Modbus
    STANDALONE    — คำนวณ PID เอง (fallback เมื่อ network ล่ม)
    MANUAL        — หยุดส่ง action (คนควบคุมเอง)

Reuse ไฟล์เดิม:
    ModbusTCP     ← comucation_modbusTCP_hardware.py
    get_hw_config ← hw_config_loader.py
    SafetyLayer   ← data_collector.py
    Watchdog      ← data_collector.py

MQTT Topics:
    subscribe: action, mode, emergency
    publish:   state, heartbeat, telemetry

วิธีรัน:
    python -m hardware.src.jetson_agent
    python -m hardware.src.jetson_agent --mode STANDALONE
    python -m hardware.src.jetson_agent --dry-run
"""

import argparse
import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

# ── path setup ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.src.utils.hw_config_loader import get_hw_config
from hardware.src.utils.comucation_modbusTCP_hardware import ModbusTCP
from hardware.src.data.data_collector import SafetyLayer, Watchdog


# ======================================================================
# PID Controller (standalone fallback)
# ======================================================================
class PIDController:

    def __init__(self, kp, ki, kd, setpoint,
                 output_min, output_max, sample_time=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sp = setpoint
        self.mn = output_min
        self.mx = output_max
        self.dt = sample_time
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, measurement: float) -> float:
        error            = self.sp - measurement
        self._integral  += error * self.dt
        self._integral   = float(np.clip(self._integral, self.mn, self.mx))
        derivative       = (error - self._prev_error) / self.dt
        self._prev_error = error
        output = (self.kp * error
                  + self.ki * self._integral
                  + self.kd * derivative)
        return float(np.clip(output, self.mn, self.mx))


# ======================================================================
# Jetson Agent
# ======================================================================
class JetsonAgent:

    MODES = ("MQTT_CONTROL", "STANDALONE", "MANUAL")

    def __init__(self, initial_mode: str = "MQTT_CONTROL",
                 dry_run: bool = False):

        # ── Load config ───────────────────────────────────────────
        self.cfg  = get_hw_config()
        mqtt_cfg  = self.cfg["mqtt"]
        mb_cfg    = self.cfg["modbus"]
        ctrl_cfg  = self.cfg["control"]
        pid_cfg   = self.cfg["pid"]

        # MQTT
        self._broker    = mqtt_cfg["broker"]
        self._port      = mqtt_cfg.get("port", 1883)
        self._qos       = mqtt_cfg.get("qos", 1)
        self._topics    = mqtt_cfg["topics"]
        self._device_id = self.cfg["device"]["id"]

        # Modbus
        self._addr_sensor   = mb_cfg["address_sensor"]
        self._addr_actuator = mb_cfg["address_actuator"]
        self._min_raw       = int(mb_cfg.get("min_raw", 0))
        self._max_raw       = int(mb_cfg.get("max_raw", 27647))

        # Control
        self._min_action     = float(ctrl_cfg["min_action"])
        self._max_action     = float(ctrl_cfg["max_action"])
        self._delay          = float(ctrl_cfg["delay_of_action"])
        self._dt             = float(ctrl_cfg.get("dt", 0.1))
        self._hb_interval    = float(ctrl_cfg.get("heartbeat_interval", 2.0))
        self._action_timeout = float(ctrl_cfg.get("heartbeat_timeout", 10.0))

        # State
        self._mode           = initial_mode
        self._running        = False
        self._emergency      = False
        self._last_action    = 0.0
        self._last_action_ts = 0.0
        self._last_hb_ts     = 0.0
        self._last_level     = 0.0
        self._setpoint       = float(pid_cfg.get("setpoint", 5.0))
        self._step           = 0
        self._episode        = 0
        self._cum_reward     = 0.0

        # ── Modbus driver ──────────────────────────────────────────
        if dry_run:
            from src.utils.mock_modbus import MockModbusTCP
            self._modbus = MockModbusTCP(
                host       = mb_cfg["host"],
                port       = mb_cfg["port"],
                min_raw    = self._min_raw,
                max_raw    = self._max_raw,
                action_max = self._max_action,
                verbose    = True,
            )
            print("[JetsonAgent] DRY RUN — MockModbusTCP active")
        else:
            self._modbus = ModbusTCP(
                host = mb_cfg["host"],
                port = mb_cfg.get("port", 502)
            )

        # ── Safety + Watchdog (reuse จาก data_collector.py) ───────
        self._safety = SafetyLayer(
            min_action = self._min_action,
            max_action = self._max_action,
            max_sensor = self._max_action * 1.1,
            max_error  = self._max_action * 2.0,
        )
        self._watchdog = Watchdog(timeout=5.0)

        # ── PID ───────────────────────────────────────────────────
        self._pid = PIDController(
            kp          = float(pid_cfg.get("kp",         2.0)),
            ki          = float(pid_cfg.get("ki",         0.1)),
            kd          = float(pid_cfg.get("kd",         0.05)),
            setpoint    = self._setpoint,
            output_min  = float(pid_cfg.get("output_min", self._min_action)),
            output_max  = float(pid_cfg.get("output_max", self._max_action)),
            sample_time = self._dt,
        )

        # ── MQTT client ───────────────────────────────────────────
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self._device_id
        )
        username = mqtt_cfg.get("username", "")
        password = mqtt_cfg.get("password", "")
        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def start(self):
        print(f"[JetsonAgent] Connecting Modbus → {self.cfg['modbus']['host']}")
        if not self._modbus.connect():
            raise ConnectionError("Cannot connect to Modbus hardware")
        print("[JetsonAgent] Modbus connected")

        print(f"[JetsonAgent] Connecting MQTT → {self._broker}:{self._port}")
        self._client.connect(self._broker, self._port, keepalive=60)
        self._client.loop_start()

        self._running = True
        print(f"[JetsonAgent] Running — mode: {self._mode}")
        self._control_loop()

    def stop(self):
        self._running = False
        self._safe_shutdown()
        self._client.loop_stop()
        self._client.disconnect()
        self._modbus.disconnect()
        print("[JetsonAgent] Stopped")

    @property
    def mode(self) -> str:
        return self._mode

    def status(self) -> dict:
        return {
            "mode":      self._mode,
            "running":   self._running,
            "emergency": self._emergency,
            "level":     round(self._last_level, 3),
            "setpoint":  round(self._setpoint,   3),
            "action":    round(self._last_action, 3),
            "step":      self._step,
            "episode":   self._episode,
        }

    # ──────────────────────────────────────────────────────────────
    # Control Loop
    # ──────────────────────────────────────────────────────────────

    def _control_loop(self):
        while self._running:
            loop_start = time.time()

            # emergency → รอ
            if self._emergency:
                time.sleep(self._delay)
                continue

            # 1. อ่าน sensor
            level = self._read_sensor()
            if level is None:
                print("[JetsonAgent] Sensor read failed — skip")
                time.sleep(self._delay)
                continue

            self._last_level = level
            self._step      += 1
            self._watchdog.kick()

            # 2. heartbeat
            now = time.time()
            if now - self._last_hb_ts >= self._hb_interval:
                self._publish_heartbeat()
                self._last_hb_ts = now

            # 3. เลือก action
            if self._mode == "MQTT_CONTROL":
                action = self._get_action_mqtt(level)
            elif self._mode == "STANDALONE":
                action = self._pid.compute(level)
            else:
                # MANUAL
                time.sleep(self._delay)
                continue

            # 4. Safety clamp
            action = self._safety.clamp_action(action)

            # 5. Check sensor range
            try:
                self._safety.check_sensor(level)
            except RuntimeError as e:
                print(f"[JetsonAgent] Safety violation: {e}")
                self._trigger_emergency()
                continue

            # 6. เขียน actuator
            self._write_actuator(action)
            self._last_action = action

            # 7. Telemetry
            error         = self._setpoint - level
            reward        = -abs(error)
            self._cum_reward += reward
            self._publish_telemetry(level, action, error, reward)

            # 8. Print status ทุก 10 steps
            if self._step % 10 == 0:
                print(
                    f"[{self._mode}] "
                    f"LV:{level:.3f} SP:{self._setpoint:.3f} "
                    f"ACT:{action:.3f} ERR:{error:.3f} "
                    f"STEP:{self._step}",
                    flush=True
                )

            # 9. รักษา timing
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self._delay - elapsed))

    # ──────────────────────────────────────────────────────────────
    # Action: MQTT_CONTROL
    # ──────────────────────────────────────────────────────────────

    def _get_action_mqtt(self, level: float) -> float:
        """ส่ง state ไป Laptop → ใช้ action ล่าสุด / fallback PID"""
        # publish state
        self._publish(self._topics["state"], {
            "level":    float(level),
            "setpoint": float(self._setpoint),
            "step":     self._step,
            "episode":  self._episode,
            "ts":       datetime.now().isoformat(),
        })

        # ถ้า action timeout → PID fallback
        if self._last_action_ts == 0.0:
            return self._pid.compute(level)

        age = time.time() - self._last_action_ts
        if age > self._action_timeout:
            print(f"[JetsonAgent] Action timeout ({age:.1f}s) → PID")
            return self._pid.compute(level)

        return self._last_action

    # ──────────────────────────────────────────────────────────────
    # Modbus IO  (reuse ModbusTCP จาก comucation_modbusTCP_hardware)
    # ──────────────────────────────────────────────────────────────

    def _read_sensor(self) -> float | None:
        """analog_read → scale เป็น engineering unit"""
        try:
            raw = self._modbus.analog_read(address=self._addr_sensor)
            if raw is None:
                return None
            return float(np.interp(
                raw,
                [self._min_raw, self._max_raw],
                [self._min_action, self._max_action]
            ))
        except Exception as e:
            print(f"[JetsonAgent] Sensor error: {e}")
            return None

    def _write_actuator(self, action: float):
        """scale action → raw → write_holding_register"""
        try:
            raw = int(np.interp(
                action,
                [self._min_action, self._max_action],
                [self._min_raw,    self._max_raw]
            ))
            self._modbus.write_holding_register(
                address=self._addr_actuator, value=raw)
        except Exception as e:
            print(f"[JetsonAgent] Actuator error: {e}")

    def _safe_shutdown(self):
        """ส่ง 0 ไป actuator"""
        try:
            self._modbus.write_holding_register(
                address=self._addr_actuator, value=0)
            print("[JetsonAgent] Actuator set to 0")
        except Exception as e:
            print(f"[JetsonAgent] Safe shutdown error: {e}")

    def _emergency_shutdown(self):
        """ปิด coils ทั้งหมด + reset actuator"""
        print("[JetsonAgent] *** EMERGENCY SHUTDOWN ***")
        # coil range จาก hardware.yaml (ถ้ามี)
        ctrl = self.cfg.get("control", {})
        c_start = ctrl.get("emergency_coil_start", "0x4000")
        c_end   = ctrl.get("emergency_coil_end",   "0x40FF")
        try:
            s = int(c_start, 16) if isinstance(c_start, str) else c_start
            e = int(c_end,   16) if isinstance(c_end,   str) else c_end
            for addr in range(s, e + 1):
                try:
                    self._modbus.digital_write(addr, 0)
                except Exception:
                    pass
        except Exception as ex:
            print(f"[JetsonAgent] Coil shutdown error: {ex}")
        self._safe_shutdown()

    def _trigger_emergency(self):
        self._emergency = True
        self._emergency_shutdown()

    # ──────────────────────────────────────────────────────────────
    # MQTT Callbacks
    # ──────────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self._topics["action"],    self._qos)
            client.subscribe(self._topics["mode"],      self._qos)
            client.subscribe(self._topics["emergency"], 2)
            print("[JetsonAgent] MQTT connected — subscribed: action, mode, emergency")
        else:
            print(f"[JetsonAgent] MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"[JetsonAgent] MQTT disconnected rc={rc}")
        if self._mode == "MQTT_CONTROL":
            print("[JetsonAgent] Network lost → STANDALONE fallback")
            self._mode = "STANDALONE"
            self._pid.reset()

    def _on_message(self, client, userdata, msg):
        data = self._parse_payload(msg.payload.decode("utf-8"))
        if data is None:
            return
        topic = msg.topic
        if topic == self._topics["action"]:
            self._handle_action(data)
        elif topic == self._topics["mode"]:
            self._handle_mode(data)
        elif topic == self._topics["emergency"]:
            self._handle_emergency(data)

    # ──────────────────────────────────────────────────────────────
    # Message Handlers
    # ──────────────────────────────────────────────────────────────

    def _handle_action(self, data: dict):
        action = float(np.clip(
            float(data.get("action", self._last_action)),
            self._min_action, self._max_action
        ))
        self._last_action    = action
        self._last_action_ts = time.time()

        # detect new episode
        ep = int(data.get("episode", self._episode))
        if ep != self._episode:
            self._episode    = ep
            self._cum_reward = 0.0
            self._pid.reset()

    def _handle_mode(self, data: dict):
        mode = str(data.get("mode", self._mode)).upper()
        if mode in self.MODES:
            print(f"[JetsonAgent] Mode: {self._mode} → {mode}")
            self._mode = mode
            if mode == "STANDALONE":
                self._pid.reset()
        else:
            print(f"[JetsonAgent] Unknown mode: {mode}")

    def _handle_emergency(self, data: dict):
        cmd = str(data.get("command", ""))
        if cmd == "EMERGENCY_STOP":
            self._trigger_emergency()
        elif cmd == "CLEAR":
            self._emergency = False
            print("[JetsonAgent] Emergency cleared")

    # ──────────────────────────────────────────────────────────────
    # MQTT Publish
    # ──────────────────────────────────────────────────────────────

    def _publish_heartbeat(self):
        self._publish(self._topics["heartbeat"], {
            "id":   self._device_id,
            "mode": self._mode,
            "ts":   datetime.now().isoformat(),
        })

    def _publish_telemetry(self, level, action, error, reward):
        self._publish(self._topics["telemetry"], {
            "level":      float(level),
            "setpoint":   float(self._setpoint),
            "action":     float(action),
            "error":      round(float(error), 4),
            "reward":     round(float(reward), 4),
            "cum_reward": round(float(self._cum_reward), 3),
            "step":       self._step,
            "episode":    self._episode,
            "mode":       self._mode,
            "ts":         datetime.now().isoformat(),
        })

    def _publish(self, topic: str, payload: dict):
        try:
            self._client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False),
                qos=self._qos
            )
        except Exception as e:
            print(f"[JetsonAgent] Publish error: {e}")

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_payload(payload: str) -> dict | None:
        """Parse JSON — รองรับ unquoted keys {level:3.5,...}"""
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            try:
                data  = {}
                clean = payload.strip('{}')
                for part in clean.split(','):
                    if ':' not in part:
                        continue
                    k, v = part.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        data[k] = float(v)
                    except ValueError:
                        data[k] = v
                return data if data else None
            except Exception:
                print(f"[JetsonAgent] Invalid payload: {payload}")
                return None


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="JetsonAgent — RC Tank control via MQTT + Modbus"
    )
    parser.add_argument(
        "--mode",
        default="MQTT_CONTROL",
        choices=["MQTT_CONTROL", "STANDALONE", "MANUAL"],
        help="Initial mode (default: MQTT_CONTROL)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="จำลอง hardware ด้วย MockModbusTCP"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("[JetsonAgent] RC Tank Control Agent")
    print(f"[JetsonAgent] Mode    : {args.mode}")
    print(f"[JetsonAgent] Dry run : {args.dry_run}")
    print("=" * 55)

    agent = JetsonAgent(
        initial_mode = args.mode,
        dry_run      = args.dry_run,
    )

    try:
        agent.start()
    except KeyboardInterrupt:
        print("\n[JetsonAgent] Interrupted")
    except Exception as e:
        print(f"[JetsonAgent] Fatal: {e}")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()