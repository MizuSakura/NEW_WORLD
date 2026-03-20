# src/controller/mqtt_controller.py
"""
MQTT Controller
---------------
Laptop ทำหน้าที่เป็น Controller
- รับ state จาก Jetson ผ่าน MQTT topic: state
- คำนวณ action ด้วย SAC model
- ส่ง action กลับไป Jetson ผ่าน MQTT topic: action
- ส่ง telemetry ไป dashboard ผ่าน MQTT topic: telemetry
- monitor heartbeat จาก Jetson

Flow:
    Jetson → state topic → MQTTController
                                ↓
                        SAC.select_action(state)
                                ↓
    Jetson ← action topic ← MQTTController
                                ↓
    Dashboard ← telemetry topic (reward, error, level)
"""

import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

import paho.mqtt.client as mqtt

from src.agent.SAC_Agent import SACAgent
from src.environment.state_builder import StateBuilder
from src.environment.reward_function_control import Reward_manager


class MQTTController:
    """
    SAC Agent ที่ทำงานผ่าน MQTT
    รับ state จาก Jetson → คำนวณ action → ส่งกลับ Jetson
    """

    def __init__(
        self,
        broker:        str,
        port:          int,
        topics:        dict,
        agent:         SACAgent,
        state_builder: StateBuilder,
        qos:           int   = 1,
        heartbeat_timeout: float = 10.0,
        deterministic: bool  = True,
    ):
        self.broker   = broker
        self.port     = port
        self.topics   = topics          # dict จาก network.yaml mqtt.topics
        self.agent    = agent
        self.sb       = state_builder
        self.qos      = qos
        self.hb_timeout    = heartbeat_timeout
        self.deterministic = deterministic

        # Internal state
        self._running       = False
        self._connected     = False
        self._last_hb       = 0.0
        self._step          = 0
        self._episode       = 0
        self._reward_mgr    = Reward_manager(buffer_size=5)
        self._last_level    = 0.0
        self._last_setpoint = 0.0
        self._last_action   = 0.0
        self._cum_reward    = 0.0

        # Status callback (optional) — ใช้ broadcast ไป dashboard
        self.on_status: callable = None   # fn(dict)

        # MQTT client
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                                   client_id="laptop_controller")
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        # heartbeat watchdog thread
        self._hb_thread = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def start(self):
        """เชื่อมต่อ MQTT แล้วเริ่ม loop"""
        self._running = True
        self._client.connect(self.broker, self.port, keepalive=60)
        self._client.loop_start()

        # heartbeat watchdog
        self._hb_thread = threading.Thread(
            target=self._heartbeat_watchdog, daemon=True)
        self._hb_thread.start()

        self._notify({
            "event":   "mqtt_ctrl_started",
            "message": f"Controller started → {self.broker}:{self.port}",
        })
        print(f"[MQTTController] Started → {self.broker}:{self.port}")

    def stop(self):
        """หยุด controller"""
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()
        self._notify({
            "event":   "mqtt_ctrl_stopped",
            "message": "Controller stopped",
        })
        print("[MQTTController] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected

    def status(self) -> dict:
        hb_age = time.time() - self._last_hb if self._last_hb > 0 else -1
        return {
            "running":       self._running,
            "connected":     self._connected,
            "jetson_alive":  hb_age < self.hb_timeout if hb_age >= 0 else False,
            "heartbeat_age": round(hb_age, 1),
            "episode":       self._episode,
            "step":          self._step,
            "last_level":    self._last_level,
            "last_setpoint": self._last_setpoint,
            "last_action":   self._last_action,
            "cum_reward":    round(self._cum_reward, 3),
        }

    # ──────────────────────────────────────────────
    # MQTT Callbacks
    # ──────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            r1 = client.subscribe(self.topics["state"],     self.qos)
            r2 = client.subscribe(self.topics["heartbeat"], self.qos)
            print(f"[DEBUG] subscribe state result: {r1}")
            print(f"[DEBUG] subscribe heartbeat result: {r2}")
            print(f"[MQTTController] Connected, subscribed state+heartbeat")
            self._notify({
                "event":   "mqtt_ctrl_connected",
                "message": "Connected to broker",
            })
        else:
            print(f"[MQTTController] Connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTTController] Disconnected rc={rc}")
        self._notify({
            "event":   "mqtt_ctrl_disconnected",
            "message": f"Disconnected rc={rc}",
        })

    def _on_message(self, client, userdata, msg):
        topic   = msg.topic
        payload = msg.payload.decode("utf-8")
        print(f"[DEBUG] received: {topic} → {payload}")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            try:
                # manual parse สำหรับ unquoted JSON เช่น {level:3.5,setpoint:5.0}
                data = {}
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
                if not data:
                    raise ValueError("empty parse result")
            except Exception:
                print(f"[MQTTController] Invalid payload: {payload}")
                return

        if topic == self.topics["state"]:
            self._handle_state(data)
        elif topic == self.topics["heartbeat"]:
            self._handle_heartbeat(data)

    # ──────────────────────────────────────────────
    # State Handler — หัวใจหลัก
    # ──────────────────────────────────────────────

    def _handle_state(self, data: dict):
        """
        รับ state จาก Jetson แล้วคำนวณ action ส่งกลับ

        Expected payload:
        {
            "level":    float,
            "setpoint": float,
            "step":     int,
            "episode":  int   (optional)
        }
        """
        if not self._running:
            return

        level    = float(data.get("level",    0.0))
        setpoint = float(data.get("setpoint", 5.0))
        step     = int(data.get("step",       0))
        episode  = int(data.get("episode",    self._episode))

        # detect new episode → reset StateBuilder
        if episode != self._episode:
            self._episode    = episode
            self._cum_reward = 0.0
            self.sb.reset(
                level    = level,
                action   = self._last_action,
                setpoint = setpoint,
                dt       = 0.1,
            )
            self._reward_mgr.reset(
                init_setpoint = setpoint,
                init_state    = level,
                init_action   = self._last_action,
            )

        self._step          = step
        self._last_level    = level
        self._last_setpoint = setpoint

        # สร้าง canonical state
        state = self.sb.update(
            level    = level,
            action   = self._last_action,
            setpoint = setpoint,
        )

        # คำนวณ action ด้วย SAC model
        action = self.agent.select_action(
            state, deterministic=self.deterministic
        )
        action_val = float(np.clip(
            np.array(action).item(),
            0.0,
            self.sb.level_max
        ))

        self._last_action = action_val

        # publish action กลับไป Jetson
        self._publish(self.topics["action"], {
            "action":  action_val,
            "step":    step,
            "episode": episode,
            "ts":      datetime.now().isoformat(),
        })

        # คำนวณ reward + telemetry
        self._reward_mgr.update(
            setpoint = setpoint,
            state    = level,
            action   = action_val,
        )
        reward = self._reward_mgr.reward_continuous_control()
        self._cum_reward += reward

        error = setpoint - level

        # publish telemetry ไป dashboard
        self._publish(self.topics["telemetry"], {
            "level":      float(level),
            "setpoint":   float(setpoint),
            "action":     float(action_val),
            "error":      round(float(error), 4),
            "reward":     round(float(reward), 4),
            "cum_reward": round(float(self._cum_reward), 3),
            "step":       step,
            "episode":    episode,
            "ts":         datetime.now().isoformat(),
        })

        # notify dashboard
        self._notify({
            "event":      "mqtt_ctrl_step",
            "level":      float(level),
            "setpoint":   float(setpoint),
            "action":     float(action_val),
            "error":      round(float(error), 4),
            "reward":     round(float(reward), 4),
            "cum_reward": round(float(self._cum_reward), 3),
            "step":       step,
            "episode":    episode,
        })

    def _handle_heartbeat(self, data: dict):
        """อัปเดต timestamp heartbeat ล่าสุด"""
        self._last_hb = time.time()
        self._notify({
            "event":     "mqtt_ctrl_heartbeat",
            "ts":        data.get("ts", ""),
            "jetson_id": data.get("id", "unknown"),
        })

    # ──────────────────────────────────────────────
    # Heartbeat Watchdog
    # ──────────────────────────────────────────────

    def _heartbeat_watchdog(self):
        """ตรวจสอบว่า Jetson ยังส่ง heartbeat อยู่ไหม"""
        while self._running:
            time.sleep(2.0)
            if self._last_hb > 0:
                age = time.time() - self._last_hb
                if age > self.hb_timeout:
                    self._notify({
                        "event":   "mqtt_ctrl_jetson_timeout",
                        "message": f"Jetson heartbeat timeout ({age:.1f}s)",
                        "age":     round(age, 1),
                    })

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _publish(self, topic: str, payload: dict):
        self._client.publish(
            topic,
            json.dumps(payload, ensure_ascii=False),
            qos=self.qos
        )

    def _notify(self, data: dict):
    # print เป็น JSON line → server อ่านผ่าน stdout
        print(json.dumps(data, ensure_ascii=False), flush=True)

        # callback ยังคงไว้สำหรับกรณีรันตรง (standalone)
        if self.on_status is not None:
            try:
                self.on_status(data)
            except Exception as e:
                print(json.dumps({"event": "error", "message": str(e)}), flush=True)


# ======================================================
# Factory function — สร้าง MQTTController จาก yaml
# ======================================================

def create_controller(
    network_cfg_path: Path,
    rl_cfg_path:      Path,
    eval_cfg_path:    Path,
) -> "MQTTController":
    """
    สร้าง MQTTController จาก config files

    Parameters
    ----------
    network_cfg_path : Path  → src/API/config/network.yaml
    rl_cfg_path      : Path  → src/API/config/rl_params.yaml
    eval_cfg_path    : Path  → src/API/config/eval_params.yaml

    Returns
    -------
    MQTTController พร้อมใช้งาน (ยังไม่ได้ start)
    """
    import yaml
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # ── โหลด network config ──────────────────────────────
    with open(network_cfg_path, "r", encoding="utf-8") as f:
        net_cfg = yaml.safe_load(f)

    mqtt_cfg = net_cfg["mqtt"]
    topics   = mqtt_cfg["topics"]
    broker   = mqtt_cfg["broker"]
    port     = mqtt_cfg.get("port", 1883)
    qos      = mqtt_cfg.get("qos", 1)

    # ── โหลด rl config ────────────────────────────────────
    # ใช้ RLConfig (pydantic) เพื่อ validate + coerce types
    from src.API.src_api.rl_schema import RLConfig
    with open(rl_cfg_path, "r", encoding="utf-8") as f:
        rl_data = yaml.safe_load(f)
    rl_cfg = RLConfig(**rl_data)

    # ── โหลด eval config — หา model path ─────────────────
    with open(eval_cfg_path, "r", encoding="utf-8") as f:
        eval_data = yaml.safe_load(f)

    mqtt_ctrl_cfg  = eval_data.get("mqtt_ctrl", eval_data.get("gym", {}))
    model_path_str = mqtt_ctrl_cfg.get("model_path", "models/checkpoint/Autosave.pt")

    # FIX: สร้าง Path ก่อน แล้วค่อยเช็ค is_absolute()
    model_path = Path(model_path_str)
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".pt")
    if not model_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        model_path   = project_root / model_path

    print(f"[Factory] Model path: {model_path}")

    # ── สร้าง StateBuilder จาก rl config ─────────────────
    state_builder = StateBuilder(rl_cfg.state.model_dump())
    print(f"[Factory] {state_builder}")

    # ── สร้าง SAC Agent จาก rl config ────────────────────
    state_dim  = rl_cfg.state.state_dim
    action_dim = 1
    min_action = np.array([0.0])
    max_action = np.array([rl_cfg.state.level_max])

    agent = SACAgent.from_config(
        rl_cfg     = rl_cfg,
        state_dim  = state_dim,
        action_dim = action_dim,
        min_action = min_action,
        max_action = max_action,
    )
    agent.load_model(path=model_path)
    print(f"[Factory] Model loaded: {model_path}")

    return MQTTController(
        broker        = broker,
        port          = port,
        topics        = topics,
        agent         = agent,
        state_builder = state_builder,
        qos           = qos,
        deterministic = mqtt_ctrl_cfg.get("deterministic", True),
    )


# ======================================================
# Standalone test
# ======================================================

if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    ctrl = create_controller(
        network_cfg_path = PROJECT_ROOT / "src/API/config/network.yaml",
        rl_cfg_path      = PROJECT_ROOT / "src/API/config/rl_params.yaml",
        eval_cfg_path    = PROJECT_ROOT / "src/API/config/eval_params.yaml",
    )

    def on_status(d):
        print(f"[STATUS] {d}")

    ctrl.on_status = on_status
    ctrl.start()

    print("Controller running... Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
            s = ctrl.status()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"EP:{s['episode']} STEP:{s['step']} "
                f"LV:{s['last_level']:.2f} SP:{s['last_setpoint']:.2f} "
                f"ACT:{s['last_action']:.2f} "
                f"Jetson:{'✓' if s['jetson_alive'] else '✗'}"
            )
    except KeyboardInterrupt:
        ctrl.stop()
        print("Stopped.")