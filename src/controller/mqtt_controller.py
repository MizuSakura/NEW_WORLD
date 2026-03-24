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

Model path priority:
    1. MQTT_CTRL_MODEL_PATH env var  ← dashboard model selector
    2. eval_params.yaml mqtt_ctrl.model_path
    3. default: models/checkpoint/Autosave.pt

State config priority:
    1. hyperparams ใน model checkpoint  ← auto-detect dim
    2. rl_params.yaml + infer ถ้า dim ไม่ตรง

Bug fixes:
    - username_pw_set ต้องเรียกหลัง สร้าง self._client
    - auto-reset StateBuilder ก่อน step แรก
    - _on_connect signature VERSION2 ต้องมี properties=None
"""

import json
import os
import copy
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import paho.mqtt.client as mqtt

from src.agent.SAC_Agent import SACAgent
from src.environment.state_builder import StateBuilder
from src.environment.reward_function_control import Reward_manager


# ======================================================
# State Config Helpers
# ======================================================

def _calc_state_dim(cfg: dict) -> int:
    d  = cfg.get("level_history",    3)
    d += cfg.get("action_history",   3)
    if cfg.get("error_history",    0) > 0: d += cfg["error_history"]
    if cfg.get("setpoint_history", 0) > 0: d += cfg["setpoint_history"]
    if cfg.get("integral",   False): d += 1
    if cfg.get("derivative", False): d += 1
    return d


def _infer_state_config(target_dim: int, base_cfg: dict) -> dict:
    """Infer state config ให้ได้ dim ตรงกับ model"""
    cfg = copy.deepcopy(base_cfg)

    # Pass 1: ปรับ error_history
    for err_h in range(4):
        cfg["error_history"] = err_h
        if _calc_state_dim(cfg) == target_dim:
            print(f"[Factory] → inferred error_history={err_h}", flush=True)
            return cfg

    # Pass 2: ปรับ error + setpoint history
    for sp_h in range(4):
        for err_h in range(4):
            cfg["setpoint_history"] = sp_h
            cfg["error_history"]    = err_h
            if _calc_state_dim(cfg) == target_dim:
                print(f"[Factory] → inferred setpoint_history={sp_h}, "
                      f"error_history={err_h}", flush=True)
                return cfg

    # Pass 3: ปรับ integral/derivative ด้วย
    for intg in [True, False]:
        for deriv in [True, False]:
            for err_h in range(4):
                cfg["integral"]      = intg
                cfg["derivative"]    = deriv
                cfg["error_history"] = err_h
                if _calc_state_dim(cfg) == target_dim:
                    print(f"[Factory] → inferred integral={intg}, "
                          f"derivative={deriv}, error_history={err_h}",
                          flush=True)
                    return cfg

    print(f"[Factory] ✗ Cannot infer config for dim={target_dim} "
          f"— using yaml as-is", flush=True)
    return copy.deepcopy(base_cfg)


# ======================================================
# MQTTController
# ======================================================

class MQTTController:

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
        username:      str   = "",
        password:      str   = "",
    ):
        self.broker        = broker
        self.port          = port
        self.topics        = topics
        self.agent         = agent
        self.sb            = state_builder
        self.qos           = qos
        self.hb_timeout    = heartbeat_timeout
        self.deterministic = deterministic

        self._running        = False
        self._connected      = False
        self._initialized    = False   # ← flag: reset แล้วหรือยัง
        self._last_hb        = 0.0
        self._step           = 0
        self._episode        = 0
        self._reward_mgr     = Reward_manager(buffer_size=5)
        self._last_level     = 0.0
        self._last_setpoint  = 0.0
        self._last_action    = 0.0
        self._cum_reward     = 0.0

        self.on_status: callable = None

        # ── สร้าง client ก่อน แล้วค่อย set credentials ──────────
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="laptop_controller"
        )
        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message
        self._hb_thread = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def start(self):
        self._running = True
        self._client.connect(self.broker, self.port, keepalive=60)
        self._client.loop_start()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_watchdog, daemon=True)
        self._hb_thread.start()
        self._notify({
            "event":   "mqtt_ctrl_started",
            "message": f"Controller started → {self.broker}:{self.port}",
        })
        print(f"[MQTTController] Started → {self.broker}:{self.port}",
              flush=True)

    def stop(self):
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()
        self._notify({
            "event":   "mqtt_ctrl_stopped",
            "message": "Controller stopped",
        })
        print("[MQTTController] Stopped", flush=True)

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

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._connected = True
            # subscribe ทั้งคู่ใน call เดียว
            client.subscribe([
                (self.topics["state"],     self.qos),
                (self.topics["heartbeat"], self.qos),
            ])
            print("[MQTTController] Connected — subscribed: state, heartbeat",
                  flush=True)
            self._notify({
                "event":   "mqtt_ctrl_connected",
                "message": "Connected to broker",
            })
        else:
            print(f"[MQTTController] Connect failed rc={rc}", flush=True)

    def _on_disconnect(self, client, userdata, rc, properties=None,
                       reasoncode=None):
        self._connected  = False
        self._initialized = False  # ← reset flag เพื่อ re-init เมื่อ reconnect
        print(f"[MQTTController] Disconnected rc={rc}", flush=True)
        self._notify({
            "event":   "mqtt_ctrl_disconnected",
            "message": f"Disconnected rc={rc}",
        })

    def _on_message(self, client, userdata, msg):
        topic   = msg.topic
        payload = msg.payload.decode("utf-8")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            try:
                data  = {}
                clean = payload.strip('{}')
                for part in clean.split(','):
                    if ':' not in part:
                        continue
                    k, v = part.split(':', 1)
                    k = k.strip(); v = v.strip()
                    try:
                        data[k] = float(v)
                    except ValueError:
                        data[k] = v
                if not data:
                    raise ValueError("empty parse")
            except Exception:
                print(f"[MQTTController] Invalid payload: {payload}",
                      flush=True)
                return

        if topic == self.topics["state"]:
            self._handle_state(data)
        elif topic == self.topics["heartbeat"]:
            self._handle_heartbeat(data)

    # ──────────────────────────────────────────────
    # State Handler
    # ──────────────────────────────────────────────

    def _handle_state(self, data: dict):
        if not self._running:
            return

        level    = float(data.get("level",    0.0))
        setpoint = float(data.get("setpoint", 5.0))
        step     = int(data.get("step",       0))
        episode  = int(data.get("episode",    self._episode))

        # ── auto-reset ครั้งแรก (ก่อน step แรกหรือหลัง reconnect) ──
        if not self._initialized:
            print(f"[MQTTController] Initializing StateBuilder "
                  f"(level={level:.2f}, sp={setpoint:.2f})", flush=True)
            self.sb.reset(
                level    = level,
                action   = 0.0,
                setpoint = setpoint,
                dt       = 0.1,
            )
            self._reward_mgr.reset(
                init_setpoint = setpoint,
                init_state    = level,
                init_action   = 0.0,
            )
            self._episode    = episode
            self._cum_reward = 0.0
            self._initialized = True

        # ── detect new episode ────────────────────────────────────
        elif episode != self._episode:
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

        # ── สร้าง state ───────────────────────────────────────────
        state = self.sb.update(
            level    = level,
            action   = self._last_action,
            setpoint = setpoint,
        )

        # ── SAC inference ─────────────────────────────────────────
        try:
            action = self.agent.select_action(
                state, deterministic=self.deterministic)
            action_val = float(np.clip(
                np.array(action).item(), 0.0, self.sb.level_max))
        except Exception as e:
            print(f"[MQTTController] inference error: {e}", flush=True)
            return

        self._last_action = action_val

        # ── publish action → Jetson ───────────────────────────────
        self._publish(self.topics["action"], {
            "action":  action_val,
            "step":    step,
            "episode": episode,
            "ts":      datetime.now().isoformat(),
        })

        # ── reward ────────────────────────────────────────────────
        self._reward_mgr.update(
            setpoint = setpoint,
            state    = level,
            action   = action_val,
        )
        reward = self._reward_mgr.reward_continuous_control()
        self._cum_reward += reward
        error = setpoint - level

        # ── publish telemetry → dashboard ─────────────────────────
        self._publish(self.topics["telemetry"], {
            "level":      float(level),
            "setpoint":   float(setpoint),
            "action":     float(action_val),
            "error":      round(float(error), 4),
            "reward":     round(float(reward), 4),
            "cum_reward": round(float(self._cum_reward), 3),
            "step":       step,
            "episode":    episode,
            "mode":       "MQTT_CONTROL",
            "ts":         datetime.now().isoformat(),
        })

        # ── notify dashboard ──────────────────────────────────────
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
            "mode":       "MQTT_CONTROL",
        })

    def _handle_heartbeat(self, data: dict):
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
        try:
            self._client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False),
                qos=self.qos
            )
        except Exception as e:
            print(f"[MQTTController] Publish error: {e}", flush=True)

    def _notify(self, data: dict):
        print(json.dumps(data, ensure_ascii=False), flush=True)
        if self.on_status is not None:
            try:
                self.on_status(data)
            except Exception as e:
                print(json.dumps({"event": "error", "message": str(e)}),
                      flush=True)


# ======================================================
# Factory function
# ======================================================

def create_controller(
    network_cfg_path: Path,
    rl_cfg_path:      Path,
    eval_cfg_path:    Path,
) -> "MQTTController":
    import yaml
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # ── network config ────────────────────────────────────
    with open(network_cfg_path, "r", encoding="utf-8") as f:
        net_cfg = yaml.safe_load(f)
    mqtt_cfg = net_cfg["mqtt"]
    topics   = mqtt_cfg["topics"]
    broker   = mqtt_cfg["broker"]
    port     = mqtt_cfg.get("port", 1883)
    qos      = mqtt_cfg.get("qos", 1)
    username = mqtt_cfg.get("username", "") or ""
    password = mqtt_cfg.get("password", "") or ""

    # ── rl config ─────────────────────────────────────────
    from src.API.src_api.rl_schema import RLConfig
    with open(rl_cfg_path, "r", encoding="utf-8") as f:
        rl_data = yaml.safe_load(f)
    rl_cfg = RLConfig(**rl_data)

    # ── eval config ───────────────────────────────────────
    with open(eval_cfg_path, "r", encoding="utf-8") as f:
        eval_data = yaml.safe_load(f)
    mqtt_ctrl_cfg = eval_data.get("mqtt_ctrl", eval_data.get("gym", {}))

    # ── Model path priority ───────────────────────────────
    project_root = Path(__file__).resolve().parents[2]

    env_model = os.environ.get("MQTT_CTRL_MODEL_PATH", "").strip()
    if env_model:
        model_path = Path(env_model)
        if not model_path.is_absolute():
            candidates = [
                Path(r"E:\server_Project\SER_VER_STORE\rc_models") / model_path,
                project_root / "models" / model_path,
                project_root / model_path,
            ]
            model_path = next(
                (p for p in candidates if p.exists()), candidates[0])
        print(f"[Factory] Model from env  : {model_path}", flush=True)
    else:
        model_path_str = mqtt_ctrl_cfg.get(
            "model_path", "models/checkpoint/Autosave.pt")
        model_path = Path(model_path_str)
        if model_path.suffix == "":
            model_path = model_path.with_suffix(".pt")
        if not model_path.is_absolute():
            model_path = project_root / model_path
        print(f"[Factory] Model from yaml  : {model_path}", flush=True)

    # ── โหลด checkpoint + inspect hyperparams ─────────────
    print("[Factory] Loading checkpoint...", flush=True)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    hp   = ckpt.get("hyperparams", {})

    print("[Factory] ── Model hyperparams ─────────────────", flush=True)
    for k, v in hp.items():
        print(f"[Factory]   {k}: {v}", flush=True)

    yaml_state     = rl_cfg.state.model_dump()
    yaml_state_dim = rl_cfg.state.state_dim
    print("[Factory] ── rl_params.yaml state ─────────────", flush=True)
    for k, v in yaml_state.items():
        print(f"[Factory]   {k}: {v}", flush=True)
    print(f"[Factory]   → calc state_dim: {yaml_state_dim}", flush=True)

    # ── Auto-detect state config จาก model ───────────────
    model_state_dim = hp.get("state_dim", None)

    if model_state_dim and model_state_dim != yaml_state_dim:
        print(f"[Factory] ⚠  MISMATCH — "
              f"model={model_state_dim} vs yaml={yaml_state_dim}", flush=True)
        state_cfg    = _infer_state_config(model_state_dim, yaml_state)
        inferred_dim = _calc_state_dim(state_cfg)
        print(f"[Factory] → Using inferred config (dim={inferred_dim})",
              flush=True)
    else:
        state_cfg = yaml_state
        print(f"[Factory] ✓ state_dim match: {yaml_state_dim}", flush=True)

    # ── StateBuilder ──────────────────────────────────────
    state_builder = StateBuilder(state_cfg)
    print(f"[Factory] {state_builder}", flush=True)

    # ── SAC Agent ─────────────────────────────────────────
    actual_dim = _calc_state_dim(state_cfg)
    action_dim = 1
    min_action = np.array([0.0])
    max_action = np.array([state_cfg.get("level_max", rl_cfg.state.level_max)])

    agent = SACAgent.from_config(
        rl_cfg     = rl_cfg,
        state_dim  = actual_dim,
        action_dim = action_dim,
        min_action = min_action,
        max_action = max_action,
    )
    agent.load_model(path=model_path)
    print(f"[Factory] ✓ Model loaded: {model_path.name}", flush=True)
    print(f"[Factory] {'='*48}", flush=True)

    return MQTTController(
        broker        = broker,
        port          = port,
        topics        = topics,
        agent         = agent,
        state_builder = state_builder,
        qos           = qos,
        deterministic = mqtt_ctrl_cfg.get("deterministic", True),
        username      = username,
        password      = password,
    )


# ======================================================
# Standalone test
# ======================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    ctrl = create_controller(
        network_cfg_path = PROJECT_ROOT / "src/API/config/network.yaml",
        rl_cfg_path      = PROJECT_ROOT / "src/API/config/rl_params.yaml",
        eval_cfg_path    = PROJECT_ROOT / "src/API/config/eval_params.yaml",
    )

    ctrl.start()
    print("Controller running... Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(1)
            s = ctrl.status()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"EP:{s['episode']} STEP:{s['step']} "
                f"LV:{s['last_level']:.2f} SP:{s['last_setpoint']:.2f} "
                f"ACT:{s['last_action']:.2f} "
                f"Jetson:{'✓' if s['jetson_alive'] else '✗'}",
                flush=True
            )
    except KeyboardInterrupt:
        ctrl.stop()
        print("Stopped.", flush=True)