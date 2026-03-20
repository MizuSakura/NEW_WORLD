from hardware.src.utils.logger_hareware import Logger
from hardware.src.utils.comucation_modbusTCP_hardware import ModbusTCP
from hardware.src.environment.signal_generator_hardware import SignalGenerator

from pathlib import Path
import time
import numpy as np

# optional yaml
try:
    import yaml
except ImportError:
    yaml = None


# ==========================================================
# Safety Layer
# ==========================================================
class SafetyLayer:
    """
    Runtime safety envelope for real hardware
    """

    def __init__(
        self,
        min_action,
        max_action,
        min_sensor=None,
        max_sensor=None,
        max_error=None
    ):
        self.min_action = min_action
        self.max_action = max_action
        self.min_sensor = min_sensor
        self.max_sensor = max_sensor
        self.max_error = max_error

    def clamp_action(self, action):
        return float(np.clip(action, self.min_action, self.max_action))

    def check_sensor(self, sensor_value):
        if self.min_sensor is not None and sensor_value < self.min_sensor:
            raise RuntimeError("Sensor value below safe range")
        if self.max_sensor is not None and sensor_value > self.max_sensor:
            raise RuntimeError("Sensor value above safe range")

    def check_error(self, action, sensor_value):
        if self.max_error is None:
            return
        if abs(action - sensor_value) > self.max_error:
            raise RuntimeError("Tracking error exceeded safety limit")


# ==========================================================
# Watchdog
# ==========================================================
class Watchdog:
    """
    Time-based watchdog for loop & communication safety
    """

    def __init__(self, timeout):
        self.timeout = timeout
        self.last_kick = time.time()

    def kick(self):
        self.last_kick = time.time()

    def check(self):
        if time.time() - self.last_kick > self.timeout:
            raise TimeoutError("Watchdog timeout")


# ==========================================================
# Monitoring Layer
# ==========================================================
class MonitoringLayer:
    """
    Non-intrusive system monitoring (observe only)
    """

    def __init__(
        self,
        enabled=True,
        print_interval=1.0,
        track_error=True
    ):
        self.enabled = enabled
        self.print_interval = print_interval
        self.track_error = track_error
        self._last_print = time.time()

    def update(self, t, action, sensor):
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_print < self.print_interval:
            return

        self._last_print = now

        msg = (
            f"[MONITOR] t={t:6.2f}s | "
            f"action={action:6.2f} | "
            f"sensor={sensor:6.2f}"
        )

        if self.track_error:
            msg += f" | error={action - sensor:6.2f}"

        print(msg)


# ==========================================================
# System Response Experiment
# ==========================================================
class response:

    def __init__(
        self,
        signal_generator,
        signal_config,
        log_dt=0.1,
        folder_save_csv=Path("./hardware/logs"),
        file_prefix="real_system",
        host_ip="192.168.1.100",
        port=502,
        min_action=0.0,
        max_action=10.0,
        delay_of_action=0.2,
        address_sensor=1,
        address_actuator=1025,
        monitoring_config=None
    ):

        # -----------------------------
        # Signal & logging
        # -----------------------------
        self.signal_generator = signal_generator
        self.signal_config = signal_config
        self.log_dt = log_dt

        self.folder_save_csv = folder_save_csv
        self.file_prefix = file_prefix

        # -----------------------------
        # Hardware config
        # -----------------------------
        self.min_action = min_action
        self.max_action = max_action
        self.delay = delay_of_action

        self.address_sensor = address_sensor
        self.address_actuator = address_actuator

        self.max_value_remote_IO = 27647
        self.min_value_remote_IO = 0

        folder_save_csv.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Communication & logger
        # -----------------------------
        self.modbus = ModbusTCP(host=host_ip, port=port)
        self.modbus.connect()

        self.logger = Logger()

        # -----------------------------
        # Safety & watchdog
        # -----------------------------
        self.safety = SafetyLayer(
            min_action=self.min_action,
            max_action=self.max_action,
            min_sensor=0.0,
            max_sensor=self.max_action * 1.05,
            max_error=11.0
        )

        self.watchdog = Watchdog(timeout=1.0)
        self.safe_action = self.min_action

        # -----------------------------
        # Monitoring
        # -----------------------------
        monitoring_config = monitoring_config or {}

        self.monitoring = MonitoringLayer(
            enabled=monitoring_config.get("enabled", True),
            print_interval=monitoring_config.get("print_interval", 1.0),
            track_error=monitoring_config.get("track_error", True)
        )

    # --------------------------------------------------
    # Signal generation
    # --------------------------------------------------
    def _generate_signal(self):
        _, signal = self.signal_generator.generate_from_config(
            self.signal_config
        )
        return signal

    # --------------------------------------------------
    # Hardware IO
    # --------------------------------------------------
    def read_sensor(self, address):
        raw = self.modbus.analog_read(address=address)
        return float(np.interp(
            raw,
            [self.min_value_remote_IO, self.max_value_remote_IO],
            [self.min_action, self.max_action]
        ))

    def write_actuator(self, address, action):
        raw = int(np.interp(
            action,
            [self.min_action, self.max_action],
            [self.min_value_remote_IO, self.max_value_remote_IO]
        ))
        self.modbus.write_holding_register(address=address, value=raw)

    # --------------------------------------------------
    # Emergency Stop
    # --------------------------------------------------
    def emergency_stop(self, reason):
        print(f"\n[EMERGENCY STOP] {reason}")
        try:
            self.write_actuator(self.address_actuator, self.safe_action)
        finally:
            self.logger.flush()
            raise SystemExit(reason)

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    def run(self):

        signal = self._generate_signal()
        dt = self.signal_generator.dt
        log_interval = max(1, int(self.log_dt / dt))

        t0 = time.time()

        for idx, val in enumerate(signal):
            try:
                self.watchdog.kick()

                action = self.min_action + val * (self.max_action - self.min_action)
                action = self.safety.clamp_action(action)

                self.write_actuator(self.address_actuator, action)
                time.sleep(self.delay)

                sensor_value = self.read_sensor(self.address_sensor)

                self.safety.check_sensor(sensor_value)
                self.safety.check_error(action, sensor_value)

                t = time.time() - t0
                self.monitoring.update(t, action, sensor_value)

                if idx % log_interval == 0:
                    self.logger.add_data_log(
                        ["TIME", "DATA_INPUT", "DATA_OUTPUT"],
                        [[t], [action], [sensor_value]]
                    )

                self.watchdog.check()
                time.sleep(max(0.0, dt - self.delay))

            except Exception as e:
                self.emergency_stop(str(e))

        # ---------- finalize ----------
        signal_type = self.signal_config.get("type", "unknown")
        params = self.signal_config.get("params", {})
        meta = "_".join(f"{k}_{v}" for k, v in params.items())

        file_name = f"{self.file_prefix}_{signal_type}_{meta}.csv"
        self.logger.save_to_csv(file_name, folder_name=self.folder_save_csv)


# ==========================================================
# CONFIG LOADER
# ==========================================================
DEFAULT_CONFIG = {
    "signal": {
        "type": "pwm",
        "params": {
            "amplitude": 1.0,
            "frequency": 0.5,
            "duty_cycle": 0.4
        }
    },
    "monitoring": {
        "enabled": True,
        "print_interval": 1.0,
        "track_error": True
    }
}


def load_config(yaml_path="experiment.yaml"):
    if yaml is None or not Path(yaml_path).exists():
        print("[INFO] No YAML found → using DEFAULT_CONFIG")
        return DEFAULT_CONFIG

    with open(yaml_path, "r") as f:
        print(f"[INFO] Loading config from {yaml_path}")
        return yaml.safe_load(f)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    import yaml
    from pathlib import Path

    _cfg     = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "config/hardware.yaml").read_text()
    )
    IP_HOST          = _cfg["modbus"]["host"]
    PORT             = _cfg["modbus"]["port"]
    ADDRESS_SENSOR   = _cfg["modbus"]["address_sensor"]
    ADDRESS_ACTUATOR = _cfg["modbus"]["address_actuator"]
    MIN_ACTION       = _cfg["control"]["min_action"]
    MAX_ACTION       = _cfg["control"]["max_action"]
    DELAY_OF_ACTION  = _cfg["control"]["delay_of_action"]
    DT               = _cfg["control"]["dt"]
    TIME_SIM         = _cfg["data_collection"]["duration_sec"]

    signal_config    = _cfg["data_collection"]["signal"]
    monitoring_config = _cfg["data_collection"]["monitoring"]
    
    
    sg = SignalGenerator(t_end=TIME_SIM, dt=DT)

    system = response(
        signal_generator=sg,
        signal_config=signal_config,
        log_dt=0.1,
        host_ip=IP_HOST,
        port=PORT,
        min_action=MIN_ACTION,
        max_action=MAX_ACTION,
        delay_of_action=DELAY_OF_ACTION,
        address_sensor=ADDRESS_SENSOR,
        address_actuator=ADDRESS_ACTUATOR,
        monitoring_config=monitoring_config
    )

    system.run()
