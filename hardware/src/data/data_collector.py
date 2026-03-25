#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/data/data_collector.py
# hardware/src/data/data_collector.py
"""
DataCollector — Jetson Side (Python 3.6.9 Compatible)
"""

from __future__ import print_function
import argparse
import sys
import time
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
# เลื่อนขึ้นไป 3 ชั้นเพื่อให้ถึง my_project
PROJECT_ROOT = current_file.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# นำเข้าโมดูลภายใน (ตรวจสอบว่าไฟล์เหล่านี้ถูกแก้เป็น 3.6 แล้วเช่นกัน)
try:
    from hardware.src.utils.hw_config_loader import get_hw_config
    from hardware.src.utils.logger_hareware import Logger
    from hardware.src.environment.signal_generator_hardware import SignalGenerator
except ImportError:
    print("[Error] Please ensure all hardware source files are in the correct path.")
    raise

# ======================================================
# Safety Layer
# ======================================================
class SafetyLayer(object):
    def __init__(self, min_action, max_action, max_sensor, max_error=None):
        self.min_action = min_action
        self.max_action = max_action
        self.max_sensor = max_sensor
        self.max_error  = max_error

    def clamp_action(self, action):
        return float(np.clip(action, self.min_action, self.max_action))

    def check_sensor(self, value):
        if value < 0 or value > self.max_sensor:
            # f-string -> .format()
            raise RuntimeError(
                "Sensor {:.3f} out of range [0, {}]".format(value, self.max_sensor)
            )

    def check_error(self, action, sensor):
        if self.max_error and abs(action - sensor) > self.max_error:
            # f-string -> .format()
            raise RuntimeError(
                "Tracking error {:.3f} > {}".format(abs(action-sensor), self.max_error)
            )


# ======================================================
# Watchdog
# ======================================================
class Watchdog(object):
    def __init__(self, timeout):
        self.timeout   = timeout
        self.last_kick = time.time()

    def kick(self):
        self.last_kick = time.time()

    def check(self):
        age = time.time() - self.last_kick
        if age > self.timeout:
            # f-string -> .format()
            raise RuntimeError("Watchdog timeout ({:.1f}s)".format(age))


# ======================================================
# DataCollector
# ======================================================
class DataCollector(object):
    MIN_RAW = 0
    MAX_RAW = 27647

    def __init__(
        self,
        signal_override   = None,
        duration_override = None,
        dry_run           = False,
    ):
        self.cfg      = get_hw_config()
        self.dry_run  = dry_run

        # ── Modbus ────────────────────────────────────────────
        mb_cfg = self.cfg["modbus"]
        ctrl   = self.cfg["control"]
        dc_cfg = self.cfg["data_collection"]

        self.min_action       = float(ctrl["min_action"])
        self.max_action       = float(ctrl["max_action"])
        self.delay            = float(ctrl["delay_of_action"])
        self.dt               = float(ctrl["dt"])
        self.address_sensor   = mb_cfg["address_sensor"]
        self.address_actuator = mb_cfg["address_actuator"]
        self.min_raw          = int(mb_cfg.get("min_raw", self.MIN_RAW))
        self.max_raw          = int(mb_cfg.get("max_raw", self.MAX_RAW))

        # ── Data collection config ─────────────────────────────
        self.signal_config  = signal_override or dc_cfg["signal"]
        self.duration_sec   = float(duration_override or dc_cfg["duration_sec"])
        self.log_dt         = float(dc_cfg.get("log_dt", 0.1))
        
        # Ensure path is handled correctly
        save_path_str = dc_cfg.get("save_folder", "hardware/logs/csv")
        self.save_folder    = PROJECT_ROOT / save_path_str
        self.upload_url     = dc_cfg.get("upload_url", "")
        self.device_id      = self.cfg["device"]["id"]

        # Python 3.6.9 supports parents=True
        self.save_folder.mkdir(parents=True, exist_ok=True)

        # ── Modbus driver ──────────────────────────────────────
        if dry_run:
            from hardware.src.utils.mock_modbus import MockModbusTCP
            self._modbus = MockModbusTCP(
                host       = mb_cfg["host"],
                port       = mb_cfg["port"],
                min_raw    = self.min_raw,
                max_raw    = self.max_raw,
                action_max = self.max_action,
                verbose    = False,
            )
            print("[DataCollector] DRY RUN — MockModbusTCP active")
        else:
            from hardware.src.utils.comucation_modbusTCP_hardware import ModbusTCP
            self._modbus = ModbusTCP(
                host=mb_cfg["host"],
                port=mb_cfg["port"]
            )

        # ── Safety + Watchdog ──────────────────────────────────
        self._safety = SafetyLayer(
            min_action = self.min_action,
            max_action = self.max_action,
            max_sensor = self.max_action * 1.1,
            max_error  = self.max_action * 1.5,
        )
        self._watchdog = Watchdog(timeout=2.0)

        # ── Logger ─────────────────────────────────────────────
        self._logger    = Logger()
        self._csv_path  = None
        self._connected = False

    def _action_to_raw(self, action):
        return int(np.interp(
            action,
            [self.min_action, self.max_action],
            [self.min_raw, self.max_raw]
        ))

    def _raw_to_level(self, raw):
        return float(np.interp(
            raw,
            [self.min_raw, self.max_raw],
            [self.min_action, self.max_action]
        ))

    def _write_actuator(self, action):
        raw = self._action_to_raw(action)
        self._modbus.write_holding_register(
            address=self.address_actuator, value=raw
        )

    def _read_sensor(self):
        raw = self._modbus.analog_read(address=self.address_sensor)
        if raw is None:
            raise RuntimeError("Sensor read returned None")
        return self._raw_to_level(raw)

    def _safe_shutdown(self):
        try:
            self._write_actuator(self.min_action)
            print("[DataCollector] Safe shutdown — actuator set to 0")
        except Exception as e:
            print("[DataCollector] Safe shutdown failed: {}".format(e))

    def _upload_csv(self, csv_path):
        if not self.upload_url:
            print("[DataCollector] upload_url not set — skipping upload")
            return False
        try:
            # Python 3.6 open() works with str(Path)
            with open(str(csv_path), "rb") as f:
                resp = requests.post(
                    self.upload_url,
                    files={"file": (csv_path.name, f, "text/csv")},
                    data={
                        "signal_type": self.signal_config.get("type", "unknown"),
                        "duration":    str(self.duration_sec),
                        "device_id":   self.device_id,
                    },
                    timeout=30,
                )
            if resp.status_code == 200:
                result = resp.json()
                print("[DataCollector] Upload OK -> {} ({} KB)".format(
                    result.get('filename'), result.get('size_kb')))
                return True
            else:
                print("[DataCollector] Upload failed: HTTP {}".format(resp.status_code))
                return False
        except Exception as e:
            print("[DataCollector] Upload error: {}".format(e))
            return False

    def run(self):
        ok = self._modbus.connect()
        if not ok:
            raise ConnectionError(
                "Cannot connect to Modbus {}".format(self.cfg['modbus']['host'])
            )
        self._connected = True
        print("[DataCollector] Connected to Modbus")

        sg = SignalGenerator(t_end=self.duration_sec, dt=self.dt)
        _, signal = sg.generate_from_config(self.signal_config)

        log_interval = max(1, int(self.log_dt / self.dt))
        signal_type  = self.signal_config.get("type", "unknown")
        
        # f-string replacement
        params_list = []
        for k, v in self.signal_config.get("params", {}).items():
            params_list.append("{}{}".format(k, v))
        params_str = "_".join(params_list)

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "{}_{}_{}_{}.csv".format(self.device_id, signal_type, params_str, ts)
        self._csv_path = self.save_folder / filename

        print("[DataCollector] Signal: {} | Duration: {}s | Points: {}".format(
            signal_type, self.duration_sec, len(signal)))
        print("[DataCollector] Saving to: {}".format(self._csv_path))

        t0 = time.time()
        rows_logged = 0

        try:
            for idx, val in enumerate(signal):
                self._watchdog.kick()

                action = self.min_action + val * (self.max_action - self.min_action)
                action = self._safety.clamp_action(action)

                self._write_actuator(action)
                time.sleep(self.delay)

                sensor_val = self._read_sensor()
                self._safety.check_sensor(sensor_val)
                self._safety.check_error(action, sensor_val)

                elapsed = time.time() - t0

                if idx % log_interval == 0:
                    self._logger.add_data_log(
                        ["TIME", "DATA_INPUT", "DATA_OUTPUT"],
                        [[elapsed], [action], [sensor_val]]
                    )
                    rows_logged += 1

                if idx % max(1, len(signal) // 20) == 0:
                    pct = (idx / float(len(signal))) * 100
                    print("  [{:5.1f}%] t={:.1f}s action={:.3f} sensor={:.3f}".format(
                        pct, elapsed, action, sensor_val))

                self._watchdog.check()
                time.sleep(max(0.0, self.dt - self.delay))

        except KeyboardInterrupt:
            print("\n[DataCollector] Interrupted by user")
        except Exception as e:
            print("\n[DataCollector] ERROR: {}".format(e))
        finally:
            self._safe_shutdown()
            # Important for 3.6: ensure logger is flushed and saved
            self._logger.save_to_csv(
                file_name=filename,
                path_name=str(self.save_folder),
                folder_name=""
            )

        elapsed_total = time.time() - t0
        print("\n[DataCollector] Done — {} rows logged in {:.1f}s".format(
            rows_logged, elapsed_total))

        if self._csv_path and self._csv_path.exists():
            self._upload_csv(self._csv_path)

        if self._connected:
            self._modbus.disconnect()

        return self._csv_path

def main():
    parser = argparse.ArgumentParser(description="DataCollector — Jetson Side")
    parser.add_argument("--signal", default=None, choices=["pwm", "step", "ramp", "sine", "sinusoid", "triangle"])
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--duty", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    signal_override = None
    if args.signal:
        cfg = get_hw_config()
        base_params = dict(cfg["data_collection"]["signal"].get("params", {}))
        if args.amplitude is not None: base_params["amplitude"] = args.amplitude
        if args.freq is not None:
            base_params["frequency"] = args.freq
            base_params["freq"]      = args.freq
        if args.duty is not None: base_params["duty_cycle"] = args.duty
        signal_override = {"type": args.signal, "params": base_params}

    collector = DataCollector(
        signal_override   = signal_override,
        duration_override = args.duration,
        dry_run           = args.dry_run,
    )
    collector.run()

if __name__ == "__main__":
    main()
