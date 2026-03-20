# hardware/src/data/data_collector.py
"""
DataCollector — Jetson Side
----------------------------
เก็บข้อมูล system response จาก hardware จริง แล้ว upload ไป Laptop server
ใช้สำหรับ train LSTM model

Flow:
    1. อ่าน config จาก hardware/config/hardware.yaml
    2. สร้าง signal ตาม signal config (pwm/step/ramp/sine/triangle)
    3. loop:
        - เขียน action → Modbus actuator
        - รอ delay
        - อ่าน sensor ← Modbus
        - log TIME, DATA_INPUT, DATA_OUTPUT
    4. save CSV → hardware/logs/csv/
    5. POST CSV → server /rc/upload-lstm-data

วิธีรัน:
    python -m hardware.src.data.data_collector
    python -m hardware.src.data.data_collector --signal pwm
    python -m hardware.src.data.data_collector --signal step --duration 120
    python -m hardware.src.data.data_collector --dry-run   # ไม่ต่อ hardware
"""

import argparse
import sys
import time
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.src.utils.hw_config_loader import get_hw_config
from hardware.src.utils.logger_hareware import Logger
from hardware.src.environment.signal_generator_hardware import SignalGenerator


# ======================================================
# Safety Layer
# ======================================================
class SafetyLayer:
    def __init__(self, min_action, max_action, max_sensor, max_error=None):
        self.min_action = min_action
        self.max_action = max_action
        self.max_sensor = max_sensor
        self.max_error  = max_error

    def clamp_action(self, action: float) -> float:
        return float(np.clip(action, self.min_action, self.max_action))

    def check_sensor(self, value: float):
        if value < 0 or value > self.max_sensor:
            raise RuntimeError(
                f"Sensor {value:.3f} out of range [0, {self.max_sensor}]"
            )

    def check_error(self, action: float, sensor: float):
        if self.max_error and abs(action - sensor) > self.max_error:
            raise RuntimeError(
                f"Tracking error {abs(action-sensor):.3f} > {self.max_error}"
            )


# ======================================================
# Watchdog
# ======================================================
class Watchdog:
    def __init__(self, timeout: float):
        self.timeout   = timeout
        self.last_kick = time.time()

    def kick(self):
        self.last_kick = time.time()

    def check(self):
        age = time.time() - self.last_kick
        if age > self.timeout:
            raise TimeoutError(f"Watchdog timeout ({age:.1f}s)")


# ======================================================
# DataCollector
# ======================================================
class DataCollector:
    """
    เก็บข้อมูล TIME, DATA_INPUT, DATA_OUTPUT จาก hardware
    แล้ว upload CSV ไป server

    Parameters
    ----------
    signal_override : dict | None
        ถ้าไม่ None ใช้ config นี้แทน hardware.yaml data_collection.signal
    duration_override : float | None
        ถ้าไม่ None ใช้ค่านี้แทน hardware.yaml data_collection.duration_sec
    dry_run : bool
        True = จำลอง hardware ด้วย MockModbusTCP (ไม่ต้องต่อ hardware จริง)
    """

    # raw value ของ Remote IO (Wago/Phoenix)
    MIN_RAW = 0
    MAX_RAW = 27647

    def __init__(
        self,
        signal_override:   dict  = None,
        duration_override: float = None,
        dry_run:           bool  = False,
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
        self.save_folder    = PROJECT_ROOT / dc_cfg.get("save_folder", "hardware/logs/csv")
        self.upload_url     = dc_cfg.get("upload_url", "")
        self.device_id      = self.cfg["device"]["id"]

        self.save_folder.mkdir(parents=True, exist_ok=True)

        # ── Modbus driver ──────────────────────────────────────
        if dry_run:
            from src.utils.mock_modbus import MockModbusTCP
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
        self._logger    = Logger(flush_every=500)
        self._csv_path  = None
        self._connected = False

    # ──────────────────────────────────────────────────────────
    # IO helpers
    # ──────────────────────────────────────────────────────────
    def _action_to_raw(self, action: float) -> int:
        return int(np.interp(
            action,
            [self.min_action, self.max_action],
            [self.min_raw, self.max_raw]
        ))

    def _raw_to_level(self, raw: int) -> float:
        return float(np.interp(
            raw,
            [self.min_raw, self.max_raw],
            [self.min_action, self.max_action]
        ))

    def _write_actuator(self, action: float):
        raw = self._action_to_raw(action)
        self._modbus.write_holding_register(
            address=self.address_actuator, value=raw
        )

    def _read_sensor(self) -> float:
        raw = self._modbus.analog_read(address=self.address_sensor)
        if raw is None:
            raise RuntimeError("Sensor read returned None")
        return self._raw_to_level(raw)

    def _safe_shutdown(self):
        """ส่ง 0 ไป actuator เมื่อหยุดหรือ error"""
        try:
            self._write_actuator(self.min_action)
            print("[DataCollector] Safe shutdown — actuator set to 0")
        except Exception as e:
            print(f"[DataCollector] Safe shutdown failed: {e}")

    # ──────────────────────────────────────────────────────────
    # Upload
    # ──────────────────────────────────────────────────────────
    def _upload_csv(self, csv_path: Path) -> bool:
        """POST CSV ไป server /rc/upload-lstm-data"""
        if not self.upload_url:
            print("[DataCollector] upload_url ไม่ได้ตั้งค่า — ข้าม upload")
            return False
        try:
            with open(csv_path, "rb") as f:
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
                print(f"[DataCollector] Upload OK → {result.get('filename')} "
                      f"({result.get('size_kb')} KB)")
                return True
            else:
                print(f"[DataCollector] Upload failed: HTTP {resp.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("[DataCollector] Upload failed: Cannot connect to server")
            return False
        except Exception as e:
            print(f"[DataCollector] Upload error: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────
    def run(self) -> Path:
        """
        รัน data collection แบบ blocking
        Returns
        -------
        Path — path ของ CSV ที่ save ไว้
        """
        # ── connect ───────────────────────────────────────────
        ok = self._modbus.connect()
        if not ok:
            raise ConnectionError(
                f"Cannot connect to Modbus {self.cfg['modbus']['host']}"
            )
        self._connected = True
        print(f"[DataCollector] Connected to Modbus")

        # ── generate signal ───────────────────────────────────
        sg = SignalGenerator(t_end=self.duration_sec, dt=self.dt)
        _, signal = sg.generate_from_config(self.signal_config)

        log_interval = max(1, int(self.log_dt / self.dt))
        signal_type  = self.signal_config.get("type", "unknown")
        params_str   = "_".join(
            f"{k}{v}" for k, v in
            self.signal_config.get("params", {}).items()
        )
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.device_id}_{signal_type}_{params_str}_{ts}.csv"
        self._csv_path = self.save_folder / filename
        self._logger.save_to_csv(
            file_name   = filename,
            path_name   = str(self.save_folder),
            folder_name = ""
        )

        print(f"[DataCollector] Signal: {signal_type} | "
              f"Duration: {self.duration_sec}s | "
              f"Points: {len(signal)}")
        print(f"[DataCollector] Saving to: {self._csv_path}")

        t0 = time.time()
        rows_logged = 0

        try:
            for idx, val in enumerate(signal):
                self._watchdog.kick()

                # ── compute action ─────────────────────────────
                action = self.min_action + val * (self.max_action - self.min_action)
                action = self._safety.clamp_action(action)

                # ── write actuator ─────────────────────────────
                self._write_actuator(action)
                time.sleep(self.delay)

                # ── read sensor ────────────────────────────────
                sensor_val = self._read_sensor()
                self._safety.check_sensor(sensor_val)
                self._safety.check_error(action, sensor_val)

                elapsed = time.time() - t0

                # ── log ────────────────────────────────────────
                if idx % log_interval == 0:
                    self._logger.add_data_log(
                        ["TIME", "DATA_INPUT", "DATA_OUTPUT"],
                        [[elapsed], [action], [sensor_val]]
                    )
                    rows_logged += 1

                # ── progress ───────────────────────────────────
                if idx % max(1, len(signal) // 20) == 0:
                    pct = idx / len(signal) * 100
                    print(f"  [{pct:5.1f}%] t={elapsed:.1f}s "
                          f"action={action:.3f} sensor={sensor_val:.3f}")

                self._watchdog.check()
                time.sleep(max(0.0, self.dt - self.delay))

        except KeyboardInterrupt:
            print("\n[DataCollector] Interrupted by user")
        except Exception as e:
            print(f"\n[DataCollector] ERROR: {e}")
        finally:
            self._safe_shutdown()
            self._logger.flush()

        # ── summary ───────────────────────────────────────────
        elapsed_total = time.time() - t0
        print(f"\n[DataCollector] Done — {rows_logged} rows logged "
              f"in {elapsed_total:.1f}s")
        print(f"[DataCollector] CSV: {self._csv_path}")

        # ── upload ─────────────────────────────────────────────
        if self._csv_path and self._csv_path.exists():
            self._upload_csv(self._csv_path)

        if self._connected:
            self._modbus.disconnect()

        return self._csv_path


# ======================================================
# CLI
# ======================================================
def main():
    parser = argparse.ArgumentParser(
        description="DataCollector — เก็บข้อมูล hardware แล้ว upload server"
    )
    parser.add_argument(
        "--signal", default=None,
        choices=["pwm", "step", "ramp", "sine", "sinusoid", "triangle"],
        help="override signal type จาก hardware.yaml"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="override duration (วินาที)"
    )
    parser.add_argument(
        "--amplitude", type=float, default=None,
        help="override signal amplitude"
    )
    parser.add_argument(
        "--freq", type=float, default=None,
        help="override signal frequency"
    )
    parser.add_argument(
        "--duty", type=float, default=None,
        help="override duty cycle (สำหรับ pwm)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="จำลอง hardware ด้วย MockModbusTCP"
    )
    args = parser.parse_args()

    # ── build signal override ──────────────────────────────────
    signal_override = None
    if args.signal:
        cfg = get_hw_config()
        base_params = dict(
            cfg["data_collection"]["signal"].get("params", {})
        )
        if args.amplitude is not None:
            base_params["amplitude"] = args.amplitude
        if args.freq is not None:
            base_params["frequency"] = args.freq
            base_params["freq"]      = args.freq
        if args.duty is not None:
            base_params["duty_cycle"] = args.duty

        signal_override = {"type": args.signal, "params": base_params}

    # ── run ────────────────────────────────────────────────────
    collector = DataCollector(
        signal_override   = signal_override,
        duration_override = args.duration,
        dry_run           = args.dry_run,
    )
    collector.run()


if __name__ == "__main__":
    main()