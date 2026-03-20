# tools/test_modbus_connection.py
"""
Modbus Connection Tester
-------------------------
ทดสอบ Modbus TCP จริงๆ โดยไม่เอา SAC มาเกี่ยว
รันได้ 2 แบบ:
    1. Hardware จริง  → python -m tools.test_modbus_connection
    2. Mock           → python -m tools.test_modbus_connection --mock

Tests ที่ทำ:
    1. Connection test
    2. Sensor read (Input Register)
    3. Actuator write + read back (Holding Register)
    4. Continuous read loop (30 ครั้ง) — ดู latency และ noise
    5. Safe shutdown (ส่ง 0 ไป actuator)
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# ── Path setup ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.comucation_modbusTCP import ModbusTCP


# ======================================================
# Config — แก้ตามระบบจริง
# ======================================================
DEFAULT_CONFIG = {
    "host":             "192.168.1.100",
    "port":             502,
    "address_sensor":   1,       # Input Register สำหรับอ่าน level
    "address_actuator": 1025,    # Holding Register สำหรับเขียน action
    "min_raw":          0,
    "max_raw":          27647,
    "min_action":       0.0,
    "max_action":       10.0,
    "level_max":        10.0,
    "read_loop_count":  30,
    "read_loop_delay":  0.1,     # วินาที
}


# ======================================================
# Helper — scale functions
# ======================================================
def raw_to_level(raw, cfg) -> float:
    return float(np.interp(raw, [cfg["min_raw"], cfg["max_raw"]],
                                [0.0, cfg["level_max"]]))

def action_to_raw(action, cfg) -> int:
    return int(np.interp(action, [cfg["min_action"], cfg["max_action"]],
                                  [cfg["min_raw"], cfg["max_raw"]]))


# ======================================================
# Individual Tests
# ======================================================

def test_connection(modbus) -> bool:
    print("\n" + "="*50)
    print("TEST 1: Connection")
    print("="*50)
    ok = modbus.connect()
    if ok:
        print(f"  ✅ Connected successfully")
    else:
        print(f"  ❌ Connection FAILED")
    return ok


def test_sensor_read(modbus, cfg) -> bool:
    print("\n" + "="*50)
    print("TEST 2: Sensor Read (Input Register)")
    print("="*50)
    try:
        raw = modbus.analog_read(address=cfg["address_sensor"])
        if raw is None:
            print(f"  ❌ Read returned None — check address or connection")
            return False
        level = raw_to_level(raw, cfg)
        print(f"  Address : {cfg['address_sensor']}")
        print(f"  Raw     : {raw}")
        print(f"  Level   : {level:.4f} (scaled)")
        print(f"  ✅ Sensor read OK")
        return True
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False


def test_actuator_write(modbus, cfg, test_values=None) -> bool:
    print("\n" + "="*50)
    print("TEST 3: Actuator Write + Read-back (Holding Register)")
    print("="*50)

    if test_values is None:
        test_values = [0.0, 2.5, 5.0, 7.5, 10.0, 0.0]  # always end with 0

    all_ok = True
    for action in test_values:
        raw = action_to_raw(action, cfg)
        ok  = modbus.write_holding_register(address=cfg["address_actuator"], value=raw)

        # อ่านกลับมาตรวจ
        regs = modbus.read_holding_registers(address=cfg["address_actuator"], count=1)
        readback_raw   = regs[0] if regs else None
        readback_action = raw_to_level(readback_raw, cfg) if readback_raw is not None else None

        status = "✅" if ok else "❌"
        print(f"  {status} action={action:5.1f} → raw={raw:5d}"
              f" | readback raw={readback_raw} ({readback_action:.4f})" if readback_action is not None
              else f"  {status} action={action:5.1f} → raw={raw:5d} | readback=N/A")

        if not ok:
            all_ok = False
        time.sleep(0.05)

    return all_ok


def test_read_loop(modbus, cfg) -> dict:
    print("\n" + "="*50)
    print(f"TEST 4: Continuous Read Loop ({cfg['read_loop_count']} samples)")
    print("="*50)

    readings = []
    latencies = []

    for i in range(cfg["read_loop_count"]):
        t0  = time.perf_counter()
        raw = modbus.analog_read(address=cfg["address_sensor"])
        dt  = (time.perf_counter() - t0) * 1000  # ms

        if raw is None:
            print(f"  [{i+1:3d}] ❌ Read failed")
            continue

        level = raw_to_level(raw, cfg)
        readings.append(level)
        latencies.append(dt)

        print(f"  [{i+1:3d}] raw={raw:6d}  level={level:6.3f}  latency={dt:6.2f} ms")
        time.sleep(cfg["read_loop_delay"])

    if not readings:
        print("  ❌ No successful reads")
        return {}

    stats = {
        "count":       len(readings),
        "mean":        float(np.mean(readings)),
        "std":         float(np.std(readings)),
        "min":         float(np.min(readings)),
        "max":         float(np.max(readings)),
        "lat_mean_ms": float(np.mean(latencies)),
        "lat_max_ms":  float(np.max(latencies)),
    }

    print(f"\n  --- Stats ---")
    print(f"  Samples   : {stats['count']}")
    print(f"  Level     : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
          f"  min={stats['min']:.4f}  max={stats['max']:.4f}")
    print(f"  Latency   : mean={stats['lat_mean_ms']:.2f} ms"
          f"  max={stats['lat_max_ms']:.2f} ms")
    print(f"  ✅ Read loop OK")
    return stats


def test_safe_shutdown(modbus, cfg) -> bool:
    print("\n" + "="*50)
    print("TEST 5: Safe Shutdown (ส่ง 0 ไป actuator)")
    print("="*50)
    ok = modbus.write_holding_register(
        address=cfg["address_actuator"], value=cfg["min_raw"]
    )
    if ok:
        print(f"  ✅ Actuator set to 0 (raw={cfg['min_raw']})")
    else:
        print(f"  ❌ Shutdown write FAILED")
    return ok


# ======================================================
# Summary
# ======================================================
def print_summary(results: dict):
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    icons = {True: "✅ PASS", False: "❌ FAIL", None: "⚠️  SKIP"}
    for name, result in results.items():
        print(f"  {icons.get(result, '?')}  {name}")
    total  = sum(1 for v in results.values() if v is not None)
    passed = sum(1 for v in results.values() if v is True)
    print(f"\n  {passed}/{total} tests passed")


# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Modbus Connection Tester")
    parser.add_argument("--mock",   action="store_true", help="ใช้ MockModbusTCP แทน hardware")
    parser.add_argument("--host",   default=DEFAULT_CONFIG["host"])
    parser.add_argument("--port",   default=DEFAULT_CONFIG["port"],  type=int)
    parser.add_argument("--sensor", default=DEFAULT_CONFIG["address_sensor"],   type=int)
    parser.add_argument("--act",    default=DEFAULT_CONFIG["address_actuator"], type=int)
    parser.add_argument("--loops",  default=DEFAULT_CONFIG["read_loop_count"],  type=int)
    parser.add_argument("--verbose",action="store_true", help="verbose mock output")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg["host"]             = args.host
    cfg["port"]             = args.port
    cfg["address_sensor"]   = args.sensor
    cfg["address_actuator"] = args.act
    cfg["read_loop_count"]  = args.loops

    # ── เลือก Modbus driver ──────────────────────────────
    if args.mock:
        from src.utils.mock_modbus import MockModbusTCP
        modbus = MockModbusTCP(
            host       = args.host,
            port       = args.port,
            verbose    = args.verbose,
        )
        print("\n🔧  Running with MockModbusTCP (no hardware needed)")
    else:
        modbus = ModbusTCP(host=args.host, port=args.port)
        print(f"\n🔌  Running with real Modbus TCP → {args.host}:{args.port}")

    results = {}

    # ── TEST 1: Connection ────────────────────────────────
    connected = test_connection(modbus)
    results["1. Connection"] = connected

    if not connected:
        print("\n⛔ Cannot continue — connection failed")
        print_summary(results)
        sys.exit(1)

    # ── TEST 2: Sensor Read ───────────────────────────────
    results["2. Sensor Read"] = test_sensor_read(modbus, cfg)

    # ── TEST 3: Actuator Write ────────────────────────────
    results["3. Actuator Write"] = test_actuator_write(modbus, cfg)

    # ── TEST 4: Read Loop ─────────────────────────────────
    stats = test_read_loop(modbus, cfg)
    results["4. Read Loop"] = bool(stats)

    # ── TEST 5: Safe Shutdown ─────────────────────────────
    results["5. Safe Shutdown"] = test_safe_shutdown(modbus, cfg)

    # ── Disconnect ────────────────────────────────────────
    modbus.disconnect()

    # ── Summary ───────────────────────────────────────────
    print_summary(results)

    all_passed = all(v is True for v in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()