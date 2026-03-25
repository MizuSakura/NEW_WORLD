#/home/rl_controller/Desktop/RL_PROJECCT/NEW_WORLD/my_project/hardware/src/utils/comucation_modbusTCP_hardware.py
# hardware/src/utils/comucation_modbusTCP_hardware.py
"""
แก้ไขให้ compatible กับ pymodbus 2.5.3 (Jetson Nano Python 3.6.9)

pymodbus 2.x API ต่างจาก 3.x:
    Import : from pymodbus.client.sync import ModbusTcpClient
    Connect: client.connect()  → return True/False
    Slave  : unit=  (ไม่ใช่ device_id=)
    Error  : result.isError()  → ยังมีเหมือนเดิม
"""

from __future__ import print_function

from pymodbus.client.sync import ModbusTcpClient   # pymodbus 2.x
import time


# =====================================================================
# MODBUS TCP CLASS
# =====================================================================
class ModbusTCP(object):
    """
    A helper class for communicating with Modbus TCP devices.
    Supports reading/writing coils, inputs, and registers.
    Supports automatic decimal/hex/bin/octal address parsing.

    Compatible: pymodbus 2.5.3
    """

    def __init__(self, host='192.168.1.100', port=502):
        self.host   = host
        self.port   = port
        self.client = ModbusTcpClient(host=host, port=port)

    # ---------------------------------------------------------
    # UTIL FUNCTIONS
    # ---------------------------------------------------------
    def _parse_address(self, addr):
        """
        Parse address from any numeric base to decimal.
        Supports decimal, hex, octal, binary, or pure hex chars.
        """
        if isinstance(addr, int):
            return addr

        if not isinstance(addr, str):
            raise ValueError("Unsupported address type: {}".format(type(addr)))

        s = addr.strip().lower()

        if s.startswith("0x"):
            return int(s, 16)
        if s.startswith("0b"):
            return int(s, 2)
        if s.startswith("0o"):
            return int(s, 8)

        if s.isdigit():
            return int(s, 10)

        hex_chars = set("0123456789abcdef")
        if all(c in hex_chars for c in s):
            return int(s, 16)

        raise ValueError("Cannot auto-detect numeric base from: {}".format(addr))

    # ---------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------
    def connect(self):
        result = self.client.connect()
        return result   # True if connected

    def disconnect(self):
        self.client.close()
        return True

    # ---------------------------------------------------------
    # DIGITAL OUTPUTS (COILS)
    # ---------------------------------------------------------
    def read_status_output(self, address, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_coils(address, count=1, unit=device_id)
        if resp is None or resp.isError():
            print("Error reading coil at {}".format(address))
            return None
        return resp.bits[0]

    def digital_write(self, address, value, device_id=1):
        address = self._parse_address(address)
        resp = self.client.write_coil(address, value, unit=device_id)
        if resp is None or resp.isError():
            print("Error writing coil at {}".format(address))
            return None
        return True

    # ---------------------------------------------------------
    # DIGITAL INPUTS
    # ---------------------------------------------------------
    def digital_input(self, address, count=1, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_discrete_inputs(address, count=count, unit=device_id)
        if resp is None or resp.isError():
            print("Error reading input at {}".format(address))
            return None
        return resp.bits[0] if count == 1 else resp.bits

    # ---------------------------------------------------------
    # INPUT REGISTERS (ANALOG)
    # ---------------------------------------------------------
    def analog_read(self, address, count=1, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_input_registers(address, count=count, unit=device_id)
        if resp is None or resp.isError():
            print("Error reading analog input at {}".format(address))
            return None
        return resp.registers[0] if count == 1 else resp.registers

    # ---------------------------------------------------------
    # HOLDING REGISTERS
    # ---------------------------------------------------------
    def read_holding_registers(self, address, count=1, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_holding_registers(address, count=count, unit=device_id)
        if resp is None or resp.isError():
            print("Error reading HR at {}".format(address))
            return None
        return resp.registers

    def write_holding_register(self, address, value, device_id=1):
        address = self._parse_address(address)
        resp = self.client.write_register(address, value, unit=device_id)
        if resp is None or resp.isError():
            print("Error writing HR at {}".format(address))
            return None
        return True


# =====================================================================
# TEST FUNCTIONS
# =====================================================================
def test_coil_output(modbus, start_hex, end_hex, delay=0.2):
    print("\n=== TEST: Coil Output ===")
    s = int(start_hex, 16)
    e = int(end_hex, 16)

    for addr in range(s, e + 1):
        modbus.digital_write(addr, 1)
        v = modbus.read_status_output(addr)
        print("COIL ON  : {} ({}) = {}".format(addr, hex(addr), v))
        time.sleep(delay)

    for addr in range(s, e + 1):
        modbus.digital_write(addr, 0)
        v = modbus.read_status_output(addr)
        print("COIL OFF : {} ({}) = {}".format(addr, hex(addr), v))
        time.sleep(delay)


def test_input_status(modbus, start_hex, end_hex, delay=0.2):
    print("\n=== TEST: Input Status ===")
    s = int(start_hex, 16)
    e = int(end_hex, 16)

    for addr in range(s, e + 1):
        v = modbus.digital_input(addr)
        print("INPUT : {} ({}) = {}".format(addr, hex(addr), v))
        time.sleep(delay)


def test_write_register(modbus, address, value=0):
    print("\n=== TEST: Write Holding Register ===")
    dec_addr = int(address)
    ok = modbus.write_holding_register(dec_addr, value)
    print("WRITE HR[{}] = {}, STATUS={}".format(dec_addr, value, ok))


# =====================================================================
# TEST MANAGER (OOP)
# =====================================================================
class TestManager(object):
    """
    OOP-based manager for Modbus tests.
    Safely executes registered test functions.
    Includes emergency shutdown to turn OFF all coils & registers.
    """

    def __init__(self, modbus):
        self.modbus = modbus
        self.tests  = {}

    def register(self, name, func, args=(), kwargs=None):
        self.tests[name] = {
            "func":   func,
            "args":   args,
            "kwargs": kwargs or {}
        }

    def run(self, name):
        if name not in self.tests:
            raise ValueError("Unknown test name: {}".format(name))

        f = self.tests[name]
        print("\n=== RUN TEST: {} ===".format(name))

        try:
            return f["func"](self.modbus, *f["args"], **f["kwargs"])
        except Exception as e:
            print("[ERROR] Test '{}' failed: {}".format(name, e))
            self.emergency_shutdown()
            return None

    def emergency_shutdown(self, coil_range=("0x4000", "0x40FF"), reg_range=(0, 200)):
        print("\n*** EMERGENCY SHUTDOWN ACTIVATED ***")

        start = int(coil_range[0], 16)
        end   = int(coil_range[1], 16)
        for addr in range(start, end + 1):
            try:
                self.modbus.digital_write(addr, 0)
            except Exception:
                pass

        r_start, r_end = reg_range
        for addr in range(r_start, r_end + 1):
            try:
                self.modbus.write_holding_register(addr, 0)
            except Exception:
                pass

        print("All coils and holding registers reset to 0.")


# =====================================================================
# MAIN PROGRAM
# =====================================================================
if __name__ == "__main__":

    modbus  = ModbusTCP(host="192.168.1.100")
    manager = TestManager(modbus)

    manager.register("coil_output",   test_coil_output,   args=("0x4000", "0x400F"))
    manager.register("input_status",  test_input_status,  args=("0x0000", "0x000F"))
    manager.register("write_register",test_write_register, args=("1025",), kwargs={"value": 0})

    try:
        if not modbus.connect():
            raise ConnectionError("Cannot connect to Modbus device.")

        print("\n===== AVAILABLE TESTS =====")
        for i, name in enumerate(manager.tests.keys(), 1):
            print("{}) {}".format(i, name))

        choice    = int(input("Select test number: "))
        test_name = list(manager.tests.keys())[choice - 1]
        manager.run(test_name)

    except Exception as e:
        print("Error:", e)
        manager.emergency_shutdown()

    finally:
        modbus.disconnect()
        print("Modbus connection closed.")

