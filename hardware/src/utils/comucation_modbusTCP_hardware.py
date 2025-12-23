from pymodbus.client import ModbusTcpClient
import time

# =====================================================================
# MODBUS TCP CLASS
# =====================================================================
class ModbusTCP:
    """
    A helper class for communicating with Modbus TCP devices.
    Supports reading/writing coils, inputs, and registers.
    Supports automatic decimal/hex/bin/octal address parsing.
    """

    def __init__(self, host='192.168.1.100', port=502):
        self.client = ModbusTcpClient(host=host, port=port)
        self.host = host
        self.port = port

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
            raise ValueError(f"Unsupported address type: {type(addr)}")

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

        raise ValueError(f"Cannot auto-detect numeric base from: {addr}")

    # ---------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------
    def connect(self):
        self.client.connect()
        return self.client.connected

    def disconnect(self):
        self.client.close()
        return not self.client.connected

    # ---------------------------------------------------------
    # DIGITAL OUTPUTS (COILS)
    # ---------------------------------------------------------
    def read_status_output(self, address, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_coils(address, count=1, device_id=device_id)
        if resp.isError():
            print(f"Error reading coil at {address}")
            return None
        return resp.bits[0]

    def digital_write(self, address, value, device_id=1):
        address = self._parse_address(address)
        resp = self.client.write_coil(address, value, device_id=device_id)
        if resp.isError():
            print(f"Error writing coil at {address}")
            return None
        return True

    # ---------------------------------------------------------
    # DIGITAL INPUTS
    # ---------------------------------------------------------
    def digital_input(self, address, count=1, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_discrete_inputs(address, count=count, device_id=device_id)
        if resp.isError():
            print(f"Error reading input at {address}")
            return None
        return resp.bits[0] if count == 1 else resp.bits

    # ---------------------------------------------------------
    # INPUT REGISTERS (ANALOG)
    # ---------------------------------------------------------
    def analog_read(self, address, count=1, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_input_registers(address, count=count, device_id=device_id)
        if resp.isError():
            print(f"Error reading analog input at {address}")
            return None
        return resp.registers[0] if count == 1 else resp.registers

    # ---------------------------------------------------------
    # HOLDING REGISTERS
    # ---------------------------------------------------------
    def read_holding_registers(self, address, count=1, device_id=1):
        address = self._parse_address(address)
        resp = self.client.read_holding_registers(address, count=count, device_id=device_id)
        if resp.isError():
            print(f"Error reading HR at {address}")
            return None
        return resp.registers

    def write_holding_register(self, address, value, device_id=1):
        address = self._parse_address(address)
        resp = self.client.write_register(address, value, device_id=device_id)
        if resp.isError():
            print(f"Error writing HR at {address}")
            return None
        return True


# =====================================================================
# TEST FUNCTIONS
# =====================================================================
def test_coil_output(modbus, start_hex, end_hex, delay=0.2):
    print("\n=== TEST: Coil Output ===")
    s = int(start_hex, 16)
    e = int(end_hex, 16)

    # Turn ON coils
    for addr in range(s, e + 1):
        modbus.digital_write(addr, 1)
        v = modbus.read_status_output(addr)
        print(f"COIL ON  : {addr} ({hex(addr)}) = {v}")
        time.sleep(delay)

    # Turn OFF coils
    for addr in range(s, e + 1):
        modbus.digital_write(addr, 0)
        v = modbus.read_status_output(addr)
        print(f"COIL OFF : {addr} ({hex(addr)}) = {v}")
        time.sleep(delay)


def test_input_status(modbus, start_hex, end_hex, delay=0.2):
    print("\n=== TEST: Input Status ===")
    s = int(start_hex, 16)
    e = int(end_hex, 16)

    for addr in range(s, e + 1):
        v = modbus.digital_input(addr)
        print(f"INPUT : {addr} ({hex(addr)}) = {v}")
        time.sleep(delay)


def test_write_register(modbus, address, value=0):
    print("\n=== TEST: Write Holding Register ===")
    dec_addr = int(address)
    ok = modbus.write_holding_register(dec_addr, value)
    print(f"WRITE HR[{dec_addr}] = {value}, STATUS={ok}")


# =====================================================================
# TEST MANAGER (OOP)
# =====================================================================
class TestManager:
    """
    OOP-based manager for Modbus tests.
    Safely executes registered test functions.
    Includes emergency shutdown to turn OFF all coils & registers.
    """

    def __init__(self, modbus):
        self.modbus = modbus
        self.tests = {}

    # --------------------------
    # Register test
    # --------------------------
    def register(self, name, func, args=(), kwargs=None):
        self.tests[name] = {
            "func": func,
            "args": args,
            "kwargs": kwargs or {}
        }

    # --------------------------
    # Run test
    # --------------------------
    def run(self, name):
        if name not in self.tests:
            raise ValueError(f"Unknown test name: {name}")

        f = self.tests[name]

        print(f"\n=== RUN TEST: {name} ===")

        try:
            return f["func"](self.modbus, *f["args"], **f["kwargs"])

        except Exception as e:
            print(f"[ERROR] Test '{name}' failed:", e)
            self.emergency_shutdown()
            return None

    # --------------------------
    # Emergency Shutdown
    # --------------------------
    def emergency_shutdown(self, coil_range=("0x4000", "0x40FF"), reg_range=(0, 200)):
        print("\n*** EMERGENCY SHUTDOWN ACTIVATED ***")

        # OFF all coils
        start = int(coil_range[0], 16)
        end = int(coil_range[1], 16)
        for addr in range(start, end + 1):
            try:
                self.modbus.digital_write(addr, 0)
            except:
                pass

        # Reset Registers
        r_start, r_end = reg_range
        for addr in range(r_start, r_end + 1):
            try:
                self.modbus.write_holding_register(addr, 0)
            except:
                pass

        print("All coils and holding registers reset to 0.")


# =====================================================================
# MAIN PROGRAM
# =====================================================================
if __name__ == "__main__":

    modbus = ModbusTCP(host="192.168.1.100")
    manager = TestManager(modbus)

    # Register available tests
    manager.register(
        "coil_output",
        test_coil_output,
        args=("0x4000", "0x400F")
    )

    manager.register(
        "input_status",
        test_input_status,
        args=("0x0000", "0x000F")
    )

    manager.register(
        "write_register",
        test_write_register,
        args=("1025",),
        kwargs={"value": 0}
    )

    # RUN
    try:
        if not modbus.connect():
            raise ConnectionError("Cannot connect to Modbus device.")

        print("\n===== AVAILABLE TESTS =====")
        for i, name in enumerate(manager.tests.keys(), 1):
            print(f"{i}) {name}")

        choice = int(input("Select test number: "))
        test_name = list(manager.tests.keys())[choice - 1]

        manager.run(test_name)

    except Exception as e:
        print("Error:", e)
        manager.emergency_shutdown()

    finally:
        modbus.disconnect()
        print("Modbus connection closed.")
