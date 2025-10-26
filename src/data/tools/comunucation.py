#NEW_WORLD/src/data/tools/comunucation.py
from pymodbus.client import ModbusTcpClient
import time

# ======================================================================
# ⚙️ Modbus TCP Client Wrapper (with NumPy-style docstrings)
# ======================================================================
class ModbusTCP:
    """
    High-level wrapper for Modbus TCP communication using `pymodbus`.

    This class provides an easy-to-use interface for reading and writing
    Modbus coils and registers over TCP/IP.

    Examples
    --------
    >>> modbus = ModbusTCP(host="192.168.1.100", port=502)
    >>> modbus.connect()
    True
    >>> modbus.digital_write(0x4000, True)
    True
    >>> modbus.analog_read(0x0001)
    1234
    >>> modbus.disconnect()
    True
    """

    def __init__(self, host='192.168.1.100', port=502):
        """
        Initialize a Modbus TCP client.

        Parameters
        ----------
        host : str, optional
            IP address of the Modbus TCP server.
        port : int, optional
            TCP port number (default is 502).
        """
        self.client = ModbusTcpClient(host=host, port=port)
        self.host = host
        self.port = port

    # ------------------------------------------------------------------
    def connect(self):
        """
        Establish connection to the Modbus TCP server.

        Returns
        -------
        bool
            True if connection is successful, False otherwise.
        """
        self.client.connect()
        return self.client.connected

    # ------------------------------------------------------------------
    def disconnect(self):
        """
        Close connection to the Modbus TCP server.

        Returns
        -------
        bool
            True if successfully disconnected, False otherwise.
        """
        self.client.close()
        return not self.client.connected

    # ==================================================================
    # 🔹 DIGITAL OUTPUTS (COILS)
    # ==================================================================
    def read_status_output(self, address):
        """
        Read the ON/OFF state of a single coil output.

        Parameters
        ----------
        address : int
            Coil address (e.g., 0x4000).

        Returns
        -------
        bool or None
            Coil state (True = ON, False = OFF), or None if error occurred.
        """
        response = self.client.read_coils(address, 1)
        if response.isError():
            print(f"❌ Error reading coil @ {address}")
            return None
        return response.bits[0]

    # ------------------------------------------------------------------
    def digital_write(self, address, value):
        """
        Write a single digital coil (Function Code 05).

        Parameters
        ----------
        address : int
            Target coil address.
        value : bool
            Coil value (True = ON, False = OFF).

        Returns
        -------
        bool or None
            True if successful, None if failed.
        """
        response = self.client.write_coil(address, value)
        if response.isError():
            print(f"❌ Error writing coil @ {address}")
            return None
        return True

    # ------------------------------------------------------------------
    def multiple_digital_write(self, address, values):
        """
        Write multiple consecutive coils (Function Code 15).

        Parameters
        ----------
        address : int
            Starting coil address.
        values : list of bool
            Sequence of ON/OFF states to write.

        Returns
        -------
        bool or None
            True if successful, None if failed.
        """
        response = self.client.write_coils(address, values)
        if response.isError():
            print(f"❌ Error writing multiple coils @ {address}")
            return None
        return True

    # ==================================================================
    # 🔹 DIGITAL INPUTS
    # ==================================================================
    def digital_input(self, address, count=1):
        """
        Read discrete input states (Function Code 02).

        Parameters
        ----------
        address : int
            Starting address of input coil(s).
        count : int, optional
            Number of inputs to read (default = 1).

        Returns
        -------
        bool or list of bool or None
            Single boolean if `count=1`, list of booleans if `count>1`,
            or None if error occurred.
        """
        response = self.client.read_discrete_inputs(address=address, count=count)
        if response.isError():
            print(f"❌ Error reading digital input @ {address}")
            return None
        return response.bits[0] if count == 1 else response.bits

    # ==================================================================
    # 🔹 ANALOG INPUTS
    # ==================================================================
    def analog_read(self, address, count=1):
        """
        Read analog input registers (Function Code 04).

        Parameters
        ----------
        address : int
            Starting input register address.
        count : int, optional
            Number of registers to read (default = 1).

        Returns
        -------
        int or list[int] or None
            Single register value if `count=1`, list of registers otherwise,
            or None if error occurred.
        """
        response = self.client.read_input_registers(address=address, count=count)
        if response.isError():
            print(f"❌ Error reading analog input @ {address}")
            return None
        return response.registers[0] if count == 1 else response.registers

    # ==================================================================
    # 🔹 HOLDING REGISTERS (R/W)
    # ==================================================================
    def read_holding_registers(self, address, count=1, slave_id=1):
        """
        Read one or more holding registers (Function Code 03).

        Parameters
        ----------
        address : int
            Starting register address.
        count : int, optional
            Number of registers to read (default = 1).
        slave_id : int, optional
            Modbus slave ID (default = 1).

        Returns
        -------
        list[int] or None
            List of register values, or None if error occurred.
        """
        response = self.client.read_holding_registers(address=address, count=count, unit=slave_id)
        if response.isError():
            print(f"❌ Error reading holding register @ {address}")
            return None
        return response.registers

    # ------------------------------------------------------------------
    def write_holding_register(self, address, value):
        """
        Write a single holding register (Function Code 06).

        Parameters
        ----------
        address : int
            Target register address.
        value : int
            Value to write to the register.

        Returns
        -------
        bool or None
            True if successful, None if failed.
        """
        response = self.client.write_register(address, value)
        if response.isError():
            print(f"❌ Error writing holding register @ {address}")
            return None
        return True

    # ------------------------------------------------------------------
    def multiple_write_holding_registers(self, address, values):
        """
        Write multiple consecutive holding registers (Function Code 16).

        Parameters
        ----------
        address : int
            Starting register address.
        values : list[int]
            List of values to write.

        Returns
        -------
        bool or None
            True if successful, None if failed.
        """
        response = self.client.write_registers(address, values)
        if response.isError():
            print(f"❌ Error writing multiple registers @ {address}")
            return None
        return True


# ======================================================================
# 🧩 Address Parser
# ======================================================================
def parse_address(value):
    """
    Parse user-provided Modbus address into base-10 integer.

    Supports input in various formats:
    - Decimal integer (e.g., `16384`)
    - Hex string (e.g., `"0x4000"`, `"4000h"`)
    - Hex integer (e.g., `0x4000`)

    Parameters
    ----------
    value : int or str
        The address to parse.

    Returns
    -------
    int
        Address converted to base-10 integer.

    Examples
    --------
    >>> parse_address('0x4000')
    16384
    >>> parse_address('4000h')
    16384
    >>> parse_address(0x4000)
    16384
    """
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    elif s.endswith("h"):
        return int(s[:-1], 16)
    return int(s, 10)

# ==========================================================
# 🔹 Helper Logging
# ==========================================================
def log_action(action, addr, success):
    status = "✅ OK" if success else "❌ FAIL"
    print(f"{action:<30} @ {addr:>5} ({hex(addr):>8}) → {status}")

# ==========================================================
# 🔹 Main Testing Routine
# ==========================================================
if __name__ == "__main__":
    IP_HOST = "192.168.1.100"
    PORT_PROTOCOL = 502

    # ---------------------------
    # Define Address Ranges
    # ---------------------------
    COIL_OUT_START, COIL_OUT_END = 0x4000, 0x4015
    COIL_IN_START, COIL_IN_END = 0x0000, 0x0010
    REG_OUT_START, REG_OUT_END = 0x0401, 0x0402
    REG_IN_START, REG_IN_END = 0x0001, 0x0002

    # ---------------------------
    # Connect
    # ---------------------------
    print("Connecting to Modbus TCP server...")
    modbus = ModbusTCP(host=IP_HOST, port=PORT_PROTOCOL)

    if not modbus.connect():
        print("❌ Failed to connect to Modbus TCP server.")
        exit(1)
    print("✅ Connected!\n")

    # ---------------------------
    # DIGITAL OUTPUT TEST
    # ---------------------------
    print(f"\n🔸 DIGITAL OUTPUTS  ({hex(COIL_OUT_START)}–{hex(COIL_OUT_END)})")

    for state, label in [(True, "ON"), (False, "OFF")]:
        for addr in range(COIL_OUT_START, COIL_OUT_END + 1):
            result = modbus.digital_write(addr, state)
            log_action(f"Write {label}", addr, result)
            time.sleep(0.05)

    # ---------------------------
    # DIGITAL INPUT TEST
    # ---------------------------
    print(f"\n🔹 DIGITAL INPUTS  ({hex(COIL_IN_START)}–{hex(COIL_IN_END)})")
    for addr in range(COIL_IN_START, COIL_IN_END + 1):
        state = modbus.digital_input(addr)
        log_action(f"Read Input ({state})", addr, state is not None)
        time.sleep(0.05)

    # ---------------------------
    # ANALOG OUTPUT TEST
    # ---------------------------
    print(f"\n🔸 ANALOG OUTPUTS  ({hex(REG_OUT_START)}–{hex(REG_OUT_END)})")
    for value in [24647, 0]:
        for addr in range(REG_OUT_START, REG_OUT_END + 1):
            result = modbus.write_holding_register(addr, value)
            log_action(f"Write Reg ({value})", addr, result)
            time.sleep(0.05)

    # ---------------------------
    # ANALOG INPUT TEST
    # ---------------------------
    print(f"\n🔹 ANALOG INPUTS  ({hex(REG_IN_START)}–{hex(REG_IN_END)})")
    for addr in range(REG_IN_START, REG_IN_END + 1):
        values = modbus.analog_read(addr, count=1)
        print(f"Read {addr:>5} ({hex(addr):>8}) → {values}")
        time.sleep(0.05)

    # ---------------------------
    # Disconnect
    # ---------------------------
    modbus.disconnect()
    print("\n🔚 Disconnected from Modbus server.")
