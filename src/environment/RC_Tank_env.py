import numpy as np


class RC_Tank_Env:
    """
    Environment จำลองระบบ RC Tank (Resistor–Capacitor Tank)
    ------------------------------------------------------------
    ใช้เพื่อจำลองระดับน้ำ (level) ภายในถังที่ตอบสนองต่อแรงดัน (voltage)
    หรือกระแส (current) ที่ส่งเข้าไปควบคุม
    ------------------------------------------------------------
    Attributes:
        R, C          : ค่าความต้านทานและค่าความจุ
        dt            : ช่วงเวลาในการจำลอง
        control_mode  : 'voltage' หรือ 'current'
        setpoint_level: ระดับน้ำเป้าหมาย
        level_max     : ค่าระดับน้ำสูงสุด
        max_action_volt / max_action_current: ขอบเขตของ action
    """

    def __init__(self, R=1.5, C=2.0, dt=0.1,
                 control_mode='voltage',
                 setpoint_level=5.0,
                 level_max=10.0,
                 max_action_volt=24.0,
                 max_action_current=5.0):

        # ตรวจสอบโหมดควบคุม
        if control_mode not in ['voltage', 'current']:
            raise ValueError("control_mode must be 'voltage' or 'current'")

        # กำหนดค่าพารามิเตอร์หลัก
        self.R = R
        self.C = C
        self.dt = dt
        self.mode = control_mode
        self.setpoint_level = setpoint_level
        self.level_max = level_max
        self.max_action_volt = max_action_volt
        self.max_action_current = max_action_current

        # ตัวแปรสถานะเริ่มต้น
        self.level = 0.0
        self.time = 0.0

    # ------------------------------------------------------------
    # Reset environment
    # ------------------------------------------------------------
    def reset(self, default=0.0):
        """
        รีเซ็ตสถานะของระบบ
        Args:
            default (float): ค่าระดับเริ่มต้น (หากไม่กำหนดจะสุ่ม)
        Returns:
            (level, done)
        """
        if default is None:
            self.level = np.random.uniform(0, self.level_max)
        else:
            self.level = float(default)

        self.time = 0.0
        done = False
        return float(self.level), done

    # ------------------------------------------------------------
    # Step simulation
    # ------------------------------------------------------------
    def step(self, action):
        """
        เดินการจำลอง 1 timestep
        Args:
            action (float): ค่าคำสั่งควบคุมแรงดันหรือกระแส
        Returns:
            (level, done)
        """
        # ควบคุมด้วยแรงดัน
        if self.mode == 'voltage':
            action = np.clip(action, 0, self.max_action_volt)
            current = (action - self.level) / self.R
            delta_level = (current / self.C) * self.dt

        # ควบคุมด้วยกระแส
        elif self.mode == 'current':
            action = np.clip(action, 0, self.max_action_current)
            net_flow = action - (self.level / self.R)
            delta_level = (net_flow / self.C) * self.dt

        else:
            raise ValueError("Invalid control_mode")

        # อัปเดตสถานะ
        self.level += delta_level
        self.level = np.clip(self.level, 0, self.level_max)
        self.time += self.dt

        # ตรวจสอบว่าอยู่ใกล้ setpoint หรือยัง
        done = abs(self.setpoint_level - self.level) <= 0.1

        return float(self.level), bool(done)
