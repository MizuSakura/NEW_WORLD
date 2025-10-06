import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    LSTM ที่ออกแบบให้ยืดหยุ่นสำหรับทั้งงานพยากรณ์ข้อมูล (Time Series)
    และรองรับการต่อยอดเป็นระบบควบคุมแบบ stateful หรือ RL-based Control ได้ในอนาคต
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1, stateful=False):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stateful = stateful

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

        # เก็บสถานะถ้า stateful=True
        self.hidden_state = None

    def reset_state(self, batch_size=1, device=None):
        """รีเซ็ต hidden/cell state (ใช้ในตอนเริ่ม simulation หรือ episode ใหม่)"""
        if device is None:
            device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        output: (batch_size, output_dim)
        """
        if self.stateful and self.hidden_state is not None:
            out, self.hidden_state = self.lstm(x, self.hidden_state)
        else:
            out, _ = self.lstm(x)

        # ดึงค่า timestep สุดท้าย
        out = self.fc(out[:, -1, :])
        return out

    def detach_state(self):
        """ใช้ในตอน train เพื่อตัด gradient จาก state เดิม (ป้องกัน gradient overflow)"""
        if self.hidden_state is not None:
            self.hidden_state = (
                self.hidden_state[0].detach(),
                self.hidden_state[1].detach()
            )


if __name__ == "__main__":
    # 🧪 ทดสอบการทำงานของ LSTMForecaster
    batch_size, seq_len, input_dim = 4, 10, 2
    model = LSTMForecaster(input_dim=input_dim, hidden_dim=32, output_dim=1, stateful=True)

    # สร้าง input จำลอง
    x = torch.randn(batch_size, seq_len, input_dim)

    # reset state ก่อนเริ่ม
    model.reset_state(batch_size=batch_size)

    # forward pass
    y = model(x)
    print("✅ Output shape:", y.shape)
    print("Sample output:", y.detach().cpu().numpy().flatten()[:5])
