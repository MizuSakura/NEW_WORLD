import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.environment.RC_Tank_env import RC_Tank_Env
from src.environment.signal_generator import SignalGenerator

# ============================================
# 1️⃣ Vanilla LSTM
# ============================================
class VanillaLSTM_MODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.0, stateful=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stateful = stateful

        # ถ้า num_layers=1, dropout ต้องเป็น 0
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_state = None

    def reset_state(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, x):
        if self.stateful and self.hidden_state is not None:
            out, self.hidden_state = self.lstm(x, self.hidden_state)
        else:
            out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def detach_state(self):
        if self.hidden_state is not None:
            self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())

# ============================================
# 2️⃣ Deep LSTM (LSTM + Dense Layers)
# ============================================
class DeepLSTM_MODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, fc_units=[64, 32], dropout=0.1, stateful=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stateful = stateful

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)

        layers = []
        in_features = hidden_dim
        for u in fc_units:
            layers += [nn.Linear(in_features, u), nn.ReLU(), nn.Dropout(dropout)]
            in_features = u
        layers.append(nn.Linear(in_features, output_dim))
        self.fc = nn.Sequential(*layers)

        self.hidden_state = None

    def reset_state(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, x):
        if self.stateful and self.hidden_state is not None:
            out, self.hidden_state = self.lstm(x, self.hidden_state)
        else:
            out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def detach_state(self):
        if self.hidden_state is not None:
            self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())

# ============================================
# 3️⃣ Bidirectional LSTM
# ============================================
class BiLSTM_MODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, fc_units=[64], dropout=0.1, stateful=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stateful = stateful

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=lstm_dropout)

        in_features = hidden_dim * 2
        layers = []
        for u in fc_units:
            layers += [nn.Linear(in_features, u), nn.ReLU(), nn.Dropout(dropout)]
            in_features = u
        layers.append(nn.Linear(in_features, output_dim))
        self.fc = nn.Sequential(*layers)

        self.hidden_state = None

    def reset_state(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_layers*2, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers*2, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, x):
        if self.stateful and self.hidden_state is not None:
            out, self.hidden_state = self.lstm(x, self.hidden_state)
        else:
            out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def detach_state(self):
        if self.hidden_state is not None:
            self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())



if __name__ == "__main__":
    # Hyperparameters
    seq_len = 5
    input_dim = 1
    output_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # สร้าง environment และ signal
    env = RC_Tank_Env()
    signal_gen = SignalGenerator(t_end=50, dt=env.dt)
    t, u_signal = signal_gen.pwm(amplitude=5.0, freq=0.05, duty=0.5)

    # สร้าง dataset จาก RC Tank simulation
    levels = []
    env.reset()
    for u in u_signal:
        level, _ = env.step(u)
        levels.append(level)
    levels = np.array(levels)

    # สร้าง sequences สำหรับ LSTM
    X_seq, y_seq = [], []
    for i in range(len(levels) - seq_len):
        X_seq.append(levels[i:i+seq_len])
        y_seq.append(levels[i+seq_len])
    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32).unsqueeze(-1)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_seq, y_seq)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # สร้างโมเดล LSTM ทั้ง 3 แบบ
    models = {
        "VanillaLSTM": VanillaLSTM_MODEL(input_dim=input_dim, hidden_dim=32),
        "DeepLSTM": DeepLSTM_MODEL(input_dim=input_dim, hidden_dim=64, fc_units=[32,16]),
        "BiLSTM": BiLSTM_MODEL(input_dim=input_dim, hidden_dim=32, fc_units=[32])
    }

    # ทดสอบ prediction และ plot
    plt.figure(figsize=(12,6))
    plt.plot(t[seq_len:], levels[seq_len:], label="True Level", color='black', linewidth=2)

    for name, model in models.items():
        model.to(device)
        model.eval()
        model.reset_state(batch_size=16, device=device)
        preds_all = []

        for xb, _ in dataloader:
            xb = xb.to(device)
            with torch.no_grad():
                pred = model(xb)
            preds_all.append(pred.cpu().numpy())
            if model.stateful:
                model.detach_state()

        preds_all = np.vstack(preds_all).flatten()
        plt.plot(t[seq_len:], preds_all, label=f"{name} Prediction", alpha=0.8)

    plt.xlabel("Time [s]")
    plt.ylabel("Tank Level")
    plt.title("RC Tank Level Prediction by LSTM Models")
    plt.legend()
    plt.grid(True)
    plt.show()