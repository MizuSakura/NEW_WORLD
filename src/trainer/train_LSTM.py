# my_project/src/trainer/train_LSTM.py
from src.data.scaling_loader import ScalingZipLoader
from src.data.sequence_builder import LazyChunkedSequenceDataset, SequenceDataset
from src.models.lstm_model import VanillaLSTM_MODEL, DeepLSTM_MODEL, BiLSTM_MODEL
from torch.utils.data import DataLoader
from src.utils.logger import Logger
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ⚡ Enable cuDNN benchmark for fixed-size LSTM input (GPU speedup)
torch.backends.cudnn.benchmark = True


class TRAIN_MODEL:
    def __init__(self,
                 data_folder,
                 scaler_zip,
                 model_save_path,

                 # sequence_builder
                 window_size=30,
                 input_col='DATA_INPUT',
                 output_col='DATA_OUTPUT',
                 batch_size=64,
                 file_ext='.csv',
                 chunksize=1000,
                 allow_padding=True,
                 pad_value=0.0,
                 dataset_type="full",

                 # model & training
                 model_type="DeepLSTM",
                 hidden_dim=128,
                 num_layers=2,
                 lr=1e-3,
                 num_epochs=100,
                 patience=20,
                 device=None,

                 # metadata
                 user_create=None,
                 name_file=None,
                 time_format="%Y-%m-%d %H:%M:%S",
                 project_version="1.0.0",
                 description="Scaling reference for ML model preprocessing",
                 notes=None):

        # Paths
        self.data_folder = Path(data_folder)
        self.scaler_zip = Path(scaler_zip)
        self.model_save_path = Path(model_save_path)

        # Sequence
        self.window_size = window_size
        self.input_col = input_col
        self.output_col = output_col
        self.batch_size = batch_size
        self.file_ext = file_ext
        self.chunksize = chunksize
        self.allow_padding = allow_padding
        self.pad_value = pad_value
        self.dataset_type = dataset_type

        # Model & training config
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Metadata
        self.user_create = user_create
        self.name_file = name_file
        self.time_format = time_format
        self.project_version = project_version
        self.description = description
        self.notes = notes

        # Internal attributes
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.logger = Logger()

    # ============================================
    # 🔹 Prepare Dataset
    # ============================================
    def prepare_data(self):
        if self.dataset_type == "full":
            print("[INFO] Using SequenceDataset (full load)")
            self.dataset = SequenceDataset(
                folder_path=self.data_folder,
                scale_path=self.scaler_zip,
                sequence_size=self.window_size,
                input_col=[self.input_col],
                output_col=[self.output_col],
                chunksize=self.chunksize,
                allow_padding=self.allow_padding,
                pad_value=self.pad_value
            )
        else:
            print("[INFO] Using LazyChunkedSequenceDataset (lazy load)")
            self.dataset = LazyChunkedSequenceDataset(
                folder_path=self.data_folder,
                scale_path=self.scaler_zip,
                sequence_size=self.window_size,
                input_col=[self.input_col],
                output_col=[self.output_col],
                chunksize=self.chunksize
            )

        # ⚡ Optimized DataLoader configuration
        num_workers = min(8, os.cpu_count() or 1)
        prefetch_factor = 4 if num_workers > 0 else None
        pin_memory = torch.cuda.is_available()
        persistent_workers = True if num_workers > 0 else False

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )

        print(f"[DATA] num_workers={num_workers} | prefetch_factor={prefetch_factor} | pin_memory={pin_memory}")
        print(f"[INFO] Dataset ready: {len(self.dataset)} samples | batch size = {self.batch_size}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU INFO] Using GPU: {gpu_name} | Initial Memory: {torch.cuda.memory_allocated()/1e6:.2f} MB")

    # ============================================
    # 🔹 Build Model
    # ============================================
    def build_model(self):
        input_dim = len([self.input_col])
        output_dim = len([self.output_col])

        if self.model_type == "VanillaLSTM":
            self.model = VanillaLSTM_MODEL(input_dim, hidden_dim=self.hidden_dim,
                                           num_layers=self.num_layers, output_dim=output_dim)
        elif self.model_type == "DeepLSTM":
            self.model = DeepLSTM_MODEL(input_dim, hidden_dim=self.hidden_dim,
                                        num_layers=self.num_layers, output_dim=output_dim)
        elif self.model_type == "BiLSTM":
            self.model = BiLSTM_MODEL(input_dim, hidden_dim=self.hidden_dim,
                                      num_layers=self.num_layers, output_dim=output_dim)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.to(self.device)

        # ⚡ compile for speed (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("[INFO] Model compiled with torch.compile() for speed boost.")
            except Exception as e:
                print(f"[WARN] torch.compile() failed: {e}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[INFO] Model {self.model_type} built successfully on {self.device}")

    # ============================================
    # 🔹 Train Model (optimized)
    # ============================================
    # ============================================
    # 🔹 Train Model (optimized)
    # ============================================
    def train(self):
        if self.dataset is None or self.model is None:
            raise RuntimeError("Dataset or Model not prepared. Run prepare_data() and build_model() first.")

        print(f"[TRAINING] Start training {self.model_type} for {self.num_epochs} epochs ...")
        best_loss = np.inf
        best_state = None
        train_losses = []
        patience_counter = 0

        # Enable mixed precision
        scaler = torch.amp.GradScaler(self.device.type, enabled=(self.device.type == "cuda"))

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            # สร้าง Progress Bar สำหรับ Dataloader
            progress_bar = tqdm(self.dataloader, desc=f"Epoch [{epoch}/{self.num_epochs}]", unit="batch")

            for xb, yb in progress_bar:
                xb = xb.to(self.device, non_blocking=True).float()
                yb = yb.to(self.device, non_blocking=True).float()

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda")):
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()

                if hasattr(self.model, "detach_state"):
                    self.model.detach_state()
                
                # อัปเดตค่า Loss ที่แสดงบน Progress Bar
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = total_loss / len(self.dataloader)
            train_losses.append(avg_loss)
            duration = time.time() - start_time
            
            # แสดงผลสรุปของ Epoch หลังทำเสร็จ
            tqdm.write(f"Epoch [{epoch}/{self.num_epochs}] -> Avg Loss: {avg_loss:.6f} | Time: {duration:.2f}s")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("[INFO] Early stopping triggered.")
                    break

        # Save best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"[SAVED] Model saved to {self.model_save_path}")

        self.plot_loss(train_losses)

    # ============================================
    # 🔹 Plot training loss
    # ============================================
    def plot_loss(self, train_losses):
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Training Loss", color='tab:blue')
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

def _get_or_create_scaler(project_root: Path, cfg: dict) -> Path:
    from src.utils.scale_referance import GlobalScalingReference

    config_dir = project_root / "config"
    data_dir   = project_root / cfg.get("data_folder", "data/raw")

    input_col  = cfg.get("input_col",  "DATA_INPUT")
    output_col = cfg.get("output_col", "DATA_OUTPUT")

    # ชื่อ zip ตาม input/output col
    zip_name   = f"AutoScaler_{input_col}_{output_col}_scalers.zip"
    zip_path   = config_dir / zip_name

    if zip_path.exists():
        print(f"[Scaler] Found existing scaler: {zip_path.name}")
        return zip_path

    print("[Scaler] Creating scaler from data/raw...")

    scaler_ref = GlobalScalingReference(
        user_create     = "auto",
        name_project    = "RC_Tank_RL",
        data_dir        = data_dir,
        save_dir        = config_dir,
        dataset_name    = f"AutoScaler_{input_col}_{output_col}",
        input_features  = [input_col],     # ← แค่ 1 column
        output_features = [output_col],
        scaler_type     = "MinMaxScaler",
        chunk_size      = 10000,
    )

    result   = scaler_ref.run()
    zip_path = result["zip"]
    print(f"[Scaler] Created: {zip_path.name}")
    return zip_path

# ============================================
# 🔹 Entry Point
# ============================================
if __name__ == "__main__":
    import yaml

    PROJECT_ROOT   = Path(__file__).resolve().parents[2]
    RL_CONFIG_PATH = PROJECT_ROOT / "src" / "API" / "config" / "rl_params.yaml"

    if RL_CONFIG_PATH.exists():
        with open(RL_CONFIG_PATH, "r", encoding="utf-8") as f:
            rl_cfg = yaml.safe_load(f)
    else:
        rl_cfg = {}

    cfg = rl_cfg.get("lstm_csv", {})

    # หรือสร้าง scaler อัตโนมัติ
    scaler_zip = _get_or_create_scaler(PROJECT_ROOT, cfg)

    trainer = TRAIN_MODEL(
        data_folder     = str(PROJECT_ROOT / cfg.get("data_folder",     "data/raw")),
        scaler_zip      = str(scaler_zip),     # ← ใช้ path จาก helper
        model_save_path = str(PROJECT_ROOT / cfg.get("model_save_path", "models/lstm_model.pth")),
        dataset_type    = cfg.get("dataset_type", "lazy"),
        model_type      = cfg.get("model_type",   "DeepLSTM"),
        num_epochs      = cfg.get("num_epochs",   50),
        batch_size      = cfg.get("batch_size",   8192),
        hidden_dim      = cfg.get("hidden_dim",   128),
        num_layers      = cfg.get("num_layers",   2),
        window_size     = cfg.get("window_size",  30),
        input_col       = cfg.get("input_col",    "DATA_INPUT"),
        output_col      = cfg.get("output_col",   "DATA_OUTPUT"),
    )

    trainer.prepare_data()
    trainer.build_model()
    trainer.train()