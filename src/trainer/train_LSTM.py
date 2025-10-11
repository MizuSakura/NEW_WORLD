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
import warnings

# cuDNN benchmark (good when input sizes are constant)
torch.backends.cudnn.benchmark = True

# Try to import BackgroundGenerator for async prefetch; fallback to no-prefetch
try:
    from prefetch_generator import BackgroundGenerator  # type: ignore
    HAS_PREFETCH_GEN = True
except Exception:
    HAS_PREFETCH_GEN = False
    BackgroundGenerator = None

# DataLoader wrapper to optionally prefetch batches in background
class DataLoaderX(DataLoader):
    def __iter__(self):
        if HAS_PREFETCH_GEN and BackgroundGenerator is not None:
            return BackgroundGenerator(super().__iter__())
        else:
            return super().__iter__()

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

                 # performance options
                 use_cache=False,            # ⚡ ถ้าปรับเป็น True จะสร้าง/load binary cache (torch) เพื่อลด I/O
                 cache_path=None,            # ที่เก็บ cache ถ้า None จะสร้างใกล้ model_save_path
                 auto_num_workers=True,      # ใช้ os.cpu_count()-2
                 prefetch_factor=8,          # จำนวน batches ที่ worker จะ prefetch (สูงขึ้นช่วยให้ GPU ได้ข้อมูลเร็วขึ้น)
                 persistent_workers=True,    # worker คงอยู่ข้าม epoch
                 pin_memory=True,            # พยายามใช้ pinned memory
                 
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
        self.cache_path = Path(cache_path) if cache_path is not None else (self.model_save_path.parent / "dataset_cache.pt")

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

        # performance options
        self.use_cache = use_cache
        self.auto_num_workers = auto_num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

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

    # --------------------------
    # Helper: build sequences & optionally save cache
    # --------------------------
    def _build_cache_from_csvs(self):
        """
        อ่าน CSV ทั้งหมด (ใช้ ScalingZipLoader) -> สร้าง X,y sequences -> บันทึกเป็น torch file
        ระวัง: การสร้าง cache อาจใช้ RAM ขึ้นอยู่กับ dataset ขนาดใหญ่
        """
        print(f"[CACHE] Building cache at: {self.cache_path} ...")
        scaling_loader = ScalingZipLoader(self.scaler_zip)
        scaler_in = scaling_loader.scaler_in
        scaler_out = scaling_loader.scaler_out

        X_list = []
        y_list = []

        files = [f for f in os.listdir(self.data_folder) if f.endswith(self.file_ext)]
        for file_name in files:
            fp = self.data_folder / file_name
            for chunk in np.array_split(pd.read_csv(fp), max(1, int(np.ceil(len(pd.read_csv(fp)) / max(1, self.chunksize))))):  # safe chunk reading
                # scale
                in_vals = chunk[self.input_col].values if isinstance(self.input_col, str) else chunk[self.input_col]
                out_vals = chunk[self.output_col].values if isinstance(self.output_col, str) else chunk[self.output_col]

                try:
                    in_scaled = scaler_in.transform(in_vals.reshape(-1, 1)) if in_vals.ndim == 1 else scaler_in.transform(in_vals)
                except Exception:
                    in_scaled = np.array(in_vals).reshape(-1, 1)
                try:
                    out_scaled = scaler_out.transform(out_vals.reshape(-1, 1)) if out_vals.ndim == 1 else scaler_out.transform(out_vals)
                except Exception:
                    out_scaled = np.array(out_vals).reshape(-1, 1)

                # create sequences (many-to-one)
                L = len(in_scaled)
                if L < self.window_size:
                    # pad
                    pad_size = self.window_size - L
                    pad = np.full((pad_size, in_scaled.shape[1]), self.pad_value)
                    X_seq = np.vstack((pad, in_scaled))
                    y_seq = out_scaled[-1]
                    X_list.append(X_seq.astype(np.float32))
                    y_list.append(y_seq.astype(np.float32))
                else:
                    for i in range(0, L - self.window_size):
                        X_list.append(in_scaled[i:i+self.window_size].astype(np.float32))
                        y_list.append(out_scaled[i+self.window_size].astype(np.float32))

        # stack and save
        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
        # convert to tensors and save
        torch.save({'X': torch.from_numpy(X), 'y': torch.from_numpy(y)}, self.cache_path)
        print(f"[CACHE] Saved cache: {self.cache_path} | samples: {len(X)}")

    # ============================================
    # 🔹 Prepare Dataset
    # ============================================
    def prepare_data(self):
        # If asked to use cache, check and build if necessary
        if self.use_cache and not self.cache_path.exists():
            try:
                import pandas as pd  # local import to avoid top-level requirement if not used
                self._build_cache_from_csvs()
            except Exception as e:
                warnings.warn(f"[CACHE] Failed to build cache automatically: {e}. Falling back to file-based dataset.")
                self.use_cache = False

        # Create dataset
        if self.use_cache and self.cache_path.exists():
            # load cached tensors -> create TensorDataset-like wrapper
            data = torch.load(self.cache_path)
            X = data['X']
            y = data['y']

            class CachedDataset(torch.utils.data.Dataset):
                def __init__(self, X, y):
                    self.X = X
                    self.y = y
                def __len__(self):
                    return self.X.shape[0]
                def __getitem__(self, idx):
                    return self.X[idx].numpy(), self.y[idx].numpy()

            self.dataset = CachedDataset(X, y)
            print(f"[INFO] Loaded dataset from cache: {self.cache_path} | samples: {len(self.dataset)}")

        else:
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

        # build DataLoader with automatic workers
        if self.auto_num_workers:
            cpu_count = os.cpu_count() or 1
            # reserve 1-2 cores for system/other tasks
            num_workers = max(1, cpu_count - 2)
        else:
            num_workers = min(8, os.cpu_count() or 1)

        # Safety: if dataset is small, reduce workers to avoid overhead
        if len(self.dataset) < 1000:
            num_workers = min(num_workers, 4)

        # Create DataLoaderX (with optional background prefetch)
        self.dataloader = DataLoaderX(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if num_workers > 0 else 2
        )

        print(f"[INFO] Dataset ready: {len(self.dataset)} samples | batch size = {self.batch_size} | num_workers={num_workers} | prefetch_gen={'YES' if HAS_PREFETCH_GEN else 'NO'}")
        if torch.cuda.is_available():
            print(f"[GPU INFO] Using: {torch.cuda.get_device_name(0)} | Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")

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

        # Try compiling (PyTorch >= 2.0) - may speed up forward/backward
        try:
            if hasattr(torch, "compile"):
                self.model = torch.compile(self.model)
        except Exception as e:
            warnings.warn(f"[COMPILE] torch.compile failed or not supported: {e}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[INFO] Model {self.model_type} built successfully on {self.device}")

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

        # ✅ Compatible GradScaler
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))
            use_amp = "torch.amp"
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
            use_amp = "torch.cuda.amp"

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            for xb, yb in self.dataloader:
                xb = xb.to(self.device, non_blocking=True).float()
                yb = yb.to(self.device, non_blocking=True).float()
                self.optimizer.zero_grad(set_to_none=True)

                # ✅ Compatible autocast
                try:
                    context = torch.amp.autocast('cuda', enabled=(self.device.type == "cuda"))
                except TypeError:
                    context = torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"))

                with context:
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            train_losses.append(avg_loss)

            if epoch % 5 == 0 or epoch == 1:
                duration = time.time() - start_time
                print(f"Epoch [{epoch}/{self.num_epochs}] - Loss: {avg_loss:.6f} | Time: {duration:.2f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("[INFO] Early stopping triggered.")
                    break

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
        plt.plot(train_losses, label="Training Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example run: high-performance mode with caching + prefetching
    trainer = TRAIN_MODEL(
        data_folder=r"D:\Project_end\New_world\my_project\data\raw",
        scaler_zip=r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip",
        model_save_path=r"D:\Project_end\New_world\my_project\models\lstm_model.pth",
        dataset_type="lazy",
        model_type="DeepLSTM",
        num_epochs=50,
        batch_size=2048,   # tune according to your GPU mem
        hidden_dim=128,
        use_cache=False,   # set True to build binary cache (faster I/O but needs extra disk & memory to build)
        cache_path=None,   # optional custom path
        auto_num_workers=True,
        prefetch_factor=8,
        persistent_workers=True,
        pin_memory=True
    )

    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
