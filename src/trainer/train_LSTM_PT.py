# my_project/src/trainer/train_LSTM_PT.py
# ======================================================================
# ⚙️ TRAINING LSTM MODEL WITH .PT DATASET
# ======================================================================

from src.data.sequence_builder import PTLazyChunkedSequenceDataset, PTSequenceDataset
from src.models.lstm_model import VanillaLSTM_MODEL, DeepLSTM_MODEL, BiLSTM_MODEL
from src.utils.logger import Logger
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import zipfile
from pathlib import Path
import shutil

def unpack_dataset_zip(zip_path, target_folder, cleanup=False):
    """
    Unpack dataset zip to a specified folder.
    
    Args:
        zip_path (str | Path): Path to the zip file.
        target_folder (str | Path): Folder where files will be extracted.
        cleanup (bool): If True, remove the zip file after extraction.
        
    Returns:
        Path: Path to the folder with extracted files.
    """
    zip_path = Path(zip_path)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(target_folder)
    print(f"[INFO] Dataset extracted to: {target_folder}")

    if cleanup:
        try:
            os.remove(zip_path)
            print(f"[INFO] Original zip file {zip_path.name} removed.")
        except Exception as e:
            print(f"[WARNING] Could not remove zip file: {e}")

    return target_folder

torch.backends.cudnn.benchmark = True  # ⚡ GPU speedup for fixed-size inputs


# ======================================================================
# 🔹 TRAINER CLASS
# ======================================================================
class TRAIN_MODEL_PT:
    def __init__(self,
                 data_folder,
                 model_save_path,
                 dataset_type="lazy",     # "full" or "lazy"
                 model_type="DeepLSTM",   # "VanillaLSTM", "DeepLSTM", "BiLSTM"
                 batch_size=1024,
                 hidden_dim=128,
                 num_layers=2,
                 lr=1e-3,
                 num_epochs=100,
                 patience=20,num_worker=0,
                 prefetch_factor=2,
                 device=None):

        self.data_folder = Path(data_folder)
        self.model_save_path = Path(model_save_path)
        self.dataset_type = dataset_type
        self.model_type = model_type

        # Training parameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_worker  = num_worker
        self.prefetch_factor = prefetch_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Internal components
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.logger = Logger()

    # ==================================================================
    # 🔸 Prepare Dataset (.pt)
    # ==================================================================
    def prepare_data(self):
        if self.dataset_type == "full":
            print("[INFO] Using PTSequenceDataset (full load)")
            self.dataset = PTSequenceDataset(self.data_folder)
        else:
            print("[INFO] Using PTLazyChunkedSequenceDataset (lazy load)")
            self.dataset = PTLazyChunkedSequenceDataset(self.data_folder)

        num_workers =  self.num_worker
        prefetch_factor = self.prefetch_factor if num_workers > 0 else None
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
            print(f"[GPU INFO] Using GPU: {gpu_name}")

    # ==================================================================
    # 🔸 Build Model
    # ==================================================================
    def build_model(self):
        # infer input/output dim directly from one sample
        sample_x, sample_y = self.dataset[0]
        input_dim = sample_x.shape[-1]
        output_dim = sample_y.shape[-1] if sample_y.ndim > 0 else 1

        if self.model_type == "VanillaLSTM":
            self.model = VanillaLSTM_MODEL(input_dim, self.hidden_dim, self.num_layers, output_dim)
        elif self.model_type == "DeepLSTM":
            self.model = DeepLSTM_MODEL(input_dim, self.hidden_dim, self.num_layers, output_dim)
        elif self.model_type == "BiLSTM":
            self.model = BiLSTM_MODEL(input_dim, self.hidden_dim, self.num_layers, output_dim)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # ⚡ Try compiling (PyTorch 2.x+)
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("[INFO] Model compiled with torch.compile()")
            except Exception as e:
                print(f"[WARN] torch.compile() failed: {e}")

        print(f"[MODEL] {self.model_type} built successfully on {self.device}")
        print(f"Input dim: {input_dim} | Output dim: {output_dim}")

    # ==================================================================
    # 🔸 Train Model
    # ==================================================================
    def train(self):
        if self.dataset is None or self.model is None:
            raise RuntimeError("Call prepare_data() and build_model() before train().")

        print(f"[TRAINING] Start training {self.model_type} for {self.num_epochs} epochs ...")

        best_loss = np.inf
        best_state = None
        train_losses = []
        patience_counter = 0

        scaler = torch.amp.GradScaler(self.device.type, enabled=(self.device.type == "cuda"))

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            for xb, yb in tqdm(self.dataloader, desc=f"Epoch [{epoch}/{self.num_epochs}]", unit="batch"):
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

            avg_loss = total_loss / len(self.dataloader)
            train_losses.append(avg_loss)
            duration = time.time() - start_time

            tqdm.write(f"Epoch [{epoch}/{self.num_epochs}] | Loss: {avg_loss:.6f} | Time: {duration:.2f}s")

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

    # ==================================================================
    # 🔸 Plot training loss
    # ==================================================================
    def plot_loss(self, train_losses):
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Training Loss", color='tab:blue')
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ======================================================================
# 🔹 ENTRY POINT
# ======================================================================
if __name__ == "__main__":

    #USER  CUSTOM
    FILE_NAME = "RC_Tank_Preprocessing_dataset_package.zip"
    FILE_NAME_MODEL = "lstm_model.pth"
    TYPE_OF_DATASET = "lazy"      # "full" or "lazy"
    TYPE_OF_MODEL = "DeepLSTM"     # choose: "VanillaLSTM", "DeepLSTM", "BiLSTM"
    #HYPERPARAMETER
    BATCH_SIZE = 1024
    HIDDEN_LAYER = 128
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 10
    PATIENCE = 10
    CORE_CPU = 1
    FUTURE_DATA = 2

    
    #RELATIVE PATH
    ROOT_DIR = Path(__file__).resolve().parents[2]
    FILE_ZIP_DATA = ROOT_DIR / "data" / "processed" / FILE_NAME
    TEMPORALY_FOLDER = ROOT_DIR / "data" / "temporaly"
    PATH_SAVE_MODEL = ROOT_DIR / "models" / FILE_NAME_MODEL
    DATA_FOLDER = unpack_dataset_zip(FILE_ZIP_DATA,TEMPORALY_FOLDER)

    trainer = TRAIN_MODEL_PT(
        data_folder = DATA_FOLDER,
        model_save_path = PATH_SAVE_MODEL,
        dataset_type = TYPE_OF_DATASET, 
        model_type=TYPE_OF_MODEL,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_LAYER,
        num_layers=NUM_LAYERS,
        lr=LEARNING_RATE,
        num_epochs=EPOCHS,
        patience=PATIENCE,
        num_worker= CORE_CPU,
        prefetch_factor= FUTURE_DATA
    )

    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
    shutil.rmtree(TEMPORALY_FOLDER)
    print(f"[INFO] Temporary extracted dataset folder removed: {TEMPORALY_FOLDER}")