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
import time


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
                 dataset_type="full",  # <--- เพิ่มตัวเลือก

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

        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print(f"[INFO] Dataset ready: {len(self.dataset)} samples | batch size = {self.batch_size}")

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[INFO] Model {self.model_type} built successfully on {self.device}")

    # ============================================
    # 🔹 Train Model
    # ============================================
    def train(self):
        if self.dataset is None or self.model is None:
            raise RuntimeError("Dataset or Model not prepared. Run prepare_data() and build_model() first.")

        print(f"[TRAINING] Start training {self.model_type} for {self.num_epochs} epochs ...")
        best_loss = np.inf
        best_state = None
        train_losses = []

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0

            for xb, yb in self.dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                xb = xb.float()
                yb = yb.float()

                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if hasattr(self.model, "detach_state"):
                    self.model.detach_state()

            avg_loss = total_loss / len(self.dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch [{epoch}/{self.num_epochs}] - Loss: {avg_loss:.6f}")

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
        plt.plot(train_losses, label="Training Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    trainer = TRAIN_MODEL(
        data_folder=r"D:\Project_end\New_world\my_project\data\raw",
        scaler_zip=r"D:\Project_end\New_world\my_project\config\Test_scale1_scalers.zip",
        model_save_path=r"D:\Project_end\New_world\my_project\models\lstm_model.pth",
        dataset_type="lazy",
        model_type="DeepLSTM",
        num_epochs=50,
        batch_size=500
    )

    trainer.prepare_data()
    trainer.build_model()
    trainer.train()