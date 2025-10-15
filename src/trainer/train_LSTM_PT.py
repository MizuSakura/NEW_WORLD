# my_project/src/trainer/train_LSTM_PT.py
# ======================================================================
# ⚙️ TRAINING LSTM MODEL WITH .PT DATASET (FULL WORKFLOW + TEST LOGGING)
# ======================================================================

from src.data.sequence_builder import PTLazyChunkedSequenceDataset, PTSequenceDataset
from src.models.lstm_model import VanillaLSTM_MODEL, DeepLSTM_MODEL, BiLSTM_MODEL
from src.utils.logger import Logger
from torch.utils.data import DataLoader, random_split
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
import shutil
from datetime import datetime ## MODIFIED: Added for timestamping
import yaml                 ## MODIFIED: Added for reading dataset metadata

# Optional data engines
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet

# ======================================================================
# 🔹 HELPER FUNCTION TO UNPACK ZIP
# ======================================================================
def unpack_dataset_zip(zip_path, target_folder, cleanup=False):
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

torch.backends.cudnn.benchmark = True  # ⚡ GPU speedup

# ======================================================================
# 🔹 TRAINER CLASS
# ======================================================================
class TRAIN_MODEL_PT:
    def __init__(self,
                 data_folder,
                 model_save_path,
                 result_folder,
                 dataset_zip_path,     ## MODIFIED: Added to store path to original zip
                 dataset_type="lazy",
                 model_type="DeepLSTM",
                 batch_size=1024,
                 hidden_dim=128,
                 num_layers=2,
                 lr=1e-3,
                 fc_units=[64, 32],
                 num_epochs=100,
                 patience=20,
                 num_worker=0,
                 prefetch_factor=2,
                 save_engine="pandas",
                 device=None):
        self.data_folder = Path(data_folder)
        self.model_save_path = Path(model_save_path)
        self.result_folder = Path(result_folder)
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.dataset_zip_path = Path(dataset_zip_path) ## MODIFIED: Store zip path for checkpointing

        self.dataset_type = dataset_type
        self.model_type = model_type
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.fc_units = fc_units
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_worker = num_worker
        self.prefetch_factor = prefetch_factor
        self.save_engine = save_engine
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.logger = Logger()

        ## MODIFIED: Attributes for comprehensive checkpointing
        self.input_dim = None
        self.output_dim = None
        self.sequence_size = None
        self.scaler_metadata = None
        


    # ==============================================================
    # 🔸 Dataset
    # ==============================================================
    def prepare_data(self):
        if self.dataset_type == "full":
            self.dataset = PTSequenceDataset(self.data_folder)
        else:
            self.dataset = PTLazyChunkedSequenceDataset(self.data_folder)
        print(f"[INFO] Loaded dataset: {len(self.dataset)} samples")

        ## MODIFIED: Load metadata from the dataset's YAML for checkpointing
        try:
            metadata_file = next(self.data_folder.glob("metadata_*.yaml"))
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
            self.sequence_size = metadata.get('preprocessing', {}).get('sequence_size', -1)
            self.scaler_metadata = metadata.get('scaling', {})
            print(f"[INFO] Loaded metadata from {metadata_file.name}. Sequence size: {self.sequence_size}")
        except (StopIteration, FileNotFoundError):
            print("[WARNING] Could not find metadata YAML. Checkpoint will have incomplete info.")
        except Exception as e:
            print(f"[WARNING] Error loading metadata: {e}. Checkpoint will have incomplete info.")


    # ==============================================================
    # 🔸 Split into Train/Val/Test
    # ==============================================================
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        dataset_size = len(self.dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        def make_loader(ds, shuffle):
            num_workers = self.num_worker
            prefetch_factor = self.prefetch_factor if num_workers > 0 else None
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=prefetch_factor,
                persistent_workers=num_workers > 0
            )
        self.train_loader = make_loader(self.train_dataset, True)
        self.val_loader = make_loader(self.val_dataset, False)
        self.test_loader = make_loader(self.test_dataset, False)
        print(f"[INFO] Split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    # ==============================================================
    # 🔸 Build LSTM model
    # ==============================================================
    def build_model(self):
        x0, y0 = self.dataset[0]
        self.input_dim = x0.shape[-1]
        self.output_dim = y0.shape[-1] if y0.ndim > 0 else 1

        if self.model_type == "VanillaLSTM":
            self.model = VanillaLSTM_MODEL(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, fc_units=self.fc_units)
        elif self.model_type == "DeepLSTM":
            self.model = DeepLSTM_MODEL(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, fc_units=self.fc_units)
        else:
            self.model = BiLSTM_MODEL(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, fc_units=self.fc_units)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[MODEL] {self.model_type} created ({self.input_dim}->{self.output_dim}) | FC Units: {self.fc_units}")
    # ==============================================================
    # 🔸 Train
    # ==============================================================
    def train(self):
        scaler = torch.amp.GradScaler(self.device.type, enabled=(self.device.type == "cuda"))
        best_loss, patience_counter = np.inf, 0
        best_state = None
        tr_losses, val_losses = [], []

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total = 0
            for xb, yb in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
                xb, yb = xb.to(self.device).float(), yb.to(self.device).float()
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda")):
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                total += loss.item()
            avg_tr = total / len(self.train_loader)
            tr_losses.append(avg_tr)

            # Validation
            self.model.eval()
            val_total = 0
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device).float(), yb.to(self.device).float()
                    pred = self.model(xb)
                    val_total += self.loss_fn(pred, yb).item()
            avg_val = val_total / len(self.val_loader)
            val_losses.append(avg_val)
            tqdm.write(f"Epoch {epoch:03d} | Train Loss: {avg_tr:.6f} | Val Loss: {avg_val:.6f}")

            # Early stop
            if avg_val < best_loss:
                best_loss = avg_val
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[STOP] Early stopping triggered after {self.patience} epochs without improvement.")
                    break
        
        ## MODIFIED: Save the best model using the comprehensive checkpoint format
        if best_state:
            self.model.load_state_dict(best_state) # Load best model state before saving
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_type": self.model_type,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "sequence_size": self.sequence_size,
                "scaling_zip": str(self.dataset_zip_path),
                "scaler_metadata": self.scaler_metadata,
                "train_losses": tr_losses,
                "val_losses": val_losses,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "fc_units": self.fc_units,
                "epochs": epoch, # Save the actual number of epochs run
                "torch_version": torch.__version__,
                "device": str(self.device),
                "timestamp": datetime.now().isoformat()
            }
            torch.save(checkpoint, self.model_save_path)
            print(f"[SAVE] Best model checkpoint saved to → {self.model_save_path}")
        else:
            print("[WARNING] Training did not improve. No model was saved.")

        self.plot_loss(tr_losses, val_losses)


    def plot_loss(self, tr, val):
        plt.figure(figsize=(7, 4))
        plt.plot(tr, label='Train Loss')
        plt.plot(val, label='Validation Loss')
        plt.title('Model Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 🔸 Evaluate and Log Result
    # ==============================================================
    def evaluate_test(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in tqdm(self.test_loader, desc="Testing", leave=False):
                xb, yb = xb.to(self.device).float(), yb.to(self.device).float()
                pred = self.model(xb)
                y_true.append(yb.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"[TEST] MSE={mse:.6f}, R²={r2:.6f}")

        # Save results
        result_file = self.result_folder / f"test_results_{self.model_type}.parquet" if self.save_engine == "pyarrow" else self.result_folder / f"test_results_{self.model_type}.csv"

        print(f"[SAVE] Writing test results → {result_file}")
        if self.save_engine == "pyarrow":
            table = pa.table({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
            pa_parquet.write_table(table, result_file)
        else:
            df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
            df.to_csv(result_file, index=False)

        plt.figure(figsize=(10, 5))
        plt.plot(y_true[:500], label="True Values", alpha=0.8)
        plt.plot(y_pred[:500], label="Predictions", linestyle='--')
        plt.title("Test Set: True vs. Predicted Values (Sample of 500)")
        plt.legend()
        plt.grid(True)
        plt.show()

# ==============================================================
# 🔹 MAIN
# ==============================================================
if __name__ == "__main__":
    FILE_NAME = "RC_Tank_Preprocessing_dataset_package.zip"
    FILE_MODEL = "lstm_model.pth"
    SAVE_ENGINE = "pyarrow"  # or "pandas"

    ROOT = Path(__file__).resolve().parents[2]
    ZIP_PATH = ROOT / "data" / "processed" / FILE_NAME
    TEMP_DIR = ROOT / "data" / "temp_extracted"
    MODEL_PATH = ROOT / "models" / FILE_MODEL
    RESULT_DIR = ROOT / "results"

    data_dir = unpack_dataset_zip(ZIP_PATH, TEMP_DIR)

    trainer = TRAIN_MODEL_PT(
        data_folder=data_dir,
        model_save_path=MODEL_PATH,
        result_folder=RESULT_DIR,
        dataset_zip_path=ZIP_PATH, ## MODIFIED: Pass the original zip path
        dataset_type="lazy",
        model_type="DeepLSTM",
        batch_size=1024,
        hidden_dim=128,
        num_layers=2,
        lr=1e-3,
        num_epochs=10,
        patience=10,
        save_engine=SAVE_ENGINE,
        num_worker=1
    )

    trainer.prepare_data()
    trainer.split_dataset()
    trainer.build_model()
    trainer.train()
    trainer.evaluate_test()
    shutil.rmtree(TEMP_DIR)
    print(f"[CLEAN] Temporary folder removed: {TEMP_DIR}")