# ======================================================================
# ⚙️ TRAINING LSTM MODEL WITH .PT DATASET (FULL WORKFLOW + TEST LOGGING)
# ======================================================================

from src.data.sequence_builder import PTLazyChunkedSequenceDataset, PTSequenceDataset
from src.models.lstm_model import VanillaLSTM_MODEL, DeepLSTM_MODEL, BiLSTM_MODEL
from src.utils.logger import Logger
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
from datetime import datetime
import yaml

# Optional data engines
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pa_parquet
from tqdm import tqdm

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
    def __init__(
        self,
        data_folder,
        model_save_path,
        result_folder,
        dataset_zip_path,
        backup_folder=None,
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
        device=None,
    ):
        # Paths
        self.data_folder = Path(data_folder)
        self.model_save_path = Path(model_save_path)
        self.result_folder = Path(result_folder)
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_zip_path = Path(dataset_zip_path)
        self.backup_folder = Path(backup_folder) if backup_folder else self.model_save_path.parent / "backup"
        self.backup_folder.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.backup_folder / "model_backup.pth"

        # Training config
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

        # Dataset & loaders
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Model
        self.model = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Logger
        self.logger = Logger()

        # Metadata
        self.input_dim = None
        self.output_dim = None
        self.sequence_size = None
        self.scaler_metadata = None

    # ============================================================== 
    # 🔸 Dataset
    # ============================================================== 
    def prepare_data(self):
        """Load dataset and metadata."""
        if self.dataset_type == "full":
            self.dataset = PTSequenceDataset(self.data_folder)
        else:
            self.dataset = PTLazyChunkedSequenceDataset(self.data_folder)

        print(f"[INFO] Loaded dataset: {len(self.dataset)} samples")

        # Load metadata YAML
        try:
            metadata_file = next(self.data_folder.glob("metadata_*.yaml"))
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            self.sequence_size = metadata.get("preprocessing", {}).get("sequence_size", -1)
            self.scaler_metadata = metadata.get("scaling", {})
            print(f"[INFO] Loaded metadata from {metadata_file.name}. Sequence size: {self.sequence_size}")
        except StopIteration:
            print("[WARNING] Could not find metadata YAML. Checkpoint will have incomplete info.")
        except Exception as e:
            print(f"[WARNING] Error loading metadata: {e}")

    # ============================================================== 
    # 🔸 Split Dataset
    # ============================================================== 
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, continuous_test=False):
        dataset_size = len(self.dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        if continuous_test:
            indices = list(range(dataset_size))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
            self.test_dataset = Subset(self.dataset, test_indices)
        else:
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
                persistent_workers=num_workers > 0,
            )

        self.train_loader = make_loader(self.train_dataset, True)
        self.val_loader = make_loader(self.val_dataset, True)
        self.test_loader = make_loader(self.test_dataset, False)

        print(f"[INFO] Split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        if continuous_test:
            print("[INFO] Continuous test set mode enabled (no shuffling for test).")

    # ============================================================== 
    # 🔸 Build Model
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
    # 🔸 Train with backup checkpoint
    # ============================================================== 
    def train(self, backup_interval=5, resume_from=None):
        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        scaler = torch.amp.GradScaler(self.device.type, enabled=(self.device.type == "cuda"))
        best_loss, patience_counter = np.inf, 0
        best_state = None
        tr_losses, val_losses = [], []

        try:
            for epoch in range(start_epoch, self.num_epochs + 1):
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

                # Early stopping
                if avg_val < best_loss:
                    best_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"[STOP] Early stopping triggered after {self.patience} epochs without improvement.")
                        break

                # Backup checkpoint overwrite
                if epoch % backup_interval == 0:
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "train_losses": tr_losses,
                        "val_losses": val_losses
                    }, self.backup_path)
                    print(f"[BACKUP] Temporary checkpoint saved (overwrite) → {self.backup_path}")

        except Exception as e:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "train_losses": tr_losses,
                "val_losses": val_losses
            }, self.backup_path)
            print(f"[CRASH] Training interrupted! Backup saved → {self.backup_path}")
            raise e

        # Save best model at the end
        if best_state:
            self.model.load_state_dict(best_state)
            torch.save({
                "model_state_dict": best_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_type": self.model_type,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "fc_units": self.fc_units,
                "sequence_size": self.sequence_size,
                "scaler_metadata": self.scaler_metadata,
                "scaling_zip": str(self.dataset_zip_path),
                "train_losses": tr_losses,
                "val_losses": val_losses,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs": epoch,
                "torch_version": torch.__version__,
                "device": str(self.device),
                "timestamp": datetime.now().isoformat()
            }, self.model_save_path)
            print(f"[SAVE] Best model checkpoint saved → {self.model_save_path}")
        else:
            print("[WARNING] Training did not improve. No model was saved.")

        self.plot_loss(tr_losses, val_losses)

    # ============================================================== 
    # 🔸 Load checkpoint
    # ============================================================== 
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[LOAD] Checkpoint loaded from {path} at epoch {checkpoint.get('epoch', 'unknown')}")
        return checkpoint.get("epoch", 0)

    # ============================================================== 
    # 🔸 Plot Loss
    # ============================================================== 
    def plot_loss(self, tr, val):
        plt.figure(figsize=(7,4))
        plt.plot(tr, label="Train Loss")
        plt.plot(val, label="Validation Loss")
        plt.title("Model Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ============================================================== 
    # 🔸 Evaluate Continuous Test (RC-like)
    # ============================================================== 
    def evaluate_test_continuous(self, num_plot=500):
        self.model.eval()
        x_all, y_true, y_pred = [], [], []

        hidden = None
        with torch.no_grad():
            for xb, yb in tqdm(self.test_loader, desc="Testing Continuous", leave=False):
                xb, yb = xb.to(self.device).float(), yb.to(self.device).float()
                x_all.append(xb.cpu().numpy())

                if hasattr(self.model, "init_hidden") and hidden is None:
                    hidden = self.model.init_hidden(xb.size(0), device=self.device)

                if hasattr(self.model, "forward_with_hidden"):
                    pred, hidden = self.model.forward_with_hidden(xb, hidden)
                else:
                    pred = self.model(xb)

                y_true.append(yb.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        x_all = np.concatenate(x_all, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        n_features = x_all.shape[2] if x_all.ndim == 3 else 1
        x_flat = [x_all[:,:,i].flatten() if n_features>1 else x_all.flatten() for i in range(n_features)]
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        min_len = min(len(y_true_flat), len(y_pred_flat), *[len(f) for f in x_flat])
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        x_flat = [f[:min_len] for f in x_flat]

        df_dict = {f"x_{i}": x_flat[i] for i in range(n_features)}
        df_dict["y_true"] = y_true_flat
        df_dict["y_pred"] = y_pred_flat
        df = pd.DataFrame(df_dict)

        result_file = (
            self.result_folder / f"test_continuous_{self.model_type}.parquet"
            if self.save_engine=="pyarrow"
            else self.result_folder / f"test_continuous_{self.model_type}.csv"
        )
        print(f"[SAVE] Writing continuous test results → {result_file}")
        if self.save_engine=="pyarrow":
            table = pa.table(df)
            pa_parquet.write_table(table, result_file)
        else:
            df.to_csv(result_file, index=False)

        plt.figure(figsize=(12,5))
        for i in range(n_features):
            plt.plot(x_flat[i][:num_plot], label=f"Input Feature {i}", alpha=0.5)
        plt.plot(y_true_flat[:num_plot], label="True y", alpha=0.8)
        plt.plot(y_pred_flat[:num_plot], label="Predicted y", linestyle="--")
        plt.title(f"Continuous Test: Input, True vs Predicted (Sample {num_plot})")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save continuous test metadata
        meta_path = self.result_folder / f"test_continuous_{self.model_type}_metadata.pth"
        torch.save({
            "x_all": x_all,
            "y_true": y_true,
            "y_pred": y_pred,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "fc_units": self.fc_units,
            "sequence_size": self.sequence_size,
            "scaler_metadata": self.scaler_metadata,
            "scaling_zip": str(self.dataset_zip_path),
            "device": str(self.device),
            "timestamp": datetime.now().isoformat()
        }, meta_path)
        print(f"[SAVE] Continuous test metadata saved → {meta_path}")

# ============================================================== 
# 🔹 MAIN
# ============================================================== 
if __name__ == "__main__":
    FILE_NAME = "test2_dataset_package.zip"
    FILE_MODEL = "lstm_model_2_s.pth"
    SAVE_ENGINE = "pandas"

    ROOT = Path(__file__).resolve().parents[2]
    ZIP_PATH = ROOT / "data" / "processed" / FILE_NAME
    TEMP_DIR = ROOT / "data" / "temp_extracted"
    MODEL_PATH = ROOT / "models" / FILE_MODEL
    RESULT_DIR = ROOT / "models" / "results"
    BACKUP_FOLDER = MODEL_PATH.parent / "backup"  # folder แยกสำหรับ backup

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data_dir = unpack_dataset_zip(ZIP_PATH, TEMP_DIR)

        trainer = TRAIN_MODEL_PT(
            data_folder=data_dir,
            model_save_path=MODEL_PATH,
            result_folder=RESULT_DIR,
            dataset_zip_path=ZIP_PATH,
            dataset_type="lazy",
            model_type="DeepLSTM",
            batch_size=512,
            hidden_dim=256,
            num_layers=3,
            lr=1e-4,
            num_epochs=1000,
            patience=20,
            save_engine=SAVE_ENGINE,
            num_worker=1,
        )

        trainer.prepare_data()
        trainer.split_dataset(continuous_test=True)
        trainer.build_model()

        # backup checkpoint ใน folder แยก
        backup_path = BACKUP_FOLDER / "model_backup.pth"

        # Resume จาก backup ถ้ามี
        if backup_path.exists():
            print("[INFO] Resuming training from backup...")
            trainer.train(backup_interval=1, resume_from=backup_path)
        else:
            trainer.train(backup_interval=1, resume_from=None)

        trainer.evaluate_test_continuous(num_plot=500)

    finally:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f"[CLEAN] Temporary folder removed: {TEMP_DIR}")