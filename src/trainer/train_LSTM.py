# my_project\src\trainer\train_LSTM.py
from src.data.scaling_loader import ScalingZipLoader
from src.data.sequence_builder import create_sequences
from src.models.lstm_model import LSTM_MODEL
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils.logger import Logger
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TRAIN_MODEL:
    def __init__(self,
                 data_folder,
                 scaler_zip,
                 model_save_path,user_creatr,name_file,
                 window_size=30,
                 batch_size=64,
                 hidden_dim=128,
                 num_layers=2,
                 lr=1e-3,
                 num_epochs=100,
                 patience=20,
                 device=None,time_format="%Y-%m-%d %H:%M:%S",
                 project_version="1.0.0",
                 description="Scaling reference for ML model preprocessing",
                 notes=None):
        # Store paths and identifiers
        self.data_folder = Path(data_folder)
        self.scaler_zip = Path(scaler_zip)
        self.model_save_path = Path(model_save_path)
        self.user_creatr = user_creatr
        self.name_file = name_file

        # Model and training parameters
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience

        # Device configuration (CPU/GPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Metadata for experiment tracking
        self.time_format = time_format
        self.project_version = project_version
        self.description = description
        self.notes = notes

    def build_model(self):
        pass

    def prepare_data(self):
        pass

