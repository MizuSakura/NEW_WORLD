# src/data/preprocess_csv2pt.py
# =============================================================================
# ⚙️ CONVERT CSV → PT DATASET WITH SCALING + SEQUENCE CREATION (with delay support)
# =============================================================================
# This script converts large CSV datasets into PyTorch tensor files (.pt),
# performing scaling (using pre-saved scalers) and sequence creation for
# supervised ML or time-series training.
# Includes optional time delay, step size, automatic metadata,
# and packaging into a single ZIP.
# =============================================================================

from src.data.scaling_loader import ScalingZipLoader
from pathlib import Path
import torch
import numpy as np
import pyarrow.csv as pv
import pyarrow.compute as pc
from tqdm import tqdm
import multiprocessing as mp
import os
from datetime import datetime
import platform
import psutil
import yaml
import zipfile
import joblib

class convert_csv2pt:
    """
    Convert large CSV datasets into .pt format for ML preprocessing.
    Includes scaling (via loaded scalers), sequence creation with delay,
    and metadata generation for reproducibility and traceability.
    Supports automatic column matching/subsetting for scalers.
    """

    def __init__(self, input_folder, output_folder, scale_path,
                 input_col, output_col, sequence_size=10, delay=1, step_size=1,
                 chunksize=100000, num_workers=4,
                 allow_padding=True, pad_value=0.0,
                 user_create=None, name_project=None,
                 time_format="%Y-%m-%d %H:%M:%S",
                 project_version="1.0.0",
                 description="Scaling reference for ML model preprocessing",
                 notes=None):
        
        # Core paths and preprocessing configs
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.scale_path = Path(scale_path)
        self.input_col = input_col
        self.output_col = output_col
        self.sequence_size = sequence_size
        self.delay = delay
        self.step_size = step_size
        self.chunksize = chunksize
        self.num_workers = num_workers
        self.allow_padding = allow_padding
        self.pad_value = pad_value

        # Metadata info
        self.user_create = user_create
        self.name_project = name_project
        self.project_version = project_version
        self.description = description
        self.notes = notes
        self.time_format = time_format

        # Load Scalers
        self.scaling_loader = ScalingZipLoader(scale_path)
        self.scaler_input = self.scaling_loader.scaler_in
        self.scaler_output = self.scaling_loader.scaler_out

        # Generate metadata on initialization
        self.metadata = self.generate_metadata()

        # Create subset scalers to match CSV columns automatically
        self.input_scaler_map = self.create_subset_scaler(self.input_col, self.scaler_input, 'input')
        self.output_scaler_map = self.create_subset_scaler(self.output_col, self.scaler_output, 'output')

        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    def create_subset_scaler(self, target_cols, scaler_obj, scaler_type='input'):
        """
        Create a new scaler object for a subset of columns in any order.
        Ensures that even if CSV columns are reordered or subsetted, the
        scaling is applied correctly.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        new_scaler = type(scaler_obj)()  # same class as original

        meta_params = self.scaling_loader.metadata['scaling']['parameters'][scaler_type]

        # Detect if MinMaxScaler or StandardScaler
        if isinstance(scaler_obj, MinMaxScaler):
            # Map min and scale by column name
            new_min = []
            new_scale = []
            for col in target_cols:
                try:
                    idx = self.scaling_loader.metadata['dataset']['input_features' if scaler_type=='input' else 'output_features'].index(col)
                except ValueError:
                    raise KeyError(f"Column '{col}' not found in scaler metadata '{scaler_type}' features")
                new_min.append(meta_params['min'][idx])
                new_scale.append(meta_params['scale'][idx])
            new_scaler.min_ = np.array(new_min)
            new_scaler.scale_ = np.array(new_scale)
            new_scaler.data_min_ = np.array([meta_params['data_min'][self.scaling_loader.metadata['dataset']['input_features' if scaler_type=='input' else 'output_features'].index(c)] for c in target_cols])
            new_scaler.data_max_ = np.array([meta_params['data_max'][self.scaling_loader.metadata['dataset']['input_features' if scaler_type=='input' else 'output_features'].index(c)] for c in target_cols])
            new_scaler.feature_range = tuple(meta_params['feature_range'])
        else:  # StandardScaler
            new_scaler.mean_ = np.array([meta_params['mean'][self.scaling_loader.metadata['dataset']['input_features' if scaler_type=='input' else 'output_features'].index(c)] for c in target_cols])
            new_scaler.scale_ = np.array([meta_params['scale'][self.scaling_loader.metadata['dataset']['input_features' if scaler_type=='input' else 'output_features'].index(c)] for c in target_cols])
            new_scaler.var_ = np.array([meta_params['var'][self.scaling_loader.metadata['dataset']['input_features' if scaler_type=='input' else 'output_features'].index(c)] for c in target_cols])
            new_scaler.n_samples_seen_ = np.array(meta_params['n_samples_seen'])

        return new_scaler

    # ----------------------------------------------------------------------
    def _read_chunk(self, file_path, skip_rows):
        """Read a chunk of CSV using PyArrow for efficient IO."""
        read_opts = pv.ReadOptions(skip_rows=skip_rows, autogenerate_column_names=False)
        parse_opts = pv.ParseOptions(delimiter=',')
        convert_opts = pv.ConvertOptions()
        table = pv.read_csv(
            file_path,
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts
            )
        table = table.slice(0, self.chunksize)
        return table.to_pandas()

    # ----------------------------------------------------------------------
    def _process_file(self, file_name):
        """Process a single CSV file: read, scale, and convert to PT."""
        file_path = self.input_folder / file_name
        total_rows = sum(1 for _ in open(file_path)) - 1  # assume header
        num_chunks = int(np.ceil(total_rows / self.chunksize))

        X_total, y_total = [], []
        for i in range(num_chunks):
            start = i * self.chunksize
            df = self._read_chunk(file_path, start)
            if df.empty:
                continue

            print(f"column name: {df.columns.tolist()}")

            # Map CSV columns to subset scalers automatically
            X_data = self.match_and_scale(df, self.input_col, self.input_scaler_map)
            y_data = self.match_and_scale(df, self.output_col, self.output_scaler_map)

            # Sequence creation with delay
            X_seq, y_seq = self.create_sequences(X_data, y_data,
                                                 delay=self.delay,
                                                 step_size=self.step_size)
            if len(X_seq) == 0:
                continue

            X_total.append(X_seq)
            y_total.append(y_seq)

        if not X_total:
            print(f"[WARNING] No sequences created for {file_name}")
            return None

        X_total = np.vstack(X_total)
        y_total = np.vstack(y_total)

        # Save as .pt file
        torch.save({'X': torch.tensor(X_total, dtype=torch.float32),
                    'y': torch.tensor(y_total, dtype=torch.float32)},
                   self.output_folder / f"{file_name.replace('.csv', '.pt')}")

        return file_name

    # ----------------------------------------------------------------------
    def match_and_scale(self, df, col_list, scaler_obj):
        """
        Extract columns from df according to col_list and apply scaler_obj.
        Automatically reshapes and handles single-column case.
        """
        X = df[col_list].to_numpy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return scaler_obj.transform(X)

    # ----------------------------------------------------------------------
    def create_sequences(self, X_data, y_data, delay=1, step_size=1):
        """Create time-series sequences with optional delay."""
        Xs, ys = [], []
        n = len(X_data)

        if n < self.sequence_size + delay:
            if self.allow_padding:
                X_pad = self.pad_or_truncate(X_data)
                y_pad = self.pad_or_truncate(y_data)
                target_idx = min(len(y_pad) - 1, self.sequence_size - 1 + delay)
                return np.array([X_pad]), np.array([y_pad[target_idx]])
            else:
                return np.array([]), np.array([])

        for i in range(0, n - self.sequence_size - delay + 1, step_size):
            X_seq = X_data[i:i + self.sequence_size]
            y_target = y_data[i + self.sequence_size - 1 + delay]
            Xs.append(X_seq)
            ys.append(y_target)

        return np.array(Xs), np.array(ys)

    # ----------------------------------------------------------------------
    def pad_or_truncate(self, seq):
        """Pad or truncate sequences to fixed length."""
        if len(seq) < self.sequence_size:
            pad_size = self.sequence_size - len(seq)
            pad = np.full((pad_size, seq.shape[1]), self.pad_value)
            seq = np.vstack((pad, seq))
        elif len(seq) > self.sequence_size:
            seq = seq[-self.sequence_size:]
        return seq

    # ----------------------------------------------------------------------
    def process_all(self):
        """Convert all CSV files in input folder to PT format."""
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        print(f"[INFO] Found {len(csv_files)} CSV files")

        if len(csv_files) == 0:
            print("[WARNING] No CSV files found. Nothing to process.")
            return

        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap(self._process_file, csv_files),
                      total=len(csv_files),
                      desc="Converting CSV → PT"))

        metadata_path = self.save_metadata()
        self.package_to_zip(metadata_path, cleanup=True)

    # ----------------------------------------------------------------------
    def generate_metadata(self):
        """Generate reproducible metadata about system, dataset, and preprocessing."""
        project_info = {
            "project_name": self.name_project,
            "created_by": self.user_create,
            "created_at": datetime.now().strftime(self.time_format),
            "version": self.project_version,
        }

        system_info = {
            "description": self.description,
            "notes": self.notes,
            "training_device": {
                "os": f"{platform.system()} {platform.release()}",
                "cpu": platform.processor(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
                "ram": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB",
                "python_version": platform.python_version()
            }
        }

        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        total_rows = 0
        for f in csv_files:
            with open(self.input_folder / f, "r", encoding="utf-8") as file:
                total_rows += sum(1 for _ in file) - 1  # assume header

        dataset_info = {
            "source_files": csv_files,
            "total_rows": total_rows,
            "input_features": self.input_col,
            "output_features": self.output_col,
            "scaling_source": str(self.scale_path)
        }

        scaling_info = {
            "reference_zip": str(self.scale_path),
            "method": type(self.scaler_input).__name__,
            "metadata_from_zip": self.scaling_loader.metadata['scaling']
        }

        preprocessing_info = {
            "sequence_size": self.sequence_size,
            "delay": self.delay,
            "step_size": self.step_size,
            "chunksize": self.chunksize,
            "allow_padding": self.allow_padding,
            "pad_value": self.pad_value,
            "num_workers": self.num_workers
        }

        data_structure = {
            "input_columns": self.input_col,
            "output_columns": self.output_col,
            "input_shape": [self.sequence_size, len(self.input_col)],
            "output_shape": [len(self.output_col)]
        }

        return {
            "project_info": project_info,
            "system_info": system_info,
            "dataset": dataset_info,
            "scaling": scaling_info,
            "preprocessing": preprocessing_info,
            "data_structure": data_structure
        }

    # ----------------------------------------------------------------------
    def save_metadata(self):
        """Save metadata as YAML."""
        metadata_path = self.output_folder / f"metadata_{self.name_project}.yaml"
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(self.metadata, f,
                      default_flow_style=False,
                      allow_unicode=True,
                      sort_keys=False)
        print(f"[INFO] Metadata saved to: {metadata_path}")
        return metadata_path

    # ----------------------------------------------------------------------
    def package_to_zip(self, metadata_path, cleanup=True):
        """Package PT files, metadata, and scalers into one ZIP."""
        zip_filename = self.output_folder / f"{self.name_project}_dataset_package.zip"
        pt_files = list(self.output_folder.glob("*.pt"))
        if not pt_files:
            print("[WARNING] No .pt files were generated to package.")
            return None

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            print(f"[INFO] Adding {len(pt_files)} .pt files to zip...")
            for pt_file in pt_files:
                zipf.write(pt_file, pt_file.name)

            if metadata_path.exists():
                zipf.write(metadata_path, metadata_path.name)

            try:
                with zipfile.ZipFile(self.scale_path, 'r') as source_zip:
                    for s_file in ['input_scaler.pkl', 'output_scaler.pkl']:
                        if s_file in source_zip.namelist():
                            zipf.writestr(s_file, source_zip.read(s_file))
                            print(f"  -> Added {s_file}")
                        else:
                            print(f"[WARNING] {s_file} not found in source zip.")
            except Exception as e:
                print(f"[ERROR] Failed to add scaler files: {e}")

        print(f"[SUCCESS] Dataset packaged to: {zip_filename}")

        if cleanup:
            for file_path in pt_files + [metadata_path]:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"[WARNING] Could not delete {file_path.name}: {e}")

        return zip_filename


# =============================================================================
# RUN EXAMPLE
# =============================================================================
if __name__ == "__main__":

    ZIP_NAME_SCALER = "Test1_scalers.zip"
    INPUT_COLUMN = ['DATA_INPUT','DATA_OUTPUT']
    OUTPUT_COULMN = ['DATA_OUTPUT']
    SEQUENCE_SIZE = 100
    DELAY = 5          
    STEP_SIZE = 1
    CHUNKSIZE = 20000
    CORE_CPU = 6

    ROOT_DIR = Path(__file__).resolve().parents[2]
    FOLDER_DATA = ROOT_DIR / "data" / "raw"
    FOLDER_DATA_PROCESSED = ROOT_DIR / "data" / "processed"
    FOLDER_SAVE_SCALE = ROOT_DIR / "config" / ZIP_NAME_SCALER

    converter = convert_csv2pt(
        input_folder=FOLDER_DATA,
        output_folder=FOLDER_DATA_PROCESSED,
        scale_path=FOLDER_SAVE_SCALE,
        input_col=INPUT_COLUMN,
        output_col=OUTPUT_COULMN,
        sequence_size=SEQUENCE_SIZE,
        delay=DELAY,
        step_size=STEP_SIZE,
        chunksize=CHUNKSIZE,
        num_workers=CORE_CPU,
        user_create="what",
        name_project="test2",
        description="Convert RC Tank raw signals into PT tensors with delay and step size",
        notes="Dataset prepared for supervised learning model training."
    )

    converter.process_all()
