# tests/test_scaling_reference.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.scale_referance import ScalingReference

@pytest.fixture
def sample_csv(tmp_path):
    """สร้างไฟล์ CSV จำลองสำหรับการทดสอบ"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "RC_Tank_Env_Training.csv"
    df = pd.DataFrame({
        "PWM_duty": np.linspace(0, 1, 5),
        "Prev_output": np.linspace(1, 2, 5),
        "Tank_level": np.linspace(2, 3, 5)
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_init_and_missing_file(tmp_path):
    """ทดสอบว่า init ตรวจจับไฟล์หายได้"""
    with pytest.raises(FileNotFoundError):
        ScalingReference(data_dir=tmp_path, dataset_name="not_exist")

    csv_path = tmp_path / "data"
    csv_path.mkdir()
    (csv_path / "RC_Tank_Env_Training.csv").write_text("PWM_duty,Prev_output,Tank_level\n1,2,3")

    ref = ScalingReference(data_dir=tmp_path, dataset_name="RC_Tank_Env_Training")
    assert ref.csv_file.exists()
    assert isinstance(ref.df, pd.DataFrame)


def test_fit_scalers(tmp_path, sample_csv):
    """ทดสอบว่า fit_scalers คำนวณ min/max ได้"""
    ref = ScalingReference(data_dir=sample_csv.parent, dataset_name="RC_Tank_Env_Training")
    metadata = ref.fit_scalers()

    assert "input_min" in metadata
    assert len(metadata["input_min"]) == 2
    assert "output_min" in metadata
    assert len(metadata["output_min"]) == 1


def test_save_creates_files(tmp_path, sample_csv):
    """ทดสอบว่า save() สร้างไฟล์ทั้งหมดได้"""
    ref = ScalingReference(data_dir=sample_csv.parent, dataset_name="RC_Tank_Env_Training")
    ref.fit_scalers()
    paths = ref.save()

    for p in paths.values():
        assert Path(p).exists()


def test_run_combines_fit_and_save(tmp_path, sample_csv):
    """ทดสอบว่า run() ทำงานทั้ง fit และ save"""
    ref = ScalingReference(data_dir=sample_csv.parent, dataset_name="RC_Tank_Env_Training")
    results = ref.run()

    assert all(Path(p).exists() for p in results.values())
