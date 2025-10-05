import pytest
import pandas as pd
from pathlib import Path
from src.utils.logger import Logger

@pytest.fixture
def logger(tmp_path):
    """สร้าง logger ใหม่และชี้ให้ current_path ไปยัง temp folder"""
    log = Logger()
    log.current_path = tmp_path
    return log

def test_add_data_log_basic(logger):
    logger.add_data_log(["time", "value"], [[0, 1, 2], [10, 20, 30]])
    assert not logger.df.empty
    assert list(logger.df.columns) == ["time", "value"]
    assert len(logger.df) == 3

def test_add_data_log_auto_expand_columns(logger):
    logger.add_data_log(["a"], [[1, 2, 3]])
    logger.add_data_log(["b"], [[4, 5, 6]])
    assert "a" in logger.df.columns
    assert "b" in logger.df.columns
    assert len(logger.df) == 6  # ถูก append ต่อกัน

def test_add_data_log_padding(logger):
    logger.add_data_log(["x", "y"], [[1, 2], [10]])
    assert len(logger.df) == 2
    assert pd.isna(logger.df.loc[1, "y"])  # padding None

def test_save_and_load_csv(logger):
    logger.add_data_log(["t", "v"], [[0, 1, 2], [5, 6, 7]])
    logger.save_to_csv("test_data", folder_name="test_folder")
    path = Path(logger.current_path / "test_folder" / "test_data.csv")
    assert path.exists()

    new_logger = Logger()
    new_logger.load_csv(path)
    assert not new_logger.df.empty
    assert (new_logger.df["v"] == [5, 6, 7]).all()

def test_append_from_csv_folder(logger, tmp_path):
    # สร้าง CSV 2 ไฟล์
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    folder = tmp_path / "csvs"
    folder.mkdir()
    df1.to_csv(folder / "file1.csv", index=False)
    df2.to_csv(folder / "file2.csv", index=False)

    logger.append_from_csv_folder(folder)
    assert len(logger.df) == 4
    assert set(logger.df.columns) == {"a", "b"}

def test_check_column_condition_true(logger):
    logger.add_data_log(["flag"], [[True, False, True, True]])
    result = logger.check_column_condition("flag", target_value=True, min_count=3)
    assert result == True

def test_check_column_condition_false(logger):
    logger.add_data_log(["flag"], [[True, False, False]])
    result = logger.check_column_condition("flag", target_value=True, min_count=3)
    assert result == False

def test_check_column_condition_with_slice(logger):
    logger.add_data_log(["flag"], [[True, False, True, False, True]])
    result = logger.check_column_condition("flag", target_value=True, min_count=2, start=0, end=3)
    assert result == True

def test_result_column(logger):
    logger.add_data_log(["x"], [[1, 2, 3]])
    result = logger.result_column("x")
    assert (result == [1, 2, 3]).all()
    assert logger.result_column("not_exist") is None
