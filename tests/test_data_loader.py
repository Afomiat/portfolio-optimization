import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data_loader import fetch_data, save_processed, load_processed


def test_fetch_data_empty_tickers_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        fetch_data([], "2020-01-01", "2021-01-01")


def test_fetch_data_invalid_date_raises():
    with pytest.raises(ValueError, match="Invalid start_date"):
        fetch_data(["TSLA"], "not-a-date", "2021-01-01")


def test_save_processed_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        save_processed({})


def test_save_processed_non_dataframe_raises(tmp_path):
    with pytest.raises(ValueError, match="Expected a DataFrame"):
        save_processed({"TSLA": "not a dataframe"}, output_dir=tmp_path)


def test_save_and_load_processed_roundtrip(tmp_path):
    df = pd.DataFrame(
        {"Adj Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, name="Date")
    )
    save_processed({"TSLA": df}, output_dir=tmp_path)
    loaded = load_processed("TSLA", data_dir=tmp_path)
    assert len(loaded) == 3
    assert "Adj Close" in loaded.columns


def test_load_processed_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No processed data found"):
        load_processed("NONEXISTENT", data_dir=tmp_path)