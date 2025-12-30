import json
import pandas as pd
from pathlib import Path

from etl.extract.reader import read_structured_safe

DATA_DIR = Path("data/test_files")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_reader_test(file_path: str):
    print(f"\n--- Testing reader on {file_path} ---")
    df, meta = read_structured_safe(file_path)

    print("Metadata:")
    print(json.dumps(meta, indent=2, default=str))
    print("DataFrame head:")
    print(df.head())

    assert not df.empty
    assert meta["rows_read"] > 0
    assert meta["columns_read"] > 0

    print("✅ Passed")


def test_csv():
    run_reader_test(DATA_DIR / "test.csv")


def test_excel():
    run_reader_test(DATA_DIR / "test.xlsx")


def test_json():
    run_reader_test(DATA_DIR / "test.json")


def test_json_nested():
    run_reader_test(DATA_DIR / "test_nested.json")


def test_parquet():
    run_reader_test(DATA_DIR / "test.parquet")


if __name__ == "__main__":
    test_csv()
    test_excel()
    test_json()
    test_json_nested()
    test_parquet()
    print("\n🎉 All reader tests passed")
