import pandas as pd
import chardet
from typing import Tuple, Dict


class CSVReadError(Exception):
    """Raised when CSV cannot be read safely."""
    pass


def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """
    Detect file encoding using chardet.
    """
    with open(file_path, "rb") as f:
        raw = f.read(sample_size)

    result = chardet.detect(raw)
    encoding = result.get("encoding")

    if not encoding:
        raise CSVReadError("Failed to detect file encoding")

    return encoding


def detect_delimiter(file_path: str, encoding: str) -> str:
    """
    Detect delimiter by sampling the first line.
    """
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        line = f.readline()

    delimiters = [",", ";", "\t", "|"]
    delimiter_counts = {d: line.count(d) for d in delimiters}

    detected = max(delimiter_counts, key=delimiter_counts.get)

    if delimiter_counts[detected] == 0:
        raise CSVReadError("Failed to detect delimiter")

    return detected


def read_csv_safe(
    file_path: str,
    max_bad_lines: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """
    Safely read a CSV file and return DataFrame + metadata.
    """

    metadata = {
        "file_path": file_path,
        "encoding": None,
        "delimiter": None,
        "bad_lines_skipped": 0,
        "rows_read": 0,
        "columns_read": 0,
    }

    try:
        encoding = detect_encoding(file_path)
        metadata["encoding"] = encoding

        try:
            delimiter = detect_delimiter(file_path, encoding)
        except CSVReadError:
            # Fallback: assume single-column CSV
            delimiter = ","

        metadata["delimiter"] = delimiter

        bad_lines = []

        def bad_line_handler(line):
            bad_lines.append(line)
            return None

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            engine="python",
            on_bad_lines=bad_line_handler
        )

        metadata["bad_lines_skipped"] = len(bad_lines)
        metadata["rows_read"] = len(df)
        metadata["columns_read"] = len(df.columns)

        if df.empty:
            raise CSVReadError("CSV read successfully but contains no data")

        return df, metadata

    except Exception as e:
        raise CSVReadError(f"CSV ingestion failed: {str(e)}")