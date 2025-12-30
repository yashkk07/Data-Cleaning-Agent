import logging
import os
import pandas as pd
import chardet
from typing import Any, Tuple, Dict
from pathlib import Path
from etl.extract.json_utils import detect_json_shape

logger = logging.getLogger(__name__)


class StructuredReadError(Exception):
    """Raised when a structured file cannot be read safely."""
    pass

# =====================================================
# File type detection
# =====================================================

def detect_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext in {".csv", ".tsv", ".txt"}:
        return "csv"
    if ext in {".xlsx", ".xls"}:
        return "excel"
    if ext == ".json":
        return "json"
    if ext == ".parquet":
        return "parquet"

    raise StructuredReadError(f"Unsupported file type: {ext}")

def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """
    Detect file encoding using chardet.
    """
    logger.debug("Entering detect_encoding: file_path=%s", file_path)
    with open(file_path, "rb") as f:
        raw = f.read(sample_size)

    result = chardet.detect(raw)
    encoding = result.get("encoding")

    if not encoding:
        raise StructuredReadError("Failed to detect file encoding")

    return encoding


def detect_delimiter(file_path: str, encoding: str) -> str:
    """
    Detect delimiter by sampling the first line.
    """
    logger.debug("Entering detect_delimiter: file_path=%s encoding=%s", file_path, encoding)
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        line = f.readline()

    delimiters = [",", ";", "\t", "|"]
    delimiter_counts = {d: line.count(d) for d in delimiters}

    detected = max(delimiter_counts, key=delimiter_counts.get)

    if delimiter_counts[detected] == 0:
        raise StructuredReadError("Failed to detect delimiter")

    return detected


def read_csv_safe(
    file_path: str,
    max_bad_lines: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """
    Safely read a CSV file and return DataFrame + metadata.
    """

    logger.debug("Entering read_csv_safe: file_path=%s", file_path)
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
        except StructuredReadError:
            # Fallback: assume single-column CSV
            delimiter = ","

        metadata["delimiter"] = delimiter

        bad_lines = []

        def bad_line_handler(line):
            logger.debug("Bad line encountered: %s", line)
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
            raise StructuredReadError("CSV read successfully but contains no data")

        return df, metadata

    except Exception as e:
        raise StructuredReadError(f"CSV ingestion failed: {str(e)}")
    
def read_excel_safe(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger.debug("Reading Excel: %s", file_path)

    df = pd.read_excel(file_path)

    if df.empty:
        raise StructuredReadError("Excel file is empty")

    return df, {
        "file_type": "excel",
        "rows_read": len(df),
        "columns_read": len(df.columns),
        "sheet_names": pd.ExcelFile(file_path).sheet_names,
    }


def read_json_safe(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely read JSON files with structure detection.

    Supports:
    - List of records
    - Dict of columns
    - Nested records
    - Single JSON object
    - Line-delimited JSON (fallback)
    """

    logger.debug("Reading JSON: %s", file_path)

    # ---------- First attempt: structured JSON ----------
    try:
        df = pd.read_json(file_path)
        shape = detect_json_shape(df)

    except ValueError:
        # ---------- Fallback: line-delimited JSON ----------
        try:
            df = pd.read_json(file_path, lines=True)
            shape = "json_lines"
        except Exception as e:
            raise StructuredReadError(f"JSON ingestion failed: {str(e)}")

    if df.empty:
        raise StructuredReadError("JSON file is empty or not tabular")

    return df, {
        "file_type": "json",
        "json_shape": shape,
        "rows_read": len(df),
        "columns_read": len(df.columns),
    }


def read_parquet_safe(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger.debug("Reading Parquet: %s", file_path)

    df = pd.read_parquet(file_path)

    if df.empty:
        raise StructuredReadError("Parquet file is empty")

    return df, {
        "file_type": "parquet",
        "rows_read": len(df),
        "columns_read": len(df.columns),
    }


# =====================================================
# Public unified API
# =====================================================

def read_structured_safe(
    file_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified reader for structured files.
    """

    logger.debug("Entering read_structured_safe: %s", file_path)

    # Normalize Path → str
    if isinstance(file_path, Path):
        file_path = str(file_path)

    file_type = detect_file_type(file_path)

    if file_type == "csv":
        return read_csv_safe(file_path)
    if file_type == "excel":
        return read_excel_safe(file_path)
    if file_type == "json":
        return read_json_safe(file_path)
    if file_type == "parquet":
        return read_parquet_safe(file_path)

    raise StructuredReadError(f"No reader implemented for {file_type}")