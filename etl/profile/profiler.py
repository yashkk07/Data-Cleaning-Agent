import logging
import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# =====================================================
# Regex patterns (HINTS only)
# =====================================================

NUMERIC_REGEX = re.compile(r"^-?\d+(\.\d+)?$")
CURRENCY_REGEX = re.compile(r"[\$\€\£\₹]")
PERCENT_REGEX = re.compile(r"%")

DATE_REGEXES = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),       # YYYY-MM-DD
    re.compile(r"\d{2}/\d{2}/\d{4}"),       # DD/MM/YYYY
    re.compile(r"\d{4}/\d{2}/\d{2}"),       # YYYY/MM/DD
    re.compile(r"\d{4}-\d{2}"),             # YYYY-MM
    re.compile(r"\d{4}-\d{2}-\d{2}T"),      # ISO datetime
]

BOOLEAN_SET = {"true", "false", "yes", "no", "y", "n", "1", "0"}


# =====================================================
# Helper functions
# =====================================================

def numeric_string_ratio(series: pd.Series) -> float:
    logger.debug("Entering numeric_string_ratio")
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0
    return float(non_null.str.match(NUMERIC_REGEX).mean())


def boolean_string_ratio(series: pd.Series) -> float:
    logger.debug("Entering boolean_string_ratio")
    non_null = series.dropna().astype(str).str.lower()
    if non_null.empty:
        return 0.0
    return float(non_null.isin(BOOLEAN_SET).mean())


def contains_currency(series: pd.Series) -> bool:
    logger.debug("Entering contains_currency")
    non_null = series.dropna().astype(str)
    return bool(non_null.str.contains(CURRENCY_REGEX).any())


def contains_percent(series: pd.Series) -> bool:
    logger.debug("Entering contains_percent")
    non_null = series.dropna().astype(str)
    return bool(non_null.str.contains(PERCENT_REGEX).any())


def datetime_string_ratio(series: pd.Series) -> float:
    logger.debug("Entering datetime_string_ratio")
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0

    matches = pd.Series(False, index=non_null.index)
    for regex in DATE_REGEXES:
        matches |= non_null.str.match(regex)

    return float(matches.mean())


def datetime_parse_ratio(series: pd.Series) -> float:
    logger.debug("Entering datetime_parse_ratio")
    non_null = series.dropna()
    if non_null.empty:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)

    return float(parsed.notna().mean())


def is_index_like(series: pd.Series, row_count: int) -> bool:
    logger.debug("Entering is_index_like")
    non_null = series.dropna()
    if len(non_null) != row_count:
        return False
    try:
        values = non_null.astype(int)
    except Exception:
        return False

    return (
        values.is_monotonic_increasing
        and values.nunique() == row_count
        and values.min() in (0, 1)
    )


def text_length_stats(series: pd.Series) -> Dict[str, float]:
    logger.debug("Entering text_length_stats")
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return {}
    lengths = non_null.str.len()
    return {
        "avg_string_length": float(lengths.mean()),
        "max_string_length": float(lengths.max()),
    }


def top_k_values(series: pd.Series, k: int = 5) -> List[str]:
    logger.debug("Entering top_k_values")
    non_null = series.dropna()
    if non_null.empty:
        return []
    return non_null.value_counts().head(k).index.astype(str).tolist()


def numeric_distribution(series: pd.Series) -> Dict[str, float]:
    logger.debug("Entering numeric_distribution")
    clean = series.dropna().astype(float)
    if clean.empty:
        return {}

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    outliers = ((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).mean()

    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "skewness": float(clean.skew()),
        "outlier_pct": float(outliers * 100),
    }


# =====================================================
# Semantic type inference (HARD RULES APPLIED)
# =====================================================

def infer_semantic_type(
    series: pd.Series,
    index_like: bool,
    dt_parse_ratio: float,
    dt_string_ratio: float
) -> str:
    logger.debug("Entering infer_semantic_type")
    non_null = series.dropna()

    if non_null.empty:
        return "empty"

    # 🔒 1. Index always wins
    if index_like:
        return "index"

    # ✅ 2. Native pandas datetime dtype → datetime (FIX)
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # 🔒 3. Numeric dtype → numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # 🔒 4. Object-based inference
    if series.dtype == object:
        if dt_parse_ratio >= 0.8 and dt_string_ratio >= 0.5:
            return "datetime"

        if numeric_string_ratio(non_null) > 0.9:
            return "numeric_like_text"

        if boolean_string_ratio(non_null) > 0.9:
            return "boolean_like_text"

        cardinality_ratio = non_null.nunique() / len(non_null)
        if cardinality_ratio < 0.05:
            return "categorical"

    return "text"



# =====================================================
# Main profiler
# =====================================================

def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    logger.debug("Entering profile_dataframe")
    profile: Dict[str, Any] = {}

    row_count = len(df)
    col_count = len(df.columns)
    total_cells = row_count * col_count
    missing_cells = int(df.isna().sum().sum())

    # ---------------- Dataset-level EDA ----------------
    profile["dataset"] = {
        "rows": row_count,
        "columns": col_count,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells_pct": float((missing_cells / total_cells) * 100) if total_cells else 0.0,
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        "row_uniqueness_ratio": float(len(df.drop_duplicates()) / row_count) if row_count else 0.0,
    }

    # ---------------- Column-level EDA ----------------
    columns_profile: Dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        missing_pct = float(series.isna().mean() * 100)
        unique_count = int(non_null.nunique())
        cardinality_ratio = float(unique_count / row_count) if row_count else 0.0

        index_like = is_index_like(series, row_count)

        # 🔒 HARD GATE: datetime logic ONLY for object, non-index columns
        dt_string_ratio = 0.0
        dt_parse_ratio = 0.0

        if series.dtype == object and not index_like:
            dt_string_ratio = datetime_string_ratio(series)
            dt_parse_ratio = datetime_parse_ratio(series)


        col_profile: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "cardinality_ratio": cardinality_ratio,
            "is_index_like": index_like,
            "datetime_string_ratio": dt_string_ratio,
            "datetime_parse_ratio": dt_parse_ratio,
        }

        # Semantic type
        col_profile["semantic_type"] = infer_semantic_type(
            series,
            index_like,
            dt_parse_ratio,
            dt_string_ratio,
        )

        # -------- Text analysis --------
        if series.dtype == object:
            col_profile.update({
                "numeric_string_ratio": numeric_string_ratio(series),
                "boolean_string_ratio": boolean_string_ratio(series),
                "contains_currency_symbols": contains_currency(series),
                "contains_percentage_symbol": contains_percent(series),
                **text_length_stats(series),
                "top_values": top_k_values(series),
            })

        # -------- Numeric analysis --------
        if pd.api.types.is_numeric_dtype(series):
            col_profile.update(numeric_distribution(series))

        columns_profile[col] = col_profile

    profile["columns"] = columns_profile
    return profile
