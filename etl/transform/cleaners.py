import logging
import pandas as pd
from typing import List, Optional, Any, Dict
import re

logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Entering clean_column_names")
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Entering standardize_missing")
    df = df.copy()

    missing_markers = {
        "": pd.NA,
        " ": pd.NA,
        "null": pd.NA,
        "n/a": pd.NA,
        "na": pd.NA,
    }

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.strip()
                .str.lower()
                .replace(missing_markers)
            )

    return df


def trim_whitespace(
    df: pd.DataFrame,
    column: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    logger.debug("Entering trim_whitespace: column=%s columns=%s", column, columns)
    df = df.copy()

    if column:
        target_cols = [column]
    elif columns:
        target_cols = columns
    else:
        target_cols = df.select_dtypes(include="object").columns

    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].where(
                df[col].isna(),
                df[col].str.strip()
            )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Entering remove_duplicates")
    return df.drop_duplicates().reset_index(drop=True)


def convert_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    logger.debug("Entering convert_numeric: column=%s", column)
    if column not in df.columns:
        return df

    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    logger.debug("Entering parse_datetime: column=%s", column)
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    logger.debug("Entering drop_column: column=%s", column)
    if column not in df.columns:
        return df
    return df.drop(columns=[column])

def normalize_currency(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    logger.debug("Entering normalize_currency: column=%s", column)
    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(r"[^\d\.]", "", regex=True)
    )

    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def normalize_percentage(df: pd.DataFrame, column: str) -> pd.DataFrame:
    logger.debug("Entering normalize_percentage: column=%s", column)
    if column not in df.columns:
        return df

    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df[column] = pd.to_numeric(df[column], errors="coerce") / 100
    return df

def fillna(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    value: Any = None
) -> pd.DataFrame:
    if column not in df.columns:
        return df

    df = df.copy()
    series = df[column]

    if strategy == "mean":
        filled = series.fillna(series.mean())
    elif strategy == "median":
        filled = series.fillna(series.median())
    elif strategy == "mode":
        if series.mode().empty:
            return df
        filled = series.fillna(series.mode()[0])
    elif strategy == "constant":
        filled = series.fillna(value)
    elif strategy == "zero":
        filled = series.fillna(0)
    elif strategy == "forward_fill":
        filled = series.ffill()
    else:
        raise ValueError(f"Unknown fillna strategy: {strategy}")

    # 🔒 Explicit dtype inference (fixes FutureWarning)
    df[column] = filled.infer_objects(copy=False)

    return df

BOOLEAN_MAP = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
    "1": True,
    "0": False,
}

def parse_boolean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    df = df.copy()
    series = df[column]

    # Skip if already boolean
    if pd.api.types.is_bool_dtype(series):
        return df

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(BOOLEAN_MAP)
    )

    # Explicit nullable boolean conversion
    df[column] = mapped.astype("boolean")

    return df

def normalize_text_case(
    df: pd.DataFrame,
    column: str,
    mode: str = "lower",
) -> pd.DataFrame:
    """
    mode: lower | upper | title
    """
    if column not in df.columns:
        return df

    df = df.copy()

    if mode == "lower":
        df[column] = df[column].str.lower()
    elif mode == "upper":
        df[column] = df[column].str.upper()
    elif mode == "title":
        df[column] = df[column].str.title()
    else:
        raise ValueError(f"Unknown text case mode: {mode}")

    return df


def scale_numeric(
    df: pd.DataFrame,
    column: str,
    method: str = "minmax",
) -> pd.DataFrame:
    """
    method: minmax | zscore
    """
    if column not in df.columns:
        return df

    df = df.copy()
    col = df[column].astype(float)

    if method == "minmax":
        min_val, max_val = col.min(), col.max()
        if min_val != max_val:
            df[column] = (col - min_val) / (max_val - min_val)
    elif method == "zscore":
        std = col.std()
        if std != 0:
            df[column] = (col - col.mean()) / std
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    return df

def cap_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    z_thresh: float = 3.0,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> pd.DataFrame:
    """
    Caps outliers in a numeric column using different methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to cap outliers in
    method : str
        Method to use: 'iqr', 'zscore', or 'percentile'
    z_thresh : float
        Z-score threshold (used if method='zscore')
    lower_pct : float
        Lower percentile (used if method='percentile')
    upper_pct : float
        Upper percentile (used if method='percentile')

    Returns
    -------
    pd.DataFrame
        DataFrame with capped outliers
    """

    if column not in df.columns:
        return df

    df = df.copy()
    col = pd.to_numeric(df[column], errors="coerce")

    if method == "iqr":
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

    elif method == "zscore":
        mean = col.mean()
        std = col.std()
        lower = mean - z_thresh * std
        upper = mean + z_thresh * std

    elif method == "percentile":
        lower = col.quantile(lower_pct)
        upper = col.quantile(upper_pct)

    else:
        raise ValueError("method must be 'iqr', 'zscore', or 'percentile'")

    df[column] = col.clip(lower, upper)
    return df

def standardize_categories(
    df: pd.DataFrame,
    column: str,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Standardizes categorical values and optionally applies category mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to standardize
    mapping : dict, optional
        Dictionary for category mapping (e.g., {'m': 'male', 'f': 'female'})

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized categorical column
    """

    if column not in df.columns:
        return df

    df = df.copy()

    # Standardize text while preserving missing values
    standardized = (
        df[column]
        .astype("string")                # preserves <NA>
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Apply category mapping if provided
    if mapping is not None:
        standardized = standardized.replace(mapping)

    df[column] = standardized
    return df


def select_model_variables(
    df: pd.DataFrame,
    target: str,
    min_corr: float = 0.1,
    max_missing_pct: float = 40.0,
) -> Dict[str, Any]:
    """
    Suggests independent variables for modeling based on correlation with a user-defined target column.

    READ-ONLY:
    - Does NOT modify the DataFrame
    - Returns metadata only

    Intended for ARIMAX-style baseline forecasting.
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    target_series = df[target]

    if not pd.api.types.is_numeric_dtype(target_series):
        raise ValueError("Target column must be numeric for correlation analysis")

    result = {
        "target": target,
        "selected_features": [],
        "candidate_features": [],
        "excluded_features": {},
        "selection_method": "pearson_correlation",
        "thresholds": {
            "min_corr": min_corr,
            "max_missing_pct": max_missing_pct,
        },
    }

    # ----------------------------------
    # Iterate through potential predictors
    # ----------------------------------
    for col in df.columns:
        if col == target:
            continue

        series = df[col]

        # Missingness check
        missing_pct = series.isna().mean() * 100
        if missing_pct > max_missing_pct:
            result["excluded_features"][col] = f"missing_pct {missing_pct:.1f}%"
            continue

        # Type checks
        if not pd.api.types.is_numeric_dtype(series):
            result["excluded_features"][col] = "non-numeric"
            continue

        # Variance check
        if series.nunique(dropna=True) <= 1:
            result["excluded_features"][col] = "zero or near-zero variance"
            continue

        # Correlation
        corr = target_series.corr(series)

        if pd.isna(corr):
            result["excluded_features"][col] = "correlation undefined"
            continue

        entry = {
            "column": col,
            "correlation": round(float(corr), 4),
            "missing_pct": round(missing_pct, 2),
        }

        result["candidate_features"].append(entry)

        if abs(corr) >= min_corr:
            result["selected_features"].append(entry)

    # ----------------------------------
    # Sort results by absolute correlation
    # ----------------------------------
    result["candidate_features"].sort(
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )

    result["selected_features"].sort(
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )

    # ----------------------------------
    # Summary flags
    # ----------------------------------
    result["summary"] = {
        "num_candidates": len(result["candidate_features"]),
        "num_selected": len(result["selected_features"]),
        "is_model_ready": len(result["selected_features"]) > 0,
    }

    return result