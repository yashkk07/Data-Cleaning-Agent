import logging
from typing import Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def is_tool_safe(
    df: pd.DataFrame,
    step: Dict[str, Any],
    profile: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Returns (is_safe, reason_if_not_safe)
    """

    logger.debug("Entering is_tool_safe: step=%s", step)
    tool = step.get("name")
    args = step.get("args", {})
    col = args.get("column")

    # Column existence
    if col and col not in df.columns:
        return False, f"Column '{col}' does not exist"

    if not col:
        return True, ""

    meta = profile["columns"].get(col, {})

    # ❌ Currency normalization on boolean-like columns
    if tool == "normalize_currency":
        if meta.get("boolean_string_ratio", 0) > 0.5:
            return False, "Currency symbol appears to be noise in boolean-like column"

        # Block text semantic type
        if meta.get("semantic_type") == "text":
            return False, "normalize_currency blocked: text column"

        # Block long text fields
        if meta.get("avg_string_length", 0) > 20:
            return False, "normalize_currency blocked: long text"

    # ❌ Datetime parsing without semantic evidence
    if tool == "parse_datetime":
        if meta.get("semantic_type") != "datetime":
            return False, "Datetime parsing without semantic_type=datetime"

    # ❌ Numeric conversion without numeric evidence
    if tool == "convert_numeric":
        if meta.get("numeric_string_ratio", 0) < 0.5:
            return False, "Numeric conversion without numeric_string_ratio >= 0.5"

    # ❌ fillna safety
    if tool == "fillna":
        if meta.get("semantic_type") == "index":
            return False, "fillna blocked: index column"

        if meta.get("missing_pct", 0) < 5:
            return False, "fillna blocked: missing_pct < 5%"

    # ❌ parse_boolean safety
    if tool == "parse_boolean":
        if meta.get("boolean_string_ratio", 0) < 0.8:
            return False, "parse_boolean blocked: insufficient boolean evidence"

    # ❌ normalize_text_case safety
    if tool == "normalize_text_case":
        if meta.get("semantic_type") not in {"text", "categorical"}:
            return False, "normalize_text_case blocked: non-text column"

    # ❌ standardize_categories safety
    if tool == "standardize_categories":
        if meta.get("semantic_type") != "categorical":
            return False, "standardize_categories blocked: non-categorical column"

    # ❌ scale_numeric safety
    if tool == "scale_numeric":
        if meta.get("semantic_type") != "numeric":
            return False, "scale_numeric blocked: non-numeric column"

        if meta.get("outlier_pct", 0) > 5:
            return False, "scale_numeric blocked: high outlier_pct"

    # ❌ cap_outliers safety
    if tool == "cap_outliers":
        if meta.get("semantic_type") != "numeric":
            return False, "cap_outliers blocked: non-numeric column"

    return True, ""
