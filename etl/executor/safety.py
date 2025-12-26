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

    return True, ""
