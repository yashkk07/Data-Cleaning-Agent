import logging
import pandas as pd
from typing import Dict, Any, List

from etl.transform import cleaners
from etl.executor.safety import is_tool_safe

logger = logging.getLogger(__name__)


# 🔒 Tool registry (single source of truth)
TOOL_REGISTRY = {
    "clean_column_names": cleaners.clean_column_names,
    "standardize_missing": cleaners.standardize_missing,
    "trim_whitespace": cleaners.trim_whitespace,
    "remove_duplicates": cleaners.remove_duplicates,
    "convert_numeric": cleaners.convert_numeric,
    "parse_datetime": cleaners.parse_datetime,
    "drop_column": cleaners.drop_column,
    "normalize_currency": cleaners.normalize_currency,
    "normalize_percentage": cleaners.normalize_percentage,
    "fillna": cleaners.fillna,
    "parse_boolean": cleaners.parse_boolean,
    "normalize_text_case": cleaners.normalize_text_case,
    "scale_numeric": cleaners.scale_numeric,
    "cap_outliers": cleaners.cap_outliers,
    "standardize_categories": cleaners.standardize_categories,
}


class ToolExecutionError(Exception):
    pass


def execute_tool_step(
    df: pd.DataFrame,
    step: Dict[str, Any],
    profile: Dict[str, Any],
    execution_log: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Executes a single tool step with safety checks.
    """

    logger.debug("Entering execute_tool_step")
    tool_name = step.get("name")
    args = step.get("args", {})

    # Safety check
    safe, reason = is_tool_safe(df, step, profile)
    if not safe:
        execution_log.append({
            "step": step,
            "status": "skipped",
            "reason": reason,
        })
        return df

    if tool_name not in TOOL_REGISTRY:
        raise ToolExecutionError(f"Tool not registered: {tool_name}")

    tool_fn = TOOL_REGISTRY[tool_name]

    try:
        new_df = tool_fn(df, **args)
        execution_log.append({
            "step": step,
            "status": "success",
        })
        return new_df

    except Exception as e:
        execution_log.append({
            "step": step,
            "status": "failed",
            "error": str(e),
        })
        raise ToolExecutionError(
            f"Error executing tool '{tool_name}': {e}"
        ) from e


def execute_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes all tool steps in sequence with safety checks.
    """

    logger.debug("Entering execute_plan")
    if "steps" not in plan:
        raise ToolExecutionError("Plan has no steps")

    current_df = df.copy()
    execution_log: List[Dict[str, Any]] = []

    for idx, step in enumerate(plan["steps"], start=1):
        if "type" not in step and "name" in step:
            step["type"] = "tool"
        if step.get("type") != "tool":
            raise ToolExecutionError(
                f"Step {idx}: only tool steps are supported"
            )

        current_df = execute_tool_step(
            current_df, step, profile, execution_log
        )

    return {
        "df": current_df,
        "log": execution_log,
    }
