from typing import Dict, Any, List
from etl.assessment.schema import ConfidenceReport, ReadinessReport


def assess_readiness(
    confidence: ConfidenceReport,
    profile: Dict[str, Any],
) -> ReadinessReport:
    """
    Determines if dataset is suitable for forecasting.
    Deterministic, explainable, ETL-safe.
    """

    blocking: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []

    dataset = profile.get("dataset", {})
    columns = profile.get("columns", {})

    rows = dataset.get("rows", 0)
    missing_pct = dataset.get("missing_cells_pct", 100.0)

    has_datetime = confidence["forecast_readiness"]["has_datetime"]
    has_numeric = confidence["forecast_readiness"]["has_numeric_targets"]

    # ---------------------------
    # Hard blockers
    # ---------------------------
    if not has_datetime:
        blocking.append("No datetime column detected")

    if not has_numeric:
        blocking.append("No numeric target variables detected")

    if rows < 200:
        blocking.append("Insufficient rows for forecasting (< 200)")

    # ---------------------------
    # Warnings
    # ---------------------------
    if rows < 1000:
        warnings.append("Low row count may reduce forecast stability")

    if missing_pct > 20:
        warnings.append("High missing value percentage")

    if confidence["confidence_score"] < 0.5:
        warnings.append("Overall confidence score is low")

    # ---------------------------
    # Recommendations
    # ---------------------------
    if not has_datetime:
        recommendations.append("Add a time or date column")

    if rows < 5000:
        recommendations.append("Collect more historical data")

    if missing_pct > 10:
        recommendations.append("Improve data completeness")

    if not blocking:
        recommendations.append("Dataset is suitable for baseline forecasting")

    return {
        "is_forecast_ready": len(blocking) == 0,
        "blocking_issues": blocking,
        "warnings": warnings,
        "recommendations": recommendations,
    }
