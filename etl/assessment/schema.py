from typing import TypedDict, List, Dict, Any


class ConfidenceReport(TypedDict):
    confidence_score: float
    confidence_band: str
    signals: Dict[str, float]
    penalties: Dict[str, float]
    forecast_readiness: Dict[str, bool]
    metadata: Dict[str, Any]


class ReadinessReport(TypedDict):
    is_forecast_ready: bool
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class AdvisorReport(TypedDict):
    data_quality_summary: str
    confidence_explanation: str
    dependent_variables: List[str]
    independent_variables: List[str]
    feasible_forecasts: List[str]
    improvement_suggestions: List[str]
