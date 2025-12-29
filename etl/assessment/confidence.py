import pandas as pd
from typing import Dict, Any


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(value, max_val))


def compute_confidence(
    df: pd.DataFrame,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes data quality & forecast readiness confidence scores.

    Returns a structured, explainable confidence report.
    """

    dataset_meta = profile.get("dataset", {})
    columns_meta = profile.get("columns", {})

    rows = dataset_meta.get("rows", 0)
    missing_cells_pct = dataset_meta.get("missing_cells_pct", 100.0)
    duplicate_rows = dataset_meta.get("duplicate_rows", 0)

    # ---------------------------
    # 1. Data volume score
    # ---------------------------
    # Rule of thumb: forecasting stabilizes after ~5k–10k rows
    volume_score = clamp(rows / 10_000)

    # ---------------------------
    # 2. Completeness score
    # ---------------------------
    completeness_score = clamp(1.0 - (missing_cells_pct / 100.0))

    # ---------------------------
    # 3. Duplicate penalty
    # ---------------------------
    duplicate_penalty = 0.0
    if rows > 0:
        duplicate_penalty = clamp(duplicate_rows / rows)

    # ---------------------------
    # 4. Temporal readiness
    # ---------------------------
    datetime_columns = [
        col for col, meta in columns_meta.items()
        if meta.get("semantic_type") == "datetime"
    ]

    temporal_score = 1.0 if datetime_columns else 0.0

    # ---------------------------
    # 5. Numeric signal strength
    # ---------------------------
    numeric_columns = [
        col for col, meta in columns_meta.items()
        if meta.get("semantic_type") == "numeric"
    ]

    numeric_ratio = (
        len(numeric_columns) / len(columns_meta)
        if columns_meta else 0.0
    )

    numeric_signal_score = clamp(numeric_ratio)

    # ---------------------------
    # 6. Noise / outlier penalty
    # ---------------------------
    outlier_pcts = []
    for meta in columns_meta.values():
        if "outlier_pct" in meta:
            outlier_pcts.append(meta["outlier_pct"])

    avg_outlier_pct = (
        sum(outlier_pcts) / len(outlier_pcts)
        if outlier_pcts else 0.0
    )

    noise_penalty = clamp(avg_outlier_pct / 100.0)

    # ---------------------------
    # 7. Feature usefulness
    # ---------------------------
    useful_columns = [
        col for col, meta in columns_meta.items()
        if meta.get("semantic_type") in {
            "numeric",
            "datetime",
            "categorical",
        }
    ]

    feature_usefulness_score = (
        len(useful_columns) / len(columns_meta)
        if columns_meta else 0.0
    )

    # ---------------------------
    # 8. Weighted confidence aggregation
    # ---------------------------
    weights = {
        "volume": 0.20,
        "completeness": 0.25,
        "temporal": 0.20,
        "numeric_signal": 0.15,
        "feature_usefulness": 0.20,
    }

    raw_score = (
        volume_score * weights["volume"]
        + completeness_score * weights["completeness"]
        + temporal_score * weights["temporal"]
        + numeric_signal_score * weights["numeric_signal"]
        + feature_usefulness_score * weights["feature_usefulness"]
    )

    penalty = clamp(duplicate_penalty + noise_penalty)

    final_score = clamp(raw_score * (1.0 - penalty))

    # ---------------------------
    # 9. Forecast readiness flags
    # ---------------------------
    readiness = {
        "has_datetime": bool(datetime_columns),
        "has_numeric_targets": len(numeric_columns) > 0,
        "sufficient_rows": rows >= 500,
        "low_missingness": missing_cells_pct <= 20.0,
    }

    # ---------------------------
    # 10. Assemble report
    # ---------------------------
    return {
        "confidence_score": round(final_score, 3),
        "confidence_band": (
            "high" if final_score >= 0.70
            else "medium" if final_score >= 0.40
            else "low"
        ),
        "signals": {
            "row_count": rows,
            "volume_score": round(volume_score, 3),
            "completeness_score": round(completeness_score, 3),
            "temporal_score": temporal_score,
            "numeric_signal_score": round(numeric_signal_score, 3),
            "feature_usefulness_score": round(feature_usefulness_score, 3),
        },
        "penalties": {
            "duplicate_penalty": round(duplicate_penalty, 3),
            "noise_penalty": round(noise_penalty, 3),
        },
        "forecast_readiness": readiness,
        "metadata": {
            "datetime_columns": datetime_columns,
            "numeric_columns": numeric_columns,
        },
    }
