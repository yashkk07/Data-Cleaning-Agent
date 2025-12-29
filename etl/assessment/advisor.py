import json
import logging
from typing import Dict, Any, List

from etl.llm.json_utils import parse_llm_json
from etl.llm.planner import call_groq, call_openai

logger = logging.getLogger(__name__)

# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """
You are a senior data scientist specializing in data quality,
statistical confidence, and forecasting readiness.

You are given:
- Dataset profiling metadata
- A computed data quality & confidence report
- A readiness assessment

Your responsibilities:
1. Assess overall data quality
2. Explain the confidence score
3. Identify likely dependent (target) variables
4. Identify independent (feature) variables
5. Suggest feasible forecasting tasks
6. Suggest concrete ways to improve confidence

STRICT RULES:
- Output ONLY valid JSON
- No markdown
- No explanations outside JSON
- Do NOT hallucinate columns
- Base ALL reasoning strictly on provided metadata
- Be conservative and realistic
"""

# ======================================================
# USER PROMPT BUILDER
# ======================================================

def build_user_prompt(
    profile: Dict[str, Any],
    confidence: Dict[str, Any],
    readiness: Dict[str, Any],
) -> str:
    logger.debug("Entering advisor.build_user_prompt")

    return f"""
DATASET PROFILE (JSON):
{json.dumps(profile, indent=2)}

CONFIDENCE REPORT (JSON):
{json.dumps(confidence, indent=2)}

READINESS REPORT (JSON):
{json.dumps(readiness, indent=2)}

Respond in STRICT JSON with EXACT schema:
{{
  "data_quality_summary": "string",
  "confidence_explanation": "string",
  "dependent_variables": ["col1"],
  "independent_variables": ["colA", "colB"],
  "feasible_forecasts": ["description"],
  "improvement_suggestions": ["suggestion"]
}}

Rules:
- Dependent variables MUST be numeric
- Independent variables MUST exist in the dataset
- Forecasts must match dataset structure
- If forecasting is weak, say so explicitly
"""

# ======================================================
# POST-LLM VALIDATION FILTERS
# ======================================================

def _filter_dependent_variables(
    proposed: List[str],
    profile: Dict[str, Any],
) -> List[str]:
    """
    Keep only raw numeric columns as valid dependent variables.
    """
    valid_numeric = {
        col
        for col, meta in profile.get("columns", {}).items()
        if meta.get("semantic_type") == "numeric"
    }

    return [col for col in proposed if col in valid_numeric]


def _filter_independent_variables(
    proposed: List[str],
    profile: Dict[str, Any],
) -> List[str]:
    """
    Keep only columns that exist and are usable as features.
    """
    valid_columns = set(profile.get("columns", {}).keys())

    return [col for col in proposed if col in valid_columns]


# ======================================================
# PUBLIC API
# ======================================================

def generate_advice(
    profile: Dict[str, Any],
    confidence: Dict[str, Any],
    readiness: Dict[str, Any],
    llm_backend: str = "openai",
) -> Dict[str, Any]:
    """
    Generates data quality & forecasting advice.

    llm_backend:
    - "openai" → GPT-4o-mini (default, explanatory)
    - "groq"   → LLaMA 3.3 70B (optional)
    """

    logger.debug("Entering advisor.generate_advice")

    user_prompt = build_user_prompt(profile, confidence, readiness)

    if llm_backend == "groq":
        raw_output = call_groq(SYSTEM_PROMPT, user_prompt)
    else:
        raw_output = call_openai(SYSTEM_PROMPT, user_prompt)

    try:
        advice = parse_llm_json(raw_output)
    except json.JSONDecodeError:
        raise ValueError(f"Advisor returned invalid JSON:\n{raw_output}")

    # ---------------------------
    # POST-LLM SAFETY FILTERS
    # ---------------------------

    advice["dependent_variables"] = _filter_dependent_variables(
        advice.get("dependent_variables", []),
        profile,
    )

    advice["independent_variables"] = _filter_independent_variables(
        advice.get("independent_variables", []),
        profile,
    )

    return advice
