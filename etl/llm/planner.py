import json
import logging
import os
from typing import Dict, Any, List, Optional
from etl.llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# CONFIG
# ======================================================

ALLOWED_TOOLS = [
    "clean_column_names",
    "standardize_missing",
    "trim_whitespace",
    "remove_duplicates",
    "convert_numeric",
    "parse_datetime",
    "drop_column",
    "normalize_currency",
    "normalize_percentage",
    "fillna",
    "parse_boolean",
    "normalize_text_case",
    #"scale_numeric", # would be done during modeling, not cleaning
    #"cap_outliers", # would be done during modeling, not cleaning
    #"standardize_categories", # would be done during modeling, not cleaning
]

PLAN_SCHEMA_EXAMPLE = {
    "steps": [
        {
            "type": "tool",
            "name": "trim_whitespace",
            "args": {"column": "example_column"}
        },
        {
            "type": "tool",
            "name": "fillna",
            "args": {
                "column": "numeric_column",
                "strategy": "mean"
            }
        }
    ]
}

# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """
STRICT MODE: DO NOT EXPLORE.
You are a data-cleaning planner for a production ETL system.

You are given a dataset profiler output.
The profiler is TRUSTED and FACTUAL.

CRITICAL RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No comments
- No text outside JSON

YOU MAY ONLY:
- Select tools from the allowed list
- Use column names EXACTLY as provided
- Provide COMPLETE arguments for every tool

YOU MUST:
- Base decisions ONLY on profiler metadata
- Be conservative and safe
- Prefer fewer steps over aggressive cleaning

NEVER:
- Guess column names
- Invent columns
- Repeat failed steps
- Apply datetime or numeric conversion without strong evidence
"""

# ======================================================
# USER PROMPT BUILDER
# ======================================================

def build_user_prompt(
    profile_json: str,
    feedback: Optional[Dict[str, Any]] = None
) -> str:
    logger.debug("Entering build_user_prompt")
    feedback_block = ""
    if feedback:
        feedback_block = f"""
PREVIOUS ATTEMPT FAILED.

Failure details:
{json.dumps(feedback, indent=2)}

STRICT INSTRUCTIONS:
- Do NOT repeat the failed step
- Fix missing or incorrect arguments
- Choose a safer alternative if uncertain
"""

    return f"""
DATASET PROFILE (JSON):
{profile_json}

IMPORTANT SEMANTIC RULES (READ CAREFULLY):
- semantic_type == "index" → drop_column
- semantic_type == "numeric" → do NOT parse datetime
- semantic_type == "datetime" → parse_datetime
- numeric_string_ratio > 0.9 → convert_numeric
- contains_currency_symbols == true AND semantic_type in ["numeric_like_text", "text"] AND boolean_string_ratio < 0.5 → normalize_currency
- contains_percentage_symbol == true → normalize_percentage
- semantic_type in ["text", "categorical"] → trim_whitespace
- duplicate_rows > 0 → remove_duplicates

STRICT SAFETY RULES:
- NEVER apply normalize_currency if semantic_type == "text"
- NEVER apply normalize_currency if avg_string_length > 20
- NEVER apply normalize_currency to free_text or description-like columns

STRICT FILLNA RULES:
- NEVER apply fillna to index columns
- NEVER apply fillna if missing_pct < 5%
- For numeric columns:
  - Use median if skewness > 1
  - Use mean only if skewness <= 1
- For text/categorical:
  - Use mode or constant only
- NEVER fill datetime unless forward_fill is safe
- fillna args MUST be:
  "column": "...", "strategy": "mean|median|mode|constant|zero|forward_fill"
- NEVER pass functions, lambdas, or nested objects

ADDITIONAL CLEANING HEURISTICS:

BOOLEAN HANDLING:
- boolean_string_ratio >= 0.9 → parse_boolean

TEXT NORMALIZATION:
- semantic_type in ["text", "categorical"] AND avg_string_length <= 30
  → normalize_text_case (mode="lower")

{feedback_block}

ALLOWED TOOLS:
{ALLOWED_TOOLS}

OUTPUT FORMAT EXAMPLE:
{PLAN_SCHEMA_EXAMPLE}

Generate a minimal, safe, justified cleaning plan.
"""

# '''
# CATEGORY STANDARDIZATION:
# - semantic_type == "categorical" → standardize_categories

# OUTLIER HANDLING:
# - semantic_type == "numeric" AND outlier_pct >= 10
#   → cap_outliers (method="iqr")

# NUMERIC SCALING (LAST STEP ONLY):
# - semantic_type == "numeric" AND outlier_pct < 5 AND missing_pct < 5
#   → scale_numeric (method="zscore")
#   '''

# ======================================================
# OpenAI Client
# ======================================================

def get_openai_key() -> str:
    logger.debug("Entering get_openai_key")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    return api_key


def call_openai(system_prompt: str, user_prompt: str) -> str:
    logger.debug("Entering call_openai")
    api_key = get_openai_key()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3200,
    )

    # extract content
    try:
        content = response.choices[0].message.content
    except Exception:
        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:
            raise ValueError(f"OpenAI returned no content: {response}")

    return content.strip()


# ======================================================
# GROQ CLIENT
# ======================================================

def get_groq_client() -> Groq:
    logger.debug("Entering get_groq_client")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)

def call_groq(system_prompt: str, user_prompt: str) -> str:
    logger.debug("Entering call_groq")
    client = get_groq_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=3200,
    )

    return response.choices[0].message.content.strip()

# ======================================================
# VALIDATION
# ======================================================

def validate_plan(plan: Dict[str, Any]) -> None:
    if "steps" not in plan or not isinstance(plan["steps"], list):
        raise ValueError("Plan must contain a 'steps' list")

    for i, step in enumerate(plan["steps"]):
        if step.get("type") != "tool":
            raise ValueError(
                f"Invalid step type at index {i}: {step.get('type')}"
            )

        name = step.get("name")
        args = step.get("args")

        if name not in ALLOWED_TOOLS:
            raise ValueError(f"Tool not allowed: {name}")

        if not isinstance(args, dict):
            raise ValueError("Tool args must be a dict")

# ======================================================
# PUBLIC API
# ======================================================

def generate_plan(
    profile: Dict[str, Any],
    feedback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    logger.debug("Entering generate_plan")
    profile_json = json.dumps(profile, indent=2)

    user_prompt = build_user_prompt(profile_json, feedback)
    #llm_output = call_openai(SYSTEM_PROMPT, user_prompt)
    llm_output = call_groq(SYSTEM_PROMPT, user_prompt)

    try:
        plan = parse_llm_json(llm_output)
        for step in plan.get("steps", []):
            if "type" not in step:
                step["type"] = "tool"
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON:\n{llm_output}")

    validate_plan(plan)
    return plan
