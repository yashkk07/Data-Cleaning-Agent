"""
Test: Tool Selection Quality Analysis
Analyzes whether GPT's lower step count means missing important cleaning steps
or if Llama is being overly aggressive.
"""

import json
import logging
from typing import Dict, Any
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.extract.reader import read_structured_safe
from etl.llm.json_utils import parse_llm_json
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Import from planner
from etl.llm.planner import SYSTEM_PROMPT, build_user_prompt

# Import LLM callers from test_llm_comparison
from tests.test_llm_comparison import call_openai, call_groq


def analyze_profile_issues(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze profile to identify actual data quality issues.
    """
    issues = {
        "missing_values": [],
        "whitespace": [],
        "duplicates": False,
        "type_conversions_needed": [],
        "datetime_parsing_needed": [],
        "index_columns": [],
        "high_cardinality_useless": [],
        "boolean_candidates": [],
        "currency_percentage": [],
    }
    
    columns = profile.get("columns", {})
    
    for col_name, col_meta in columns.items():
        # Missing values
        missing_ratio = col_meta.get("missing_ratio", 0)
        if missing_ratio > 0:
            issues["missing_values"].append({
                "column": col_name,
                "ratio": missing_ratio
            })
        
        # Whitespace (inferred from semantic type)
        if col_meta.get("semantic_type") == "text":
            issues["whitespace"].append(col_name)
        
        # Type conversions
        numeric_string_ratio = col_meta.get("numeric_string_ratio", 0)
        if numeric_string_ratio > 0.8 and col_meta.get("semantic_type") != "numeric":
            issues["type_conversions_needed"].append({
                "column": col_name,
                "ratio": numeric_string_ratio
            })
        
        # Datetime parsing
        if col_meta.get("semantic_type") == "datetime":
            issues["datetime_parsing_needed"].append(col_name)
        
        # Index columns
        if col_meta.get("semantic_type") == "index":
            issues["index_columns"].append(col_name)
        
        # High cardinality useless columns
        all_unique_ratio = col_meta.get("all_unique_ratio", 0)
        if all_unique_ratio == 1.0:
            issues["high_cardinality_useless"].append(col_name)
        
        # Boolean candidates
        if col_meta.get("unique_count", 999) == 2:
            issues["boolean_candidates"].append(col_name)
    
    # Duplicates
    duplicates = profile.get("duplicates", {}).get("duplicate_count", 0)
    if duplicates > 0:
        issues["duplicates"] = True
    
    return issues


def evaluate_tool_coverage(tools_selected: list, actual_issues: Dict) -> Dict:
    """
    Evaluate how well the selected tools cover the actual issues.
    """
    coverage = {
        "missing_values": False,
        "whitespace": False,
        "duplicates": False,
        "type_conversions": False,
        "datetime_parsing": False,
        "index_columns": False,
        "useless_columns": False,
    }
    
    coverage_details = []
    
    for tool_info in tools_selected:
        tool_name = tool_info if isinstance(tool_info, str) else tool_info.get("name")
        
        if tool_name == "fillna" or tool_name == "standardize_missing":
            coverage["missing_values"] = True
            coverage_details.append("✓ Handles missing values")
        
        if tool_name == "trim_whitespace":
            coverage["whitespace"] = True
            coverage_details.append("✓ Handles whitespace")
        
        if tool_name == "remove_duplicates":
            coverage["duplicates"] = True
            coverage_details.append("✓ Removes duplicates")
        
        if tool_name == "convert_numeric":
            coverage["type_conversions"] = True
            coverage_details.append("✓ Converts numeric types")
        
        if tool_name == "parse_datetime":
            coverage["datetime_parsing"] = True
            coverage_details.append("✓ Parses datetime")
        
        if tool_name == "drop_column":
            coverage["index_columns"] = True
            coverage["useless_columns"] = True
            coverage_details.append("✓ Drops unnecessary columns")
    
    return {
        "coverage": coverage,
        "details": coverage_details,
    }


def compare_quality(dataset_path: str):
    """
    Compare quality of tool selection between GPT and Llama.
    """
    print(f"\n{'='*80}")
    print(f"📋 TOOL SELECTION QUALITY ANALYSIS")
    print(f"{'='*80}")
    print(f"\n📁 Dataset: {os.path.basename(dataset_path)}")
    
    # Load and profile
    df, _ = read_structured_safe(dataset_path)
    profile = ensure_json_serializable(profile_dataframe(df))
    
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Analyze actual issues
    print(f"\n{'─'*80}")
    print("🔍 ACTUAL DATA QUALITY ISSUES")
    print(f"{'─'*80}")
    
    actual_issues = analyze_profile_issues(profile)
    
    print(f"\n  Missing Values:")
    if actual_issues["missing_values"]:
        for item in actual_issues["missing_values"]:
            print(f"    - {item['column']}: {item['ratio']:.1%} missing")
    else:
        print(f"    ✓ No missing values")
    
    print(f"\n  Type Conversions Needed:")
    if actual_issues["type_conversions_needed"]:
        for item in actual_issues["type_conversions_needed"]:
            print(f"    - {item['column']}: {item['ratio']:.0%} numeric strings")
    else:
        print(f"    ✓ No type conversions needed")
    
    print(f"\n  Datetime Parsing Needed:")
    if actual_issues["datetime_parsing_needed"]:
        for col in actual_issues["datetime_parsing_needed"]:
            print(f"    - {col}")
    else:
        print(f"    ✓ No datetime columns")
    
    print(f"\n  Index/Useless Columns:")
    if actual_issues["index_columns"]:
        for col in actual_issues["index_columns"]:
            print(f"    - {col} (index column)")
    else:
        print(f"    ✓ No obvious index columns")
    
    if actual_issues["high_cardinality_useless"]:
        for col in actual_issues["high_cardinality_useless"]:
            print(f"    - {col} (all unique)")
    
    print(f"\n  Duplicates: {'Yes' if actual_issues['duplicates'] else 'No'}")
    
    # Get tool selections
    profile_json = json.dumps(profile, indent=2)
    user_prompt = build_user_prompt(profile_json)
    
    # GPT Selection
    print(f"\n{'─'*80}")
    print("🤖 GPT-3.5 TURBO TOOL SELECTION")
    print(f"{'─'*80}")
    
    gpt_output, gpt_time = call_openai(SYSTEM_PROMPT, user_prompt)
    gpt_plan = parse_llm_json(gpt_output)
    gpt_steps = gpt_plan.get("steps", [])
    
    print(f"\n  Steps: {len(gpt_steps)}")
    print(f"  Tools selected:")
    for i, step in enumerate(gpt_steps, 1):
        tool_name = step.get("name")
        tool_args = step.get("args", {})
        print(f"    {i}. {tool_name} {tool_args}")
    
    gpt_eval = evaluate_tool_coverage([s.get("name") for s in gpt_steps], actual_issues)
    print(f"\n  Coverage:")
    for detail in gpt_eval["details"]:
        print(f"    {detail}")
    
    # Llama Selection
    print(f"\n{'─'*80}")
    print("🤖 LLAMA 3.3 70B TOOL SELECTION")
    print(f"{'─'*80}")
    
    llama_output, llama_time = call_groq(SYSTEM_PROMPT, user_prompt)
    llama_plan = parse_llm_json(llama_output)
    llama_steps = llama_plan.get("steps", [])
    
    print(f"\n  Steps: {len(llama_steps)}")
    print(f"  Tools selected:")
    for i, step in enumerate(llama_steps, 1):
        tool_name = step.get("name")
        tool_args = step.get("args", {})
        print(f"    {i}. {tool_name} {tool_args}")
    
    llama_eval = evaluate_tool_coverage([s.get("name") for s in llama_steps], actual_issues)
    print(f"\n  Coverage:")
    for detail in llama_eval["details"]:
        print(f"    {detail}")
    
    # Quality Comparison
    print(f"\n{'='*80}")
    print("📊 QUALITY COMPARISON")
    print(f"{'='*80}")
    
    gpt_coverage = sum(gpt_eval["coverage"].values())
    llama_coverage = sum(llama_eval["coverage"].values())
    
    print(f"\n  Issue Coverage:")
    print(f"    GPT:   {gpt_coverage} issues addressed")
    print(f"    Llama: {llama_coverage} issues addressed")
    
    print(f"\n  Efficiency:")
    print(f"    GPT:   {len(gpt_steps)} steps = {gpt_coverage/max(len(gpt_steps),1):.2f} issues per step")
    print(f"    Llama: {len(llama_steps)} steps = {llama_coverage/max(len(llama_steps),1):.2f} issues per step")
    
    # Analyze differences
    gpt_tools = set(s.get("name") for s in gpt_steps)
    llama_tools = set(s.get("name") for s in llama_steps)
    
    missing_in_gpt = llama_tools - gpt_tools
    missing_in_llama = gpt_tools - llama_tools
    
    print(f"\n  Differences:")
    if missing_in_gpt:
        print(f"    GPT missing: {', '.join(missing_in_gpt)}")
        print(f"      → Is this a problem? ", end="")
        # Check if missed tools address actual issues
        if "convert_numeric" in missing_in_gpt and actual_issues["type_conversions_needed"]:
            print("⚠️  YES - GPT missed needed numeric conversions")
        elif "remove_duplicates" in missing_in_gpt and actual_issues["duplicates"]:
            print("⚠️  YES - GPT missed duplicate removal")
        else:
            print("✓ NO - These steps are optional/redundant")
    else:
        print(f"    GPT missing: None")
    
    if missing_in_llama:
        print(f"    Llama missing: {', '.join(missing_in_llama)}")
    else:
        print(f"    Llama missing: None")
    
    print(f"\n  Verdict:")
    if gpt_coverage >= llama_coverage and len(gpt_steps) <= len(llama_steps):
        print(f"    🏆 GPT is more efficient (same or better coverage with fewer steps)")
    elif llama_coverage > gpt_coverage:
        print(f"    🏆 Llama is more thorough (addresses more issues)")
    else:
        print(f"    ⚖️  Mixed - different trade-offs")
    
    print(f"\n{'='*80}\n")
    
    return {
        "actual_issues": actual_issues,
        "gpt": {"steps": len(gpt_steps), "coverage": gpt_coverage, "plan": gpt_plan},
        "llama": {"steps": len(llama_steps), "coverage": llama_coverage, "plan": llama_plan},
    }


if __name__ == "__main__":
    # Test on different datasets
    datasets = [
        "data/uploads/test_advanced.csv",
        "data/uploads/currency_percentage.csv",
        "data/uploads/index_whitespace_dupes.csv",
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            compare_quality(dataset)
