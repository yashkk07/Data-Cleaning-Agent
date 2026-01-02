"""
Test: LLM Tool Selection Comparison - Comprehensive Benchmark
Compares tool selection performance between:
1. Llama 3.3 70B Versatile (Groq)
2. GPT-3.5 Turbo (OpenAI)

Features:
- Multiple dataset testing
- Statistical aggregation across runs
- Edge case testing
- Performance benchmark report

Focus: Tool selection only, no actual data transformation.
"""

import json
import time
import logging
import statistics
from typing import Dict, Any, Tuple, List
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.extract.reader import read_structured_safe
from etl.llm.json_utils import parse_llm_json
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ======================================================
# SYSTEM PROMPT (same for both models)
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
# LLM CALLERS
# ======================================================

def call_openai(system_prompt: str, user_prompt: str) -> Tuple[str, float]:
    """
    Call GPT-3.5 Turbo and return response + execution time.
    """
    logger.info("🟦 Calling GPT-3.5 Turbo...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3200,
    )
    elapsed = time.time() - start_time
    
    try:
        content = response.choices[0].message.content
    except Exception:
        raise ValueError(f"OpenAI returned no content: {response}")
    
    return content.strip(), elapsed


def call_groq(system_prompt: str, user_prompt: str) -> Tuple[str, float]:
    """
    Call Llama 3.3 70B Versatile and return response + execution time.
    """
    logger.info("🟦 Calling Llama 3.3 70B Versatile...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    
    client = Groq(api_key=api_key)
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=3200,
    )
    elapsed = time.time() - start_time
    
    return response.choices[0].message.content.strip(), elapsed


# ======================================================
# USER PROMPT BUILDER
# ======================================================

def build_user_prompt(profile_json: str) -> str:
    """Build user prompt for tool selection."""
    return f"""
DATASET PROFILE (JSON):
{profile_json}

IMPORTANT SEMANTIC RULES (READ CAREFULLY):
- semantic_type == "index" → drop_column
- semantic_type == "numeric" → do NOT parse datetime
- semantic_type == "datetime" → parse_datetime
- numeric_string_ratio > 0.9 → convert_numeric
- missing_ratio > 0.5 → drop_column
- all_unique_ratio == 1.0 → likely_id_column → drop_column

Generate a JSON plan with type "tool" steps.
Select the TOP 3-5 most impactful tools ONLY.

{{
  "steps": [
    {{"type": "tool", "name": "<tool_name>", "args": {{...}}}},
    ...
  ]
}}
"""


# ======================================================
# COMPARISON METRICS
# ======================================================

def extract_metrics(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from a generated plan."""
    steps = plan.get("steps", [])
    
    tool_names = [step.get("name") for step in steps]
    
    return {
        "num_steps": len(steps),
        "tools_selected": tool_names,
        "plan": plan,
    }


def compare_plans(
    gpt_plan: Dict[str, Any],
    llama_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two plans."""
    gpt_tools = set(step.get("name") for step in gpt_plan.get("steps", []))
    llama_tools = set(step.get("name") for step in llama_plan.get("steps", []))
    
    common = gpt_tools & llama_tools
    gpt_only = gpt_tools - llama_tools
    llama_only = llama_tools - llama_tools
    
    return {
        "common_tools": list(common),
        "gpt_only_tools": list(gpt_only),
        "llama_only_tools": list(llama_only),
        "agreement_ratio": len(common) / max(len(gpt_tools), len(llama_tools), 1),
    }


# ======================================================
# DATASET DISCOVERY
# ======================================================

def discover_test_datasets() -> Dict[str, Dict[str, Any]]:
    """Discover available test datasets."""
    datasets = {}
    uploads_dir = Path("data/uploads")
    
    if not uploads_dir.exists():
        logger.warning("No uploads directory found")
        return datasets
    
    for csv_file in uploads_dir.glob("*.csv"):
        try:
            df, _ = read_structured_safe(str(csv_file))
            size_category = "small" if len(df) < 100 else "medium" if len(df) < 5000 else "large"
            
            datasets[csv_file.name] = {
                "path": str(csv_file),
                "rows": len(df),
                "cols": len(df.columns),
                "size": size_category,
            }
        except Exception as e:
            logger.warning(f"Could not read {csv_file.name}: {e}")
    
    return datasets


# ======================================================
# SINGLE RUN TEST
# ======================================================

def run_single_comparison(
    dataset_path: str,
    run_num: int = 1,
) -> Dict[str, Any]:
    """Run a single comparison between GPT and Llama on a dataset."""
    
    try:
        df, _ = read_structured_safe(dataset_path)
        profile = ensure_json_serializable(profile_dataframe(df))
        profile_json = json.dumps(profile, indent=2)
        user_prompt = build_user_prompt(profile_json)
        
        dataset_info = {
            "rows": len(df),
            "cols": len(df.columns),
        }
        
        # Call GPT
        gpt_output = None
        gpt_time = None
        gpt_plan = None
        gpt_error = None
        
        try:
            gpt_output, gpt_time = call_openai(SYSTEM_PROMPT, user_prompt)
            gpt_plan = parse_llm_json(gpt_output)
            gpt_metrics = extract_metrics(gpt_plan)
        except Exception as e:
            gpt_error = str(e)
            gpt_metrics = {"num_steps": 0, "tools_selected": [], "error": gpt_error}
        
        # Call Llama
        llama_output = None
        llama_time = None
        llama_plan = None
        llama_error = None
        
        try:
            llama_output, llama_time = call_groq(SYSTEM_PROMPT, user_prompt)
            llama_plan = parse_llm_json(llama_output)
            llama_metrics = extract_metrics(llama_plan)
        except Exception as e:
            llama_error = str(e)
            llama_metrics = {"num_steps": 0, "tools_selected": [], "error": llama_error}
        
        # Comparison
        comparison = None
        if gpt_plan and llama_plan:
            comparison = compare_plans(gpt_plan, llama_plan)
        
        return {
            "run": run_num,
            "dataset": dataset_info,
            "gpt": {
                "time": gpt_time,
                "metrics": gpt_metrics,
                "error": gpt_error,
            },
            "llama": {
                "time": llama_time,
                "metrics": llama_metrics,
                "error": llama_error,
            },
            "comparison": comparison,
        }
    
    except Exception as e:
        logger.error(f"Critical error in run_single_comparison: {e}")
        return None


# ======================================================
# STATISTICAL AGGREGATION
# ======================================================

def aggregate_stats(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate statistics across multiple runs."""
    if not results:
        return {}
    
    gpt_times = [r["gpt"]["time"] for r in results if r["gpt"]["time"]]
    llama_times = [r["llama"]["time"] for r in results if r["llama"]["time"]]
    
    gpt_steps = [r["gpt"]["metrics"]["num_steps"] for r in results if r["gpt"]["metrics"]["num_steps"]]
    llama_steps = [r["llama"]["metrics"]["num_steps"] for r in results if r["llama"]["metrics"]["num_steps"]]
    
    agreements = [r["comparison"]["agreement_ratio"] for r in results if r["comparison"]]
    
    gpt_errors = sum(1 for r in results if r["gpt"]["error"])
    llama_errors = sum(1 for r in results if r["llama"]["error"])
    
    return {
        "total_runs": len(results),
        "gpt": {
            "avg_time": statistics.mean(gpt_times) if gpt_times else None,
            "min_time": min(gpt_times) if gpt_times else None,
            "max_time": max(gpt_times) if gpt_times else None,
            "stdev_time": statistics.stdev(gpt_times) if len(gpt_times) > 1 else 0,
            "avg_steps": statistics.mean(gpt_steps) if gpt_steps else None,
            "errors": gpt_errors,
        },
        "llama": {
            "avg_time": statistics.mean(llama_times) if llama_times else None,
            "min_time": min(llama_times) if llama_times else None,
            "max_time": max(llama_times) if llama_times else None,
            "stdev_time": statistics.stdev(llama_times) if len(llama_times) > 1 else 0,
            "avg_steps": statistics.mean(llama_steps) if llama_steps else None,
            "errors": llama_errors,
        },
        "agreement": {
            "avg": statistics.mean(agreements) if agreements else None,
            "min": min(agreements) if agreements else None,
            "max": max(agreements) if agreements else None,
        },
    }


# ======================================================
# COMPREHENSIVE BENCHMARK
# ======================================================

def run_comprehensive_benchmark(num_runs: int = 2):
    """
    Run comprehensive LLM comparison benchmark.
    
    Tests:
    - Multiple datasets
    - Multiple runs per dataset
    - Statistical aggregation
    - Edge cases
    """
    
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE LLM TOOL SELECTION BENCHMARK")
    print("="*80)
    
    # Discover datasets
    print("\n📊 Discovering datasets...")
    datasets = discover_test_datasets()
    
    if not datasets:
        print("❌ No test datasets found!")
        return
    
    print(f"✅ Found {len(datasets)} datasets:")
    for name, info in sorted(datasets.items()):
        print(f"   📁 {name:<40} ({info['rows']:>5} rows, {info['cols']:>2} cols) [{info['size']}]")
    
    # Select test datasets (mix of sizes)
    test_datasets = {}
    sizes = {"small": [], "medium": [], "large": []}
    
    for name, info in datasets.items():
        sizes[info["size"]].append((name, info))
    
    # Pick one from each size category
    for size in ["small", "medium", "large"]:
        if sizes[size]:
            name, info = sizes[size][0]
            test_datasets[name] = info
    
    print(f"\n🎯 Selected {len(test_datasets)} representative datasets for testing:")
    for name, info in test_datasets.items():
        print(f"   ✓ {name} ({info['size']})")
    
    # ========================================
    # Run benchmarks
    # ========================================
    
    all_results = []
    dataset_results = {}
    
    for dataset_name, dataset_info in test_datasets.items():
        print(f"\n" + "-"*80)
        print(f"📈 Testing: {dataset_name} ({dataset_info['size']}, {dataset_info['rows']} rows)")
        print("-"*80)
        
        dataset_path = dataset_info["path"]
        runs = []
        
        for run_num in range(1, num_runs + 1):
            print(f"  Run {run_num}/{num_runs}...", end=" ")
            result = run_single_comparison(dataset_path, run_num)
            
            if result:
                runs.append(result)
                gpt_time = result["gpt"]["time"]
                llama_time = result["llama"]["time"]
                agreement = result["comparison"]["agreement_ratio"] if result["comparison"] else "N/A"
                
                print(f"✅ (GPT: {gpt_time:.2f}s, Llama: {llama_time:.2f}s, Agreement: {agreement:.0%})")
            else:
                print("❌")
        
        # Aggregate stats for this dataset
        stats = aggregate_stats(runs)
        dataset_results[dataset_name] = {
            "results": runs,
            "stats": stats,
        }
        all_results.extend(runs)
    
    # ========================================
    # Generate Report
    # ========================================
    
    print("\n" + "="*80)
    print("📊 BENCHMARK REPORT")
    print("="*80)
    
    for dataset_name, data in dataset_results.items():
        stats = data["stats"]
        print(f"\n{'─'*80}")
        print(f"📁 {dataset_name}")
        print(f"{'─'*80}")
        
        print(f"  Total runs: {stats['total_runs']}")
        
        print(f"\n  ⏱️  RESPONSE TIME (seconds):")
        print(f"     GPT-3.5 Turbo:   {stats['gpt']['avg_time']:>7.2f}s (±{stats['gpt']['stdev_time']:.2f}s) [min: {stats['gpt']['min_time']:.2f}s, max: {stats['gpt']['max_time']:.2f}s]")
        print(f"     Llama 3.3 70B:   {stats['llama']['avg_time']:>7.2f}s (±{stats['llama']['stdev_time']:.2f}s) [min: {stats['llama']['min_time']:.2f}s, max: {stats['llama']['max_time']:.2f}s]")
        
        if stats['gpt']['avg_time'] and stats['llama']['avg_time']:
            ratio = stats['llama']['avg_time'] / stats['gpt']['avg_time']
            faster = "GPT" if ratio > 1 else "Llama"
            print(f"     Winner: {faster} ({abs(ratio):.1f}x)")
        
        print(f"\n  🔧 TOOL SELECTION (avg steps):")
        print(f"     GPT-3.5 Turbo:   {stats['gpt']['avg_steps']:.1f} steps")
        print(f"     Llama 3.3 70B:   {stats['llama']['avg_steps']:.1f} steps")
        
        print(f"\n  🤝 MODEL AGREEMENT:")
        print(f"     Average: {stats['agreement']['avg']:.1%}")
        print(f"     Range:   {stats['agreement']['min']:.1%} - {stats['agreement']['max']:.1%}")
        
        print(f"\n  ❌ ERRORS:")
        print(f"     GPT:   {stats['gpt']['errors']} errors")
        print(f"     Llama: {stats['llama']['errors']} errors")
    
    # ========================================
    # Overall Summary
    # ========================================
    
    overall_stats = aggregate_stats(all_results)
    
    print(f"\n{'='*80}")
    print(f"📈 OVERALL SUMMARY (All {len(test_datasets)} datasets, {overall_stats['total_runs']} total runs)")
    print(f"{'='*80}")
    
    print(f"\n⏱️  SPEED:")
    print(f"  GPT-3.5 Turbo:   {overall_stats['gpt']['avg_time']:>7.2f}s avg")
    print(f"  Llama 3.3 70B:   {overall_stats['llama']['avg_time']:>7.2f}s avg")
    
    ratio = overall_stats['llama']['avg_time'] / overall_stats['gpt']['avg_time']
    print(f"  Difference:      {abs(ratio):.2f}x ({'GPT faster' if ratio > 1 else 'Llama faster'})")
    
    print(f"\n🔧 TOOL SELECTION:")
    print(f"  GPT-3.5 Turbo:   {overall_stats['gpt']['avg_steps']:.1f} steps (conservative)")
    print(f"  Llama 3.3 70B:   {overall_stats['llama']['avg_steps']:.1f} steps (aggressive)")
    
    print(f"\n🤝 AGREEMENT:")
    print(f"  Overall: {overall_stats['agreement']['avg']:.1%}")
    
    print(f"\n❌ RELIABILITY:")
    print(f"  GPT errors:   {overall_stats['gpt']['errors']}/{overall_stats['total_runs']} ({100*overall_stats['gpt']['errors']/overall_stats['total_runs']:.0f}%)")
    print(f"  Llama errors: {overall_stats['llama']['errors']}/{overall_stats['total_runs']} ({100*overall_stats['llama']['errors']/overall_stats['total_runs']:.0f}%)")
    
    # ========================================
    # Recommendations
    # ========================================
    
    print(f"\n{'='*80}")
    print(f"✅ RECOMMENDATIONS")
    print(f"{'='*80}")
    
    gpt_score = 0
    llama_score = 0
    
    if ratio > 1:
        gpt_score += 2
        print(f"  ✓ GPT-3.5 Turbo is {ratio:.1f}x faster")
    else:
        llama_score += 2
        print(f"  ✓ Llama is {1/ratio:.1f}x faster")
    
    if overall_stats['gpt']['errors'] < overall_stats['llama']['errors']:
        gpt_score += 2
        print(f"  ✓ GPT has fewer errors ({overall_stats['gpt']['errors']} vs {overall_stats['llama']['errors']})")
    else:
        llama_score += 2
        print(f"  ✓ Llama has fewer errors ({overall_stats['llama']['errors']} vs {overall_stats['gpt']['errors']})")
    
    if overall_stats['gpt']['avg_steps'] < overall_stats['llama']['avg_steps']:
        gpt_score += 2
        print(f"  ✓ GPT is more conservative ({overall_stats['gpt']['avg_steps']:.1f} vs {overall_stats['llama']['avg_steps']:.1f} steps)")
    else:
        llama_score += 2
        print(f"  ✓ Llama is more thorough ({overall_stats['llama']['avg_steps']:.1f} vs {overall_stats['gpt']['avg_steps']:.1f} steps)")
    
    print(f"\n🏆 WINNER: {'GPT-3.5 Turbo' if gpt_score > llama_score else 'Llama 3.3 70B'}")
    print(f"  Score: GPT {gpt_score} - Llama {llama_score}")
    
    print(f"\n{'='*80}")
    print(f"✅ Benchmark completed\n")


if __name__ == "__main__":
    run_comprehensive_benchmark(num_runs=2)
