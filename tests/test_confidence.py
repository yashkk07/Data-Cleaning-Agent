import json
from etl.extract.reader import read_csv_safe
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.assessment.confidence import compute_confidence
from etl.assessment.readiness import assess_readiness
from etl.assessment.advisor import generate_advice

# 1. Load CSV
df, _ = read_csv_safe("data/outputs/cleaned_profiler_test.csv")

# 2. Build profile
profile = ensure_json_serializable(profile_dataframe(df))

# 3. Compute confidence
confidence = compute_confidence(df, profile)

# 4. Print nicely
print(json.dumps(confidence, indent=2))

readiness = assess_readiness(confidence, profile)

print(readiness)

advisor = generate_advice(profile, confidence, readiness)

print(json.dumps(advisor, indent=2))

required_keys = {
    "data_quality_summary",
    "confidence_explanation",
    "dependent_variables",
    "independent_variables",
    "feasible_forecasts",
    "improvement_suggestions",
}

missing = required_keys - set(advisor.keys())
assert not missing, f"Missing keys in advisor output: {missing}"

# 7. Pretty print output
print("=== Advisor Output ===")
print(json.dumps(advisor, indent=2))

print("\n✅ Advisor test passed\n")
