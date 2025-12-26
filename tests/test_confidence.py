import json
from etl.extract.reader import read_csv_safe
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.assessment.confidence import compute_confidence

# 1. Load CSV
df, _ = read_csv_safe("data/outputs/cleaned_profiler_test.csv")

# 2. Build profile
profile = ensure_json_serializable(profile_dataframe(df))

# 3. Compute confidence
confidence = compute_confidence(df, profile)

# 4. Print nicely
print(json.dumps(confidence, indent=2))
