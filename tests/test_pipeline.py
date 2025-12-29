import json
from etl.pipeline import run_pipeline

# ----------------------------
# CONFIG
# ----------------------------
SMALL_INPUT = "data/uploads/test_small.csv"
LARGE_INPUT = "data/uploads/test_large.csv"

SMALL_OUTPUT = "data/outputs/cleaned_test_small.csv"
LARGE_OUTPUT = "data/outputs/cleaned_test_large.csv"


def test_pipeline(input_path, output_path):
    print(f"\n=== Running pipeline on {input_path} ===\n")

    result = run_pipeline(
        input_csv_path=input_path,
        output_csv_path=output_path,
        max_iterations=3,
    )

    assert result["status"] == "success"

    assessment = result["assessment"]
    confidence = assessment["confidence"]
    readiness = assessment["readiness"]
    advisor = assessment["advisor"]

    print("CONFIDENCE REPORT")
    print(json.dumps(confidence, indent=2))

    print("\nREADINESS REPORT")
    print(json.dumps(readiness, indent=2))

    print("\nADVISOR REPORT")
    print(json.dumps(advisor, indent=2))

    print("\n✅ Pipeline test passed\n")


if __name__ == "__main__":
    #test_pipeline(SMALL_INPUT, SMALL_OUTPUT)
    test_pipeline(LARGE_INPUT, LARGE_OUTPUT)