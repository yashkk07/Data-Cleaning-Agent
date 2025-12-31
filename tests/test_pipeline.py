import json
from etl.pipeline import run_pipeline

# ----------------------------
# CONFIG
# ----------------------------
ADV_INPUT = "data/uploads/test_advanced.csv"
FOR_INPUT = "data/uploads/VS_data.csv"

ADV_OUTPUT = "data/outputs/cleaned_test_advanced.csv"
FOR_OUTPUT = "data/outputs/cleaned_VS_data.csv"

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
    #test_pipeline(ADV_INPUT, ADV_OUTPUT)
    test_pipeline(FOR_INPUT, FOR_OUTPUT)