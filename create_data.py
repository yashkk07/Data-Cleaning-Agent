import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/test_files")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Base synthetic dataset
# -------------------------------
df = pd.DataFrame({
    "index_id": range(1, 101),
    "posted_date": pd.date_range("2023-01-01", periods=100),
    "salary": [50000 + i * 100 for i in range(100)],
    "completion_pct": [i % 100 for i in range(100)],
    "is_active": ["Yes", "No"] * 50,
    "category": ["A", "B", "C", "D"] * 25,
    "region": ["US", "EU", "APAC", "IN"] * 25,
})

# -------------------------------
# CSV
# -------------------------------
df.to_csv(DATA_DIR / "test.csv", index=False)

# -------------------------------
# Excel (multi-sheet)
# -------------------------------
with pd.ExcelWriter(DATA_DIR / "test.xlsx") as writer:
    df.to_excel(writer, sheet_name="data", index=False)
    df.head(10).to_excel(writer, sheet_name="preview", index=False)

# -------------------------------
# JSON (records)
# -------------------------------
df.to_json(DATA_DIR / "test.json", orient="records", indent=2)

# -------------------------------
# Parquet
# -------------------------------
df.to_parquet(DATA_DIR / "test.parquet")

print("✅ Synthetic test files generated")
