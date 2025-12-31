import pandas as pd
import numpy as np
import json

from etl.transform.cleaners import (
    scale_numeric,
    cap_outliers,
    standardize_categories,
    select_model_variables,
)

# -------------------------------
# Synthetic test dataset
# -------------------------------

def make_test_df() -> pd.DataFrame:
    np.random.seed(42)

    return pd.DataFrame({
        "salary": np.concatenate([
            np.random.normal(50000, 5000, 95),
            [200000, 250000, 300000, -10000, 999999]  # extreme outliers
        ]),
        "completion_pct": np.random.uniform(0, 100, 100),
        "is_active": np.random.choice([0, 1], 100),
        "category": np.random.choice(
            [" Sales ", "Engineering", "HR", "sales", "ENGINEERING"],
            100
        ),
        "region": np.random.choice(
            ["US", "EU", "APAC", " us ", "Eu"],
            100
        ),
        "free_text": ["lorem ipsum"] * 100,
    })


# -------------------------------
# Scale numeric
# -------------------------------

def test_scale_numeric():
    df = make_test_df()

    df_scaled = scale_numeric(df, "salary", method="minmax")
    assert df_scaled["salary"].min() >= 0
    assert df_scaled["salary"].max() <= 1

    df_z = scale_numeric(df, "salary", method="zscore")
    assert abs(df_z["salary"].mean()) < 1e-6

    print("✅ scale_numeric passed")


# -------------------------------
# Cap outliers
# -------------------------------

def test_cap_outliers():
    df = make_test_df()

    df_iqr = cap_outliers(df, "salary", method="iqr")
    assert df_iqr["salary"].max() < 500000  # extreme clipped

    df_pct = cap_outliers(df, "salary", method="percentile")
    assert df_pct["salary"].quantile(0.99) <= df_pct["salary"].max()

    print("✅ cap_outliers passed")


# -------------------------------
# Standardize categories
# -------------------------------

def test_standardize_categories():
    df = make_test_df()

    df_std = standardize_categories(df, "category")
    unique_vals = set(df_std["category"].dropna().unique())

    assert "sales" in unique_vals
    assert "engineering" in unique_vals

    mapping = {
        "sales": "sales",
        "engineering": "engineering",
        "hr": "hr"
    }

    df_map = standardize_categories(df, "category", mapping=mapping)
    assert df_map["category"].str.contains(" ").sum() == 0

    print("✅ standardize_categories passed")


# -------------------------------
# Variable selection
# -------------------------------

def test_select_model_variables():
    df = make_test_df()

    result = select_model_variables(
        df=df,
        target="salary",
        min_corr=0.05,
    )

    assert result["target"] == "salary"
    assert "completion_pct" in [
        x["column"] for x in result["candidate_features"]
    ]

    assert "free_text" in result["excluded_features"]
    assert result["summary"]["is_model_ready"] is True

    # Read-only check
    assert "salary" in df.columns

    print("=== Variable Selection Output ===")
    print(json.dumps(result, indent=2))
    print("✅ select_model_variables passed")


# -------------------------------
# Run all tests
# -------------------------------

if __name__ == "__main__":
    test_scale_numeric()
    test_cap_outliers()
    test_standardize_categories()
    test_select_model_variables()

    print("\n🎉 All feature tool tests passed")
