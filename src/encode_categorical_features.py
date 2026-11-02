
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_lag_features.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_lag_features.csv"

TRAIN_OUTPUT = PROJECT_ROOT / "data" / "processed" / "train_encoded.csv"
TEST_OUTPUT = PROJECT_ROOT / "data" / "processed" / "test_encoded.csv"

# Columns that need categorical encoding
CATEGORICAL_COLUMNS = ["family", "city", "state", "type"]


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

print("Loading training and testing datasets...")

category_dtypes = {
    column: "category"
    for column in CATEGORICAL_COLUMNS
}

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype=category_dtypes
)

test_df = pd.read_csv(
    TEST_PATH,
    dtype=category_dtypes
)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# --------------------------------------------------
# 3. Check category consistency
# --------------------------------------------------

print("\nChecking categorical values...")

for column in CATEGORICAL_COLUMNS:
    train_categories = set(train_df[column].dropna().unique())
    test_categories = set(test_df[column].dropna().unique())

    unknown_categories = test_categories - train_categories

    if unknown_categories:
        raise ValueError(
            f"Unknown categories in {column}: {unknown_categories}"
        )

    print(f"{column}: {len(train_categories)} categories")


# --------------------------------------------------
# 4. Apply one-hot encoding
# --------------------------------------------------

print("\nApplying one-hot encoding...")

train_encoded = pd.get_dummies(
    train_df,
    columns=CATEGORICAL_COLUMNS,
    dtype="uint8"
)

test_encoded = pd.get_dummies(
    test_df,
    columns=CATEGORICAL_COLUMNS,
    dtype="uint8"
)


# --------------------------------------------------
# 5. Align training and testing columns
# --------------------------------------------------

print("\nAligning training and testing columns...")

test_encoded = test_encoded.reindex(
    columns=train_encoded.columns,
    fill_value=0
)

# Reindexing adds a sales column to test because
# sales exists in training but not in testing.
# Remove it because actual test sales are unknown.

test_encoded = test_encoded.drop(columns=["sales"], errors="ignore")


# --------------------------------------------------
# 6. Validate encoded datasets
# --------------------------------------------------

print("\nValidating results...")

expected_encoded_columns = sum(
    train_df[column].nunique()
    for column in CATEGORICAL_COLUMNS
)

print("Expected categorical columns:", expected_encoded_columns)

actual_encoded_columns = [
    column for column in train_encoded.columns
    if any(column.startswith(f"{category}_") 
           for category in CATEGORICAL_COLUMNS)
]

print("Generated categorical columns:", len(actual_encoded_columns))

assert len(actual_encoded_columns) == expected_encoded_columns

assert "sales" in train_encoded.columns
assert "sales" not in test_encoded.columns

assert not any(
    column in train_encoded.columns
    for column in CATEGORICAL_COLUMNS
)

assert list(test_encoded.columns) == [
    column for column in train_encoded.columns if column != "sales"
]

print("All validation checks passed!")


# --------------------------------------------------
# 7. Save encoded datasets
# --------------------------------------------------

print("\nSaving encoded datasets...")

train_encoded.to_csv(TRAIN_OUTPUT, index=False)
test_encoded.to_csv(TEST_OUTPUT, index=False)

print("Training dataset saved:", TRAIN_OUTPUT)
print("Testing dataset saved:", TEST_OUTPUT)

print("\nFinal training shape:", train_encoded.shape)
print("Final testing shape:", test_encoded.shape)

print("\nCategorical encoding completed successfully!")
