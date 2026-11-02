
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data/processed/train_encoded.csv"
TEST_PATH = PROJECT_ROOT / "data/processed/test_encoded.csv"

CHUNK_SIZE = 100_000

print("Starting complete feature validation...")

# Read column names without loading the full datasets
train_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
test_columns = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

# Check feature alignment
expected_test_columns = [
    column for column in train_columns if column != "sales"
]

print("\n1. Checking column alignment")

assert "sales" in train_columns
assert "sales" not in test_columns
assert expected_test_columns == test_columns

print("Column alignment: PASSED")

# Validate complete files in chunks
def validate_file(file_path, expected_rows, label):
    total_rows = 0
    total_missing = 0
    duplicate_ids = 0
    seen_ids = set()

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        # Count missing values in every column
        total_missing += int(chunk.isnull().sum().sum())

        # Check duplicate IDs across chunks
        if "id" in chunk.columns:
            ids = chunk["id"].tolist()

            duplicate_ids += sum(
                identifier in seen_ids
                for identifier in ids
            )

            seen_ids.update(ids)

    print(f"\n{label} validation")
    print("Total rows:", total_rows)
    print("Total missing values:", total_missing)
    print("Duplicate IDs:", duplicate_ids)

    assert total_rows == expected_rows
    assert duplicate_ids == 0

    return total_missing


print("\n2. Validating training dataset")

train_missing = validate_file(
    TRAIN_PATH,
    3_000_888,
    "TRAINING"
)

print("\n3. Validating testing dataset")

test_missing = validate_file(
    TEST_PATH,
    28_512,
    "TESTING"
)

print("\n4. Final validation summary")

print("Training missing values:", train_missing)
print("Testing missing values:", test_missing)

print("\nComplete dataset validation finished.")
 