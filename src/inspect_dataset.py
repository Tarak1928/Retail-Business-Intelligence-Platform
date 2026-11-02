
import pandas as pd
from pathlib import Path

# Find the project directory
project_root = Path(__file__).resolve().parents[1]

# Kaggle dataset directory
data_folder = project_root / "data" / "raw" / "store-sales-kaggle"

# Expected CSV files
required_files = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
    "sample_submission.csv"
]

print("=" * 60)
print("KAGGLE STORE SALES DATASET INSPECTION")
print("=" * 60)

# Check whether the dataset directory exists
if not data_folder.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {data_folder}"
    )

# Inspect every CSV file
for file_name in required_files:

    file_path = data_folder / file_name

    print(f"\n{'-' * 60}")
    print("FILE:", file_name)

    # Check file existence
    if not file_path.exists():
        print("STATUS: FILE NOT FOUND")
        continue

    # Load CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Display dataset information
    print("STATUS: LOADED SUCCESSFULLY")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 3 Records:")
    print(df.head(3))

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)
