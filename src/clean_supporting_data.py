import pandas as pd
from pathlib import Path

# Find project directory
project_root = Path(__file__).resolve().parents[1]

# Define folders
raw_folder = project_root / "data" / "raw" / "store-sales-kaggle"
processed_folder = project_root / "data" / "processed"

processed_folder.mkdir(parents=True, exist_ok=True)

# Check previously cleaned files
required_files = [
    "train_clean.csv",
    "test_clean.csv",
    "stores_clean.csv",
    "oil_clean.csv"
]

for file_name in required_files:
    file_path = processed_folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing processed file: {file_name}. "
            "Complete Step 4.1 first."
        )

print("Previous cleaned files verified.")

# Load supporting datasets
holidays = pd.read_csv(raw_folder / "holidays_events.csv")
transactions = pd.read_csv(raw_folder / "transactions.csv")

# Create independent copies
holidays_clean = holidays.copy()
transactions_clean = transactions.copy()

# Convert date columns
holidays_clean["date"] = pd.to_datetime(
    holidays_clean["date"]
)

transactions_clean["date"] = pd.to_datetime(
    transactions_clean["date"]
)

# Check missing values
print("\nMissing values before cleaning:")
print("Holidays:", holidays_clean.isnull().sum().sum())
print("Transactions:", transactions_clean.isnull().sum().sum())

# Remove exact duplicate rows if any
holidays_clean = holidays_clean.drop_duplicates()
transactions_clean = transactions_clean.drop_duplicates()

# Sort by date
holidays_clean = holidays_clean.sort_values("date")
transactions_clean = transactions_clean.sort_values(
    ["date", "store_nbr"]
)

# Check duplicate transaction keys
duplicate_transaction_keys = transactions_clean.duplicated(
    subset=["date", "store_nbr"]
).sum()

print("\nDuplicate transaction date-store combinations:")
print(duplicate_transaction_keys)

# Save cleaned supporting datasets
holidays_clean.to_csv(
    processed_folder / "holidays_clean.csv",
    index=False
)

transactions_clean.to_csv(
    processed_folder / "transactions_clean.csv",
    index=False
)

# Final validation
print("\nFinal dataset information:")
print("Holiday rows:", len(holidays_clean))
print("Transaction rows:", len(transactions_clean))

print("\nHoliday missing values:",
      holidays_clean.isnull().sum().sum())

print("Transaction missing values:",
      transactions_clean.isnull().sum().sum())

print("\nSupporting datasets saved successfully.")
print("Output folder:", processed_folder)