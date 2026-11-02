import pandas as pd
from pathlib import Path

# 1. Find the project directory
project_root = Path(__file__).resolve().parents[1]

# 2. Define input and output folders
raw_folder = project_root / "data" / "raw" / "store-sales-kaggle"
processed_folder = project_root / "data" / "processed"

# Create the processed folder if it does not exist
processed_folder.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STORE SALES DATA CLEANING")
print("=" * 60)

# 3. Load the original CSV files
train = pd.read_csv(raw_folder / "train.csv")
test = pd.read_csv(raw_folder / "test.csv")
stores = pd.read_csv(raw_folder / "stores.csv")
oil = pd.read_csv(raw_folder / "oil.csv")

print("\nOriginal files loaded successfully.")

# 4. Convert date columns to datetime
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
oil["date"] = pd.to_datetime(oil["date"])

# 5. Create clean copies
train_clean = train.copy()
test_clean = test.copy()
stores_clean = stores.copy()
oil_clean = oil.copy()

# 6. Check missing values before cleaning
print("\nMissing values BEFORE cleaning:")
print(oil_clean.isnull().sum())

# 7. Sort oil prices by date
oil_clean = oil_clean.sort_values("date")

# 8. Fill missing oil prices using time interpolation
oil_clean = oil_clean.set_index("date")

oil_clean["dcoilwtico"] = (
    oil_clean["dcoilwtico"]
    .interpolate(method="time")
    .ffill()
    .bfill()
)

# Restore date as a normal column
oil_clean = oil_clean.reset_index()

# 9. Verify missing values after cleaning
print("\nMissing values AFTER cleaning:")
print("Training:", train_clean.isnull().sum().sum())
print("Testing:", test_clean.isnull().sum().sum())
print("Stores:", stores_clean.isnull().sum().sum())
print("Oil:", oil_clean.isnull().sum().sum())

# 10. Save cleaned datasets separately
train_clean.to_csv(
    processed_folder / "train_clean.csv",
    index=False
)

test_clean.to_csv(
    processed_folder / "test_clean.csv",
    index=False
)

stores_clean.to_csv(
    processed_folder / "stores_clean.csv",
    index=False
)

oil_clean.to_csv(
    processed_folder / "oil_clean.csv",
    index=False
)

print("\nCleaned files saved successfully.")

print("\nOutput folder:")
print(processed_folder)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)