import pandas as pd
from pathlib import Path

# Locate the project directory
project_root = Path(__file__).resolve().parents[1]

# Locate the Kaggle dataset
data_folder = project_root / "data" / "raw" / "store-sales-kaggle"

# Load the main datasets
train = pd.read_csv(data_folder / "train.csv")
test = pd.read_csv(data_folder / "test.csv")
stores = pd.read_csv(data_folder / "stores.csv")

# Convert date columns into datetime format
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

print("=" * 55)
print("UNDERSTANDING THE STORE SALES DATASET")
print("=" * 55)

# 1. Training date range
print("\nTRAINING DATA DATE RANGE")
print("Start:", train["date"].min())
print("End:", train["date"].max())

# 2. Testing date range
print("\nTESTING DATA DATE RANGE")
print("Start:", test["date"].min())
print("End:", test["date"].max())

# 3. Number of unique stores
print("\nSTORE INFORMATION")
print("Unique stores in training:", train["store_nbr"].nunique())
print("Stores in stores.csv:", stores["store_nbr"].nunique())

# 4. Product families
print("\nPRODUCT FAMILY INFORMATION")
print("Unique product families:", train["family"].nunique())
print(train["family"].unique())

# 5. Sales statistics
print("\nSALES STATISTICS")
print(train["sales"].describe())

# 6. Promotion statistics
print("\nPROMOTION STATISTICS")
print(train["onpromotion"].describe())

# 7. Duplicate records
print("\nDUPLICATE RECORDS")
print("Training duplicates:", train.duplicated().sum())
print("Testing duplicates:", test.duplicated().sum())

print("\n" + "=" * 55)
print("DATASET UNDERSTANDING COMPLETED")
print("=" * 55)