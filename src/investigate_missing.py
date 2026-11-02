import pandas as pd
from pathlib import Path

# Locate project folder
project_root = Path(__file__).resolve().parents[1]
processed_folder = project_root / "data" / "processed"

# Load merged datasets
train = pd.read_csv(
    processed_folder / "train_merged.csv",
    parse_dates=["date"]
)

test = pd.read_csv(
    processed_folder / "test_merged.csv",
    parse_dates=["date"]
)

print("=" * 60)
print("MISSING VALUE INVESTIGATION")
print("=" * 60)

# 1. Check missing values by column
print("\nTRAINING MISSING VALUES")
print(train.isnull().sum())

print("\nTESTING MISSING VALUES")
print(test.isnull().sum())

# 2. Check date ranges for missing oil prices
print("\nOIL PRICE INVESTIGATION")

print("Training date range:")
print(train["date"].min(), "to", train["date"].max())

print("Dates with missing oil prices:")
print(
    train.loc[
        train["dcoilwtico"].isnull(), "date"
    ].min(),
    "to",
    train.loc[
        train["dcoilwtico"].isnull(), "date"
    ].max()
)

# 3. Check transaction availability
print("\nTRANSACTION INVESTIGATION")

print("Training rows with transactions:",
      train["transactions"].notnull().sum())

print("Training rows without transactions:",
      train["transactions"].isnull().sum())

# 4. Check missing transactions by date
missing_transactions_by_date = (
    train.groupby("date")["transactions"]
    .apply(lambda values: values.isnull().sum())
)

print("\nFirst 10 dates with missing transactions:")
print(missing_transactions_by_date[
    missing_transactions_by_date > 0
].head(10))

# 5. Check whether missing transaction values vary by store
missing_transactions_by_store = (
    train.groupby("store_nbr")["transactions"]
    .apply(lambda values: values.isnull().sum())
)

print("\nMissing transactions by store:")
print(missing_transactions_by_store)

# 6. Check test period transaction availability
print("\nTEST PERIOD TRANSACTIONS")
print("Total test rows:", len(test))
print("Missing test transactions:",
      test["transactions"].isnull().sum())

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETED")
print("=" * 60)