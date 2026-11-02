import pandas as pd
from pathlib import Path

# 1. Locate project folders
project_root = Path(__file__).resolve().parents[1]
processed_folder = project_root / "data" / "processed"

# 2. Load merged datasets
train = pd.read_csv(
    processed_folder / "train_merged.csv",
    parse_dates=["date"]
)

test = pd.read_csv(
    processed_folder / "test_merged.csv",
    parse_dates=["date"]
)

oil = pd.read_csv(
    processed_folder / "oil_clean.csv",
    parse_dates=["date"]
)

print("=" * 60)
print("FINAL MODEL DATA PREPARATION")
print("=" * 60)

# 3. Create a complete daily calendar
start_date = min(train["date"].min(), test["date"].min())
end_date = max(train["date"].max(), test["date"].max())

daily_calendar = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)

# 4. Reindex oil prices to include every calendar date
oil_daily = (
    oil[["date", "dcoilwtico"]]
    .drop_duplicates(subset=["date"])
    .set_index("date")
    .reindex(daily_calendar)
)

oil_daily.index.name = "date"

# 5. Fill missing oil prices using time interpolation
oil_daily["dcoilwtico"] = (
    oil_daily["dcoilwtico"]
    .interpolate(method="time")
    .ffill()
    .bfill()
)

oil_daily = oil_daily.reset_index()

print("\nOIL PRICE VALIDATION")
print("Daily oil records:", len(oil_daily))
print(
    "Missing oil prices:",
    oil_daily["dcoilwtico"].isnull().sum()
)

# 6. Replace old oil-price column with complete daily values
train = train.drop(columns=["dcoilwtico", "transactions"])
test = test.drop(columns=["dcoilwtico", "transactions"])

train = train.merge(
    oil_daily,
    on="date",
    how="left",
    validate="many_to_one"
)

test = test.merge(
    oil_daily,
    on="date",
    how="left",
    validate="many_to_one"
)

# 7. Check row counts
print("\nROW COUNT VALIDATION")
print("Training rows:", len(train))
print("Testing rows:", len(test))

# 8. Check missing values
print("\nTRAINING MISSING VALUES")
print(train.isnull().sum())

print("\nTESTING MISSING VALUES")
print(test.isnull().sum())

# 9. Check duplicate IDs
print("\nDUPLICATE ID VALIDATION")
print("Training:", train["id"].duplicated().sum())
print("Testing:", test["id"].duplicated().sum())

# 10. Save model-ready datasets
train.to_csv(
    processed_folder / "train_model_ready.csv",
    index=False
)

test.to_csv(
    processed_folder / "test_model_ready.csv",
    index=False
)

# Save complete daily oil-price dataset
oil_daily.to_csv(
    processed_folder / "oil_daily.csv",
    index=False
)

print("\nModel-ready datasets saved successfully.")
print("\n" + "=" * 60)
print("FINAL DATA PREPARATION COMPLETED")
print("=" * 60)