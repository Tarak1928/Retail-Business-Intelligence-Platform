import pandas as pd
from pathlib import Path

# 1. Locate project folders
project_root = Path(__file__).resolve().parents[1]

processed_folder = project_root / "data" / "processed"

# 2. Load cleaned datasets
train = pd.read_csv(
    processed_folder / "train_clean.csv",
    parse_dates=["date"]
)

test = pd.read_csv(
    processed_folder / "test_clean.csv",
    parse_dates=["date"]
)

stores = pd.read_csv(
    processed_folder / "stores_clean.csv"
)

oil = pd.read_csv(
    processed_folder / "oil_clean.csv",
    parse_dates=["date"]
)

transactions = pd.read_csv(
    processed_folder / "transactions_clean.csv",
    parse_dates=["date"]
)

holidays = pd.read_csv(
    processed_folder / "holidays_clean.csv",
    parse_dates=["date"]
)

print("All cleaned datasets loaded.")

# 3. Summarize holiday events by date
holiday_summary = (
    holidays.groupby("date")
    .agg(
        holiday_event_count=("description", "count")
    )
    .reset_index()
)

holiday_summary["has_holiday_event"] = (
    holiday_summary["holiday_event_count"] > 0
)

# 4. Define a function to enrich sales data
def enrich_sales_data(sales_data):

    # Add store information
    result = sales_data.merge(
        stores,
        on="store_nbr",
        how="left",
        validate="many_to_one"
    )

    # Add oil prices
    result = result.merge(
        oil[["date", "dcoilwtico"]],
        on="date",
        how="left",
        validate="many_to_one"
    )

    # Add transaction information
    result = result.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one"
    )

    # Add daily holiday summary
    result = result.merge(
        holiday_summary,
        on="date",
        how="left",
        validate="many_to_one"
    )

    # Dates without recorded holiday events
    result["holiday_event_count"] = (
        result["holiday_event_count"].fillna(0)
    )

    result["has_holiday_event"] = (
        result["has_holiday_event"].fillna(False)
    )

    return result


# 5. Enrich training and testing data
print("\nMerging training data...")
train_merged = enrich_sales_data(train)

print("Merging testing data...")
test_merged = enrich_sales_data(test)

# 6. Validate row counts
print("\nROW COUNT VALIDATION")

print("Original training rows:", len(train))
print("Merged training rows:", len(train_merged))

print("Original testing rows:", len(test))
print("Merged testing rows:", len(test_merged))

# 7. Check missing values in important columns
print("\nTRAINING MERGE MISSING VALUES")

print(
    train_merged[
        ["city", "dcoilwtico", "transactions"]
    ].isnull().sum()
)

print("\nTESTING MERGE MISSING VALUES")

print(
    test_merged[
        ["city", "dcoilwtico", "transactions"]
    ].isnull().sum()
)

# 8. Check whether IDs remain unique
print("\nID VALIDATION")

print(
    "Training duplicate IDs:",
    train_merged["id"].duplicated().sum()
)

print(
    "Testing duplicate IDs:",
    test_merged["id"].duplicated().sum()
)

# 9. Save enriched datasets
train_merged.to_csv(
    processed_folder / "train_merged.csv",
    index=False
)

test_merged.to_csv(
    processed_folder / "test_merged.csv",
    index=False
)

print("\nMerged datasets saved successfully.")

print("\nTraining columns:")
print(train_merged.columns.tolist())

print("\nOutput folder:", processed_folder)