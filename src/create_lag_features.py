
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# STEP 1: PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# STEP 2: LOAD FEATURE DATASETS
# --------------------------------------------------

print("Loading training features...")
train_df = pd.read_csv(DATA_DIR / "train_features.csv")

print("Loading testing features...")
test_df = pd.read_csv(DATA_DIR / "test_features.csv")


# --------------------------------------------------
# STEP 3: PREPARE THE DATA
# --------------------------------------------------

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

# Mark which rows belong to the test dataset.
train_df["_is_test"] = False
test_df["_is_test"] = True

# Test data does not contain actual sales.
# Add an empty sales column so both datasets can be combined.
test_df["sales"] = float("nan")

# Combine training and testing data.
combined_df = pd.concat(
    [train_df, test_df],
    ignore_index=True,
    sort=False
)

# Sort each store and product family chronologically.
combined_df = combined_df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)


# --------------------------------------------------
# STEP 4: CHECK FOR DUPLICATE STORE/FAMILY/DATES
# --------------------------------------------------

duplicate_count = combined_df.duplicated(
    subset=["store_nbr", "family", "date"]
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Found {duplicate_count} duplicate store/family/date records."
    )


# --------------------------------------------------
# STEP 5: CREATE LAG FEATURES
# --------------------------------------------------

group_columns = ["store_nbr", "family"]

print("\nCreating lag features...")

grouped_sales = combined_df.groupby(
    group_columns,
    sort=False
)["sales"]

# Sales from 1 day earlier
combined_df["sales_lag_1"] = grouped_sales.shift(1)

# Sales from 7 days earlier
combined_df["sales_lag_7"] = grouped_sales.shift(7)

# Sales from 14 days earlier
combined_df["sales_lag_14"] = grouped_sales.shift(14)

# Sales from 30 days earlier
combined_df["sales_lag_30"] = grouped_sales.shift(30)


# --------------------------------------------------
# STEP 6: CREATE 7-DAY ROLLING AVERAGE
# --------------------------------------------------

print("Creating rolling average feature...")

combined_df["sales_rolling_mean_7"] = (
    combined_df
    .groupby(group_columns, sort=False)["sales"]
    .transform(
        lambda values: (
            values.shift(1)
            .rolling(window=7, min_periods=7)
            .mean()
        )
    )
)


# --------------------------------------------------
# STEP 7: SEPARATE TRAINING AND TEST DATA
# --------------------------------------------------

train_lag_features = combined_df[
    combined_df["_is_test"] == False
].copy()

test_lag_features = combined_df[
    combined_df["_is_test"] == True
].copy()

# Remove the temporary marker.
train_lag_features.drop(columns=["_is_test"], inplace=True)
test_lag_features.drop(columns=["_is_test", "sales"], inplace=True)


# --------------------------------------------------
# STEP 8: SAVE OUTPUT FILES
# --------------------------------------------------

train_output = DATA_DIR / "train_lag_features.csv"
test_output = DATA_DIR / "test_lag_features.csv"

print("\nSaving training lag features...")
train_lag_features.to_csv(train_output, index=False)

print("Saving testing lag features...")
test_lag_features.to_csv(test_output, index=False)


# --------------------------------------------------
# STEP 9: DISPLAY RESULTS
# --------------------------------------------------

lag_columns = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_30",
    "sales_rolling_mean_7"
]

print("\n" + "=" * 60)
print("LAG FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nTraining dataset shape:", train_lag_features.shape)
print("Testing dataset shape:", test_lag_features.shape)

print("\nMissing lag values in training dataset:")
print(train_lag_features[lag_columns].isnull().sum())

print("\nMissing lag values in testing dataset:")
print(test_lag_features[lag_columns].isnull().sum())

print("\nSample training lag features:")
print(
    train_lag_features[
        ["date", "store_nbr", "family", "sales"] + lag_columns
    ].head(10).to_string(index=False)
)

print("\nSaved files:")
print(train_output)
print(test_output)

print("\nSTATUS: SUCCESS")
