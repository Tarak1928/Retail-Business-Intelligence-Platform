
from pathlib import Path
import pandas as pd


# STEP 1: FIND PROJECT DIRECTORIES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# STEP 2: LOAD TRAINING AND TEST DATA

train_path = DATA_DIR / "train_model_ready.csv"
test_path = DATA_DIR / "test_model_ready.csv"

print("Loading training dataset...")
train_df = pd.read_csv(train_path)

print("Loading testing dataset...")
test_df = pd.read_csv(test_path)


# STEP 3: CONVERT DATE COLUMN

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


# STEP 4: CREATE DATE FEATURES

def create_date_features(df):

    df = df.copy()

    # Extract year
    df["year"] = df["date"].dt.year

    # Extract month
    df["month"] = df["date"].dt.month

    # Extract day
    df["day"] = df["date"].dt.day

    # Extract weekday number (Monday = 0, Sunday = 6)
    df["day_of_week"] = df["date"].dt.day_of_week

    # Extract quarter
    df["quarter"] = df["date"].dt.quarter

    # Extract day of year
    df["day_of_year"] = df["date"].dt.day_of_year

    # Check whether the date is Saturday or Sunday
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Check whether the date is the first day of a month
    df["is_month_start"] = (
        df["date"].dt.is_month_start
    ).astype(int)

    # Check whether the date is the last day of a month
    df["is_month_end"] = (
        df["date"].dt.is_month_end
    ).astype(int)

    return df


# STEP 5: APPLY FEATURES TO BOTH DATASETS

print("Creating training date features...")
train_features = create_date_features(train_df)

print("Creating testing date features...")
test_features = create_date_features(test_df)


# STEP 6: SAVE FEATURE DATASETS

train_output = DATA_DIR / "train_features.csv"
test_output = DATA_DIR / "test_features.csv"

train_features.to_csv(train_output, index=False)
test_features.to_csv(test_output, index=False)


# STEP 7: DISPLAY RESULTS

print("\n" + "=" * 55)
print("DATE FEATURE ENGINEERING COMPLETED")
print("=" * 55)

print("\nTraining dataset shape:", train_features.shape)
print("Testing dataset shape:", test_features.shape)

print("\nNew date features:")
print([
    "year",
    "month",
    "day",
    "day_of_week",
    "quarter",
    "day_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end"
])

print("\nMissing values in new training features:")
print(train_features[
    [
        "year", "month", "day", "day_of_week",
        "quarter", "day_of_year", "is_weekend",
        "is_month_start", "is_month_end"
    ]
].isnull().sum())

print("\nSaved files:")
print(train_output)
print(test_output)
