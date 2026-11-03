
import gc
import json
import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data/processed/train_encoded.csv"
TEST_PATH = ROOT / "data/processed/test_encoded.csv"

SAMPLE_PATH = (
    ROOT / "data/raw/store-sales-kaggle/sample_submission.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "final_lightgbm_model.txt"

PREDICTION_PATH = REPORT_DIR / "test_forecast_predictions.csv"

SUBMISSION_PATH = ROOT / "submission.csv"

SUMMARY_PATH = REPORT_DIR / "test_forecast_summary.json"

TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

# --------------------------------------------------
# 2. Identify model features
# --------------------------------------------------

print("Reading training dataset columns...")

all_columns = pd.read_csv(
    TRAIN_PATH,
    nrows=0
).columns.tolist()

family_columns = [
    column for column in all_columns
    if column.startswith("family_")
]

feature_columns = [
    column for column in all_columns
    if column not in [
        "id",
        "date",
        "sales",
        "dcoilwtico"
    ]
]

safe_feature_names = [
    f"feature_{i}"
    for i in range(len(feature_columns))
]

print("Total model features:", len(feature_columns))

# --------------------------------------------------
# 3. Load complete training dataset
# --------------------------------------------------

print("\nLoading complete training dataset...")

dtype_map = {
    column: (
        "uint8"
        if column.startswith(
            ("family_", "city_", "state_", "type_")
        )
        else "float32"
    )
    for column in feature_columns + ["sales"]
}

train = pd.read_csv(
    TRAIN_PATH,
    usecols=feature_columns + ["sales"],
    dtype=dtype_map
)

print("Training shape:", train.shape)

X_train = train[feature_columns].copy()

X_train.columns = safe_feature_names

y_train = np.log1p(train["sales"])

print("Training features:", X_train.shape)

# --------------------------------------------------
# 4. Train final LightGBM model
# --------------------------------------------------

print("\nTraining final LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=10,
    min_child_samples=30,
    reg_lambda=2.0,
    max_bin=127,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(
    X_train,
    y_train
)

model.booster_.save_model(
    str(FINAL_MODEL_PATH)
)

print("\nFinal model trained and saved!")
print(FINAL_MODEL_PATH)

del train, X_train, y_train
gc.collect()

# --------------------------------------------------
# 5. Load Kaggle test dataset
# --------------------------------------------------

print("\nLoading Kaggle test dataset...")

test = pd.read_csv(TEST_PATH)

test["date"] = pd.to_datetime(test["date"])

if len(test) != 28512:
    raise ValueError(
        f"Unexpected test row count: {len(test)}"
    )

# Verify model feature alignment.
missing_features = [
    column for column in feature_columns
    if column not in test.columns
]

if missing_features:
    raise ValueError(
        f"Missing test features: {missing_features}"
    )

test["family"] = (
    test[family_columns]
    .idxmax(axis=1)
    .str.replace("family_", "", regex=False)
)

test = test.sort_values(
    ["date", "store_nbr", "family"]
).reset_index(drop=True)

print("Test shape:", test.shape)
print("Test start:", test["date"].min().date())
print("Test end:", test["date"].max().date())

# --------------------------------------------------
# 6. Load historical sales
# --------------------------------------------------

print("\nLoading historical sales...")

history_columns = [
    "date",
    "store_nbr",
    "sales"
] + family_columns

history = {}

for chunk in pd.read_csv(
    TRAIN_PATH,
    usecols=history_columns,
    chunksize=100_000
):

    chunk["date"] = pd.to_datetime(chunk["date"])

    chunk = chunk[
        chunk["date"] >=
        TEST_START - pd.Timedelta(days=30)
    ]

    if chunk.empty:
        continue

    chunk["family"] = (
        chunk[family_columns]
        .idxmax(axis=1)
        .str.replace("family_", "", regex=False)
    )

    for row in chunk[
        ["date", "store_nbr", "family", "sales"]
    ].itertuples(index=False):

        key = (
            row.store_nbr,
            row.family,
            row.date
        )

        history[key] = row.sales

print("Historical records loaded:", len(history))

# --------------------------------------------------
# 7. Recursive test forecasting
# --------------------------------------------------

print("\nGenerating future sales forecasts...")

all_predictions = []

for current_date in pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
):

    daily = test[
        test["date"] == current_date
    ].copy()

    lag_1 = []
    lag_7 = []
    lag_14 = []
    lag_30 = []
    rolling_7 = []

    for row in daily.itertuples(index=False):

        store = row.store_nbr
        family = row.family

        def get_sales(days):

            key = (
                store,
                family,
                current_date - pd.Timedelta(days=days)
            )

            if key not in history:
                raise ValueError(
                    f"Missing historical sales: {key}"
                )

            return history[key]

        lag_1.append(get_sales(1))
        lag_7.append(get_sales(7))
        lag_14.append(get_sales(14))
        lag_30.append(get_sales(30))

        previous_seven_days = [
            get_sales(day)
            for day in range(1, 8)
        ]

        rolling_7.append(
            np.mean(previous_seven_days)
        )

    # Update lag features.
    daily["sales_lag_1"] = lag_1
    daily["sales_lag_7"] = lag_7
    daily["sales_lag_14"] = lag_14
    daily["sales_lag_30"] = lag_30
    daily["sales_rolling_mean_7"] = rolling_7

    # Prepare model features.
    X_day = daily[feature_columns].copy()

    X_day.columns = safe_feature_names

    # Predict log-transformed sales.
    log_predictions = model.predict(X_day)

    # Convert predictions to original sales scale.
    predictions = np.expm1(log_predictions)

    predictions = np.maximum(
        predictions,
        0
    )

    all_predictions.extend(
        zip(daily["id"].tolist(), predictions.tolist())
    )

    # Store predictions for future lag calculations.
    for row, prediction in zip(
        daily.itertuples(index=False),
        predictions
    ):

        key = (
            row.store_nbr,
            row.family,
            current_date
        )

        history[key] = prediction

    print(
        f"Completed forecast: {current_date.date()}"
    )

# --------------------------------------------------
# 8. Create prediction dataframe
# --------------------------------------------------

print("\nCreating prediction report...")

predictions_df = pd.DataFrame(
    all_predictions,
    columns=["id", "sales"]
)

if predictions_df["id"].duplicated().any():
    raise ValueError("Duplicate prediction IDs found!")

if not np.isfinite(predictions_df["sales"]).all():
    raise ValueError("Invalid prediction values found!")

if len(predictions_df) != len(test):
    raise ValueError("Prediction row count mismatch!")

predictions_df.to_csv(
    PREDICTION_PATH,
    index=False
)

print("Prediction report saved:", PREDICTION_PATH)

# --------------------------------------------------
# 9. Create Kaggle submission file
# --------------------------------------------------

print("\nCreating Kaggle submission file...")

sample_submission = pd.read_csv(SAMPLE_PATH)

submission = sample_submission[["id"]].merge(
    predictions_df,
    on="id",
    how="left",
    validate="one_to_one"
)

if submission["sales"].isna().any():
    raise ValueError(
        "Submission contains missing predictions!"
    )

submission["sales"] = submission["sales"].clip(lower=0)

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)

print("Submission saved:", SUBMISSION_PATH)

# --------------------------------------------------
# 10. Save forecast summary
# --------------------------------------------------

summary = {
    "model": "Final LightGBM",
    "training_records": 3000888,
    "features": len(feature_columns),
    "test_records": len(submission),
    "forecast_start": str(TEST_START.date()),
    "forecast_end": str(TEST_END.date()),
    "forecast_days": 16,
    "validation_rmsle": 0.4335676764,
    "prediction_mean": float(submission["sales"].mean()),
    "prediction_total": float(submission["sales"].sum()),
    "prediction_minimum": float(submission["sales"].min()),
    "prediction_maximum": float(submission["sales"].max()),
    "test_rmsle": None,
    "test_rmsle_note": "Actual test sales are unavailable."
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print("\nFINAL FORECAST SUMMARY")
print("----------------------")

print("Test records:", len(submission))
print("Forecast days:", summary["forecast_days"])
print("Average predicted sales:", round(summary["prediction_mean"], 2))
print("Total predicted sales:", round(summary["prediction_total"], 2))
print("Minimum prediction:", round(summary["prediction_minimum"], 2))
print("Maximum prediction:", round(summary["prediction_maximum"], 2))

print("\nAll forecasting tasks completed successfully!")
