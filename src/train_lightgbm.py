
import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data/processed/train_period.csv"
VALIDATION_PATH = ROOT / "data/processed/validation_period.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_sales_model.txt"
PREDICTION_PATH = REPORT_DIR / "lightgbm_predictions.csv"

VALIDATION_START = pd.Timestamp("2017-07-31")
VALIDATION_END = pd.Timestamp("2017-08-15")

# --------------------------------------------------
# 2. Identify model features
# --------------------------------------------------

print("Reading dataset column names...")

all_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()

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

print("Number of model features:", len(feature_columns))

# Use smaller data types to reduce memory usage.
dtype_map = {
    column: (
        "uint8" if column in family_columns
        or column.startswith(("city_", "state_", "type_"))
        else "float32"
    )
    for column in feature_columns + ["sales"]
}

# --------------------------------------------------
# 3. Load training dataset
# --------------------------------------------------

print("\nLoading training dataset...")

train = pd.read_csv(
    TRAIN_PATH,
    usecols=feature_columns + ["sales"],
    dtype=dtype_map
)

print("Training shape:", train.shape)

# Create safe feature names for LightGBM.
original_feature_names = feature_columns.copy()

safe_feature_names = [
    f"feature_{i}"
    for i in range(len(feature_columns))
]

X_train = train[feature_columns].copy()

# Rename columns to avoid special-character errors.
X_train.columns = safe_feature_names

# Apply log transformation to the sales target
y_train = np.log1p(train["sales"])

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

# --------------------------------------------------
# 4. Train LightGBM
# --------------------------------------------------

print("\nTraining LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_bin=127,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(X_train, y_train)

print("\nLightGBM training completed!")

# Save trained model.
model.booster_.save_model(str(MODEL_PATH))

print("Model saved to:", MODEL_PATH)

# Release training data before validation.
del train, X_train, y_train

# --------------------------------------------------
# 5. Load validation data
# --------------------------------------------------

print("\nLoading validation dataset...")

validation = pd.read_csv(VALIDATION_PATH)

validation["date"] = pd.to_datetime(validation["date"])

validation["family"] = (
    validation[family_columns]
    .idxmax(axis=1)
    .str.replace("family_", "", regex=False)
)

validation = validation.sort_values(
    ["date", "store_nbr", "family"]
).reset_index(drop=True)

# --------------------------------------------------
# 6. Load historical sales for recursive forecasting
# --------------------------------------------------

print("\nLoading historical sales for lag features...")

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
        VALIDATION_START - pd.Timedelta(days=30)
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
# 7. Recursive validation forecasting
# --------------------------------------------------

# --------------------------------------------------
# 7. Recursive validation forecasting
# --------------------------------------------------

print("\nGenerating recursive predictions...")

all_predictions = []

for current_date in pd.date_range(
    VALIDATION_START,
    VALIDATION_END,
    freq="D"
):

    daily = validation[
        validation["date"] == current_date
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
                    f"Missing history for {key}"
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

    daily["sales_lag_1"] = lag_1
    daily["sales_lag_7"] = lag_7
    daily["sales_lag_14"] = lag_14
    daily["sales_lag_30"] = lag_30
    daily["sales_rolling_mean_7"] = rolling_7

    # Prepare validation features
    X_day = daily[original_feature_names].copy()
    X_day.columns = safe_feature_names

    # Predict log-transformed sales
    log_predictions = model.predict(X_day)

    # Convert predictions back to normal sales
    predictions = np.expm1(log_predictions)

    # Sales cannot be negative
    predictions = np.maximum(predictions, 0)

    all_predictions.extend(predictions)

    # Store predictions for recursive forecasting
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
        f"Completed forecast for {current_date.date()}"
    )

# --------------------------------------------------
# 8. Calculate evaluation metrics
# --------------------------------------------------

actual = validation["sales"].to_numpy()

predicted = np.array(all_predictions)

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    np.mean((actual - predicted) ** 2)
)

rmsle = np.sqrt(
    np.mean(
        (np.log1p(actual) - np.log1p(predicted)) ** 2
    )
)

print("\nLIGHTGBM MODEL RESULTS")
print("----------------------")

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")

# --------------------------------------------------
# 9. Compare with baseline
# --------------------------------------------------

baseline_path = REPORT_DIR / "baseline_predictions.csv"

baseline = pd.read_csv(baseline_path)

baseline_actual = baseline["sales"].to_numpy()
baseline_predictions = baseline["predicted_sales"].to_numpy()

baseline_rmsle = np.sqrt(
    np.mean(
        (
            np.log1p(baseline_actual)
            - np.log1p(baseline_predictions)
        ) ** 2
    )
)

improvement = (
    (baseline_rmsle - rmsle) / baseline_rmsle
) * 100

print("\nMODEL COMPARISON")
print("----------------")

print(f"Baseline RMSLE: {baseline_rmsle:.4f}")
print(f"LightGBM RMSLE: {rmsle:.4f}")
print(f"Improvement:    {improvement:.2f}%")

# --------------------------------------------------
# 10. Save predictions
# --------------------------------------------------

results = validation[
    ["id", "date", "store_nbr", "family", "sales"]
].copy()

results["predicted_sales"] = predicted

results.to_csv(
    PREDICTION_PATH,
    index=False
)

print("\nPredictions saved to:", PREDICTION_PATH)
print("LightGBM forecasting completed!")
