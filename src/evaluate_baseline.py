
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data/processed/train_period.csv"
VALIDATION_PATH = ROOT / "data/processed/validation_period.csv"

OUTPUT_PATH = ROOT / "reports/baseline_predictions.csv"

VALIDATION_START = pd.Timestamp("2017-07-31")
VALIDATION_END = pd.Timestamp("2017-08-15")

# --------------------------------------------------
# 2. Identify product-family columns
# --------------------------------------------------

columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()

family_columns = [
    column for column in columns
    if column.startswith("family_")
]

required_columns = ["date", "store_nbr", "sales"] + family_columns

print("Product-family columns:", len(family_columns))

# --------------------------------------------------
# 3. Load the final 7 training days
# --------------------------------------------------

print("\nLoading historical sales...")

history_data = []

for chunk in pd.read_csv(
    TRAIN_PATH,
    usecols=required_columns,
    chunksize=100_000
):
    chunk["date"] = pd.to_datetime(chunk["date"])

    chunk = chunk[
        chunk["date"] >= VALIDATION_START - pd.Timedelta(days=7)
    ]

    if not chunk.empty:
        history_data.append(chunk)

history_df = pd.concat(history_data, ignore_index=True)

history_df["family"] = (
    history_df[family_columns]
    .idxmax(axis=1)
    .str.replace("family_", "", regex=False)
)

# Store previous known sales by store, family and date
history = {}

for row in history_df[
    ["date", "store_nbr", "family", "sales"]
].itertuples(index=False):

    key = (row.store_nbr, row.family, row.date)

    history[key] = row.sales

print("Historical records loaded:", len(history_df))

# --------------------------------------------------
# 4. Load validation data
# --------------------------------------------------

print("\nLoading validation data...")

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
# 5. Recursive seasonal naive forecasting
# --------------------------------------------------

print("\nGenerating baseline predictions...")

predictions = []

for current_date in pd.date_range(
    VALIDATION_START,
    VALIDATION_END,
    freq="D"
):

    daily_data = validation[
        validation["date"] == current_date
    ]

    for row in daily_data.itertuples(index=False):

        previous_date = current_date - pd.Timedelta(days=7)

        key = (
            row.store_nbr,
            row.family,
            previous_date
        )

        if key not in history:
            raise ValueError(
                f"Missing historical sales for {key}"
            )

        prediction = history[key]

        predictions.append(prediction)

        # Save the prediction as the known value
        # for future recursive forecasts.
        current_key = (
            row.store_nbr,
            row.family,
            current_date
        )

        history[current_key] = prediction

# --------------------------------------------------
# 6. Evaluate forecasting performance
# --------------------------------------------------

actual = validation["sales"].to_numpy()
predicted = np.array(predictions)

mae = np.mean(np.abs(actual - predicted))

rmse = np.sqrt(
    np.mean((actual - predicted) ** 2)
)

rmsle = np.sqrt(
    np.mean(
        (
            np.log1p(actual) - np.log1p(predicted)
        ) ** 2
    )
)

print("\nBASELINE MODEL RESULTS")
print("----------------------")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")

# --------------------------------------------------
# 7. Save predictions
# --------------------------------------------------

results = validation[
    ["id", "date", "store_nbr", "sales"]
].copy()

results["predicted_sales"] = predicted

results.to_csv(OUTPUT_PATH, index=False)

print("\nPredictions saved to:", OUTPUT_PATH)

print("\nBaseline forecasting completed!")
