
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error
)

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

RESULT_PATH = REPORT_DIR / "hyperparameter_results.csv"

VALIDATION_START = pd.Timestamp("2017-07-31")
VALIDATION_END = pd.Timestamp("2017-08-15")

BASELINE_RMSLE = 0.4484

# --------------------------------------------------
# 2. Define candidate configurations
# --------------------------------------------------

experiments = {
    "Experiment_A": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_lambda": 1.0
    },

    "Experiment_B": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 10,
        "min_child_samples": 30,
        "reg_lambda": 2.0
    },

    "Experiment_C": {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 40,
        "max_depth": 12,
        "min_child_samples": 40,
        "reg_lambda": 3.0
    }
}

# --------------------------------------------------
# 3. Identify model features
# --------------------------------------------------

print("Reading dataset columns...")

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
# 4. Load training data
# --------------------------------------------------

print("\nLoading training data...")

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

X_train = train[feature_columns].copy()
X_train.columns = safe_feature_names

y_train = np.log1p(train["sales"])

print("Training shape:", X_train.shape)

# --------------------------------------------------
# 5. Load validation data
# --------------------------------------------------

print("\nLoading validation data...")

validation = pd.read_csv(VALIDATION_PATH)

validation["date"] = pd.to_datetime(
    validation["date"]
)

validation["family"] = (
    validation[family_columns]
    .idxmax(axis=1)
    .str.replace("family_", "", regex=False)
)

validation = validation.sort_values(
    ["date", "store_nbr", "family"]
).reset_index(drop=True)

actual_sales = validation["sales"].to_numpy(
    dtype=np.float64
)

# --------------------------------------------------
# 6. Load historical sales
# --------------------------------------------------

print("\nLoading historical sales...")

history_columns = [
    "date",
    "store_nbr",
    "sales"
] + family_columns

base_history = {}

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

        base_history[key] = row.sales

print("Historical records loaded:", len(base_history))

# --------------------------------------------------
# 7. Recursive forecasting function
# --------------------------------------------------

def generate_predictions(model):

    history = base_history.copy()

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

        daily["sales_lag_1"] = lag_1
        daily["sales_lag_7"] = lag_7
        daily["sales_lag_14"] = lag_14
        daily["sales_lag_30"] = lag_30
        daily["sales_rolling_mean_7"] = rolling_7

        X_day = daily[feature_columns].copy()
        X_day.columns = safe_feature_names

        log_predictions = model.predict(X_day)

        predictions = np.expm1(log_predictions)

        predictions = np.maximum(
            predictions,
            0
        )

        all_predictions.extend(predictions)

        # Store predictions, not actual validation sales.
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
            f"  Forecast completed: {current_date.date()}"
        )

    return np.array(all_predictions)

# --------------------------------------------------
# 8. Train and evaluate experiments
# --------------------------------------------------

results = []

for experiment_name, parameters in experiments.items():

    print("\n" + "=" * 55)
    print("RUNNING:", experiment_name)
    print("=" * 55)

    model = lgb.LGBMRegressor(
        objective="regression",
        max_bin=127,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **parameters
    )

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    print("\nGenerating recursive forecasts...")

    predictions = generate_predictions(model)

    mae = mean_absolute_error(
        actual_sales,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_sales,
            predictions
        )
    )

    rmsle = np.sqrt(
        mean_squared_log_error(
            actual_sales,
            predictions
        )
    )

    improvement = (
        (BASELINE_RMSLE - rmsle)
        / BASELINE_RMSLE
    ) * 100

    print("\nEXPERIMENT RESULTS")
    print("------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RMSLE: {rmsle:.4f}")
    print(f"Improvement over baseline: {improvement:.2f}%")

    model_path = (
        MODEL_DIR / f"{experiment_name}_model.txt"
    )

    model.booster_.save_model(
        str(model_path)
    )

    results.append({
        "Experiment": experiment_name,
        **parameters,
        "MAE": mae,
        "RMSE": rmse,
        "RMSLE": rmsle,
        "Baseline_RMSLE": BASELINE_RMSLE,
        "Improvement_Percent": improvement
    })

    del model, predictions
    gc.collect()

# --------------------------------------------------
# 9. Compare experiments
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSLE",
    ascending=True
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\n" + "=" * 55)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 55)

print(
    results_df[
        [
            "Experiment",
            "MAE",
            "RMSE",
            "RMSLE",
            "Improvement_Percent"
        ]
    ].to_string(index=False)
)

best_experiment = results_df.iloc[0]

print("\nBEST EXPERIMENT:", best_experiment["Experiment"])
print("Best RMSLE:", round(best_experiment["RMSLE"], 4))

print("\nResults saved to:", RESULT_PATH)

print("\nHyperparameter tuning completed!")
