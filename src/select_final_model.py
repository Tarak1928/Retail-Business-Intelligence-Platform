
import pandas as pd
import lightgbm as lgb
import json
import shutil

from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RESULT_PATH = REPORT_DIR / "hyperparameter_results.csv"

FINAL_MODEL_PATH = MODEL_DIR / "final_lightgbm_model.txt"

SUMMARY_PATH = REPORT_DIR / "final_model_summary.json"

# --------------------------------------------------
# 2. Load experiment results
# --------------------------------------------------

print("Loading hyperparameter experiment results...")

results = pd.read_csv(RESULT_PATH)

print("\nExperiment results:")
print(
    results[
        ["Experiment", "MAE", "RMSE", "RMSLE"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 3. Select the best model
# --------------------------------------------------

print("\nSelecting the best model...")

best_model = results.loc[
    results["RMSLE"].idxmin()
]

best_experiment = best_model["Experiment"]

print("Selected experiment:", best_experiment)
print("Best RMSLE:", round(best_model["RMSLE"], 4))

# --------------------------------------------------
# 4. Locate the selected model
# --------------------------------------------------

source_model_path = (
    MODEL_DIR / f"{best_experiment}_model.txt"
)

if not source_model_path.exists():
    raise FileNotFoundError(
        f"Model file not found: {source_model_path}"
    )

# --------------------------------------------------
# 5. Validate the selected model
# --------------------------------------------------

print("\nValidating selected model...")

model = lgb.Booster(
    model_file=str(source_model_path)
)

print("Model loaded successfully!")
print("Number of features:", model.num_feature())
print("Number of boosting iterations:", model.current_iteration())

# --------------------------------------------------
# 6. Save final model
# --------------------------------------------------

shutil.copy2(
    source_model_path,
    FINAL_MODEL_PATH
)

print("\nFinal model saved successfully!")
print(FINAL_MODEL_PATH)

# --------------------------------------------------
# 7. Save model configuration and metrics
# --------------------------------------------------

summary = {
    "model_name": "Final LightGBM Sales Forecasting Model",
    "selected_experiment": best_experiment,
    "model_file": "final_lightgbm_model.txt",
    "objective": "regression",
    "target_transformation": "log1p",
    "prediction_inverse_transformation": "expm1",
    "validation_start": "2017-07-31",
    "validation_end": "2017-08-15",
    "validation_metrics": {
        "MAE": float(best_model["MAE"]),
        "RMSE": float(best_model["RMSE"]),
        "RMSLE": float(best_model["RMSLE"])
    },
    "hyperparameters": {
        "n_estimators": int(best_model["n_estimators"]),
        "learning_rate": float(best_model["learning_rate"]),
        "num_leaves": int(best_model["num_leaves"]),
        "max_depth": int(best_model["max_depth"]),
        "min_child_samples": int(best_model["min_child_samples"]),
        "reg_lambda": float(best_model["reg_lambda"])
    },
    "number_of_features": model.num_feature()
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=4
    )

print("\nModel summary saved:")
print(SUMMARY_PATH)

# --------------------------------------------------
# 8. Final confirmation
# --------------------------------------------------

print("\nFINAL MODEL SUMMARY")
print("-------------------")

print("Selected model:", best_experiment)
print("MAE:", summary["validation_metrics"]["MAE"])
print("RMSE:", summary["validation_metrics"]["RMSE"])
print("RMSLE:", summary["validation_metrics"]["RMSLE"])

print("\nFinal model selection completed!")
